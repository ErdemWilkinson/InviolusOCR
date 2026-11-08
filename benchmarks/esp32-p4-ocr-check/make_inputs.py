"""240 real TS-TR test lines -> int8 input blob for the ESP32-P4 firmware, plus PC TFLite reference outputs."""
import json, sys
from pathlib import Path
import numpy as np, tensorflow as tf
sys.path.insert(0, str(Path.cwd()))
from benchmark import read_csv, load_image, ROOT
rows = read_csv(ROOT / "data" / "tstr_test_labels.csv")[::8][:240]  # 240 lines spread over the test split
images = np.stack([load_image(Path(r["base"]) / r["image"]) for r in rows])
models = {"tiny": ROOT / "artifacts_T_1102/erdem_ocr_int8.tflite", "D": ROOT / "artifacts_D_1101/erdem_ocr_int8.tflite",
          "P2": ROOT / "artifacts_P2_1107/erdem_ocr_int8.tflite"}
blob, ref = None, {"texts": [r["text"] for r in rows]}
for name, path in models.items():
    it = tf.lite.Interpreter(model_content=path.read_bytes()); it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, z = inp["quantization"]
    q = np.clip(np.round(images / s + z), -128, 127).astype(np.int8)
    if blob is None: blob = q
    assert np.array_equal(blob, q), "models use different input quantization"
    ids = []
    for x in q:
        it.set_tensor(inp["index"], x[None]); it.invoke()
        ids.append(it.get_tensor(out["index"])[0].argmax(-1).tolist())
    ref[name] = ids
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("main"); (out_dir / "ocr_in.tflite").write_bytes(blob.tobytes())
(out_dir.parent / "reference.json").write_text(json.dumps(ref))
import shutil; shutil.copy(models["P2"], out_dir / "ocr_p2.tflite")
print("inputs", blob.shape, blob.nbytes, "bytes; reference saved")
