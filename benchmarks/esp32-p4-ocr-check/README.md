# ESP32-P4: InviolusOCR speed and on-device output check

Runs three InviolusOCR int8 files on an ESP32-P4 with TensorFlow Lite Micro:

1. **Speed and arena use** with pseudo-random input (20 timed runs after 2 warm-up runs).
2. **Real input:** 240 real photos from the TS-TR test split (every 8th line), the same
   int8 tensors the PC interpreter gets. The device prints the most likely class per
   timestep; `compare.py` decodes it and compares it with the PC TFLite interpreter and
   with the ground truth.

Measured 2026-11-08 on one ESP32-P4 board (rev v1.3, 360 MHz, 32 MB PSRAM at 200 MHz),
ESP-IDF 5.3.5, `esp-tflite-micro` 1.4.2.

| Model | File | Arena | Latency, internal SRAM | Exact lines, PC | Exact lines, device | Same text on both |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 171 KB | 172 KB | 419 ms | 81 / 240 (33.8%) | 84 / 240 (35.0%) | 180 / 240 |
| P2 (pruned, candidate) | 298 KB | 235 KB | 760 ms | 114 / 240 (47.5%) | 110 / 240 (45.8%) | 201 / 240 |
| D | 598 KB | 339 KB | 1404 ms | 136 / 240 (56.7%) | 133 / 240 (55.4%) | 202 / 240 |

What this shows:

- **Accuracy holds on the device.** Device and PC differ by at most 1.7 points on these
  240 lines, in both directions, which is within the noise of a 240-line sample.
- **The output is not bit-identical.** On 16-25% of lines the decoded text differs. The
  differences are on lines the model is unsure about; the device's int8 kernels round
  differently from the PC's. Do not expect identical strings from device and PC.
- **P2 meets the 1 s per line target at 760 ms; D does not (1404 ms).**
- tiny and D ran 2 to 3% slower than in the 2026-11-06 run (413 and 1365 ms) with this
  larger firmware build.

Not measured: image decoding, resizing and CTC decoding time; whether ESP-NN kernels are
active; other boards. One board.

## Files

| File | What |
|---|---|
| `main/main.cc` | Firmware: latency of each model, then the 240-line check |
| `make_inputs.py` | Builds `main/ocr_in.tflite` (240 int8 input tensors) and `reference.json` (PC outputs and the true texts; not committed, because TS-TR is not redistributed here); run from `ocr/scripts` |
| `compare.py` | Decodes the device output and compares it with `reference.json` |
| `results.txt` | Raw latency lines of the run |
| `check.json` | Output of `compare.py` for the run |
| `partitions.csv`, `sdkconfig.defaults` | A 3 MB app partition, because three models and the inputs do not fit the 1.5 MB default |

## Run it

1. Put the int8 files into `main/` as `ocr_t.tflite` (tiny), `ocr_d.tflite` (D) and
   `ocr_p2.tflite` (P2), and run `make_inputs.py <path to main/>` from `ocr/scripts`.
2. With ESP-IDF 5.3: `idf.py set-target esp32p4 && idf.py build && idf.py -p COMx flash`.
3. `python readserial.py 1200` (writes `serial.log`), then
   `python compare.py serial.log reference.json ../../artifacts/alphabet.json`.
