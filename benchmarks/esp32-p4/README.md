# ESP32-P4 benchmark

Runs InviolusOCR and VoxHOD on an ESP32-P4 with TensorFlow Lite Micro and prints
latency and arena use over the serial port. It uses random int8 input, so it
measures **speed and memory only, not accuracy**.

Measured 2026-11-06 on one ESP32-P4 board (rev v1.3, 360 MHz, 32 MB flash,
32 MB PSRAM) with ESP-IDF 5.3.5 and `esp-tflite-micro` 1.4.2. `results.txt` is
the raw output of that run (arena in internal SRAM and in PSRAM, 20 timed runs
after 2 warm-up runs).

## Run it

1. Copy these four files into `main/` under the names below:

   | Name in `main/` | File |
   |---|---|
   | `ocr_t.tflite` | InviolusOCR tiny, `inviolusocr_tiny_int8.tflite` on Hugging Face |
   | `ocr_d.tflite` | InviolusOCR D, `inviolusocr_int8.tflite` on Hugging Face |
   | `vox_v3.tflite` | VoxHOD v3, `voxhod_qat_int8.tflite` (the published files were checked byte for byte) |
   | `vox_nano.tflite` | VoxHOD nano, `voxhod_nano_int8.tflite` |

2. With ESP-IDF 5.3 active: `idf.py set-target esp32p4 && idf.py build && idf.py -p COMx flash`.
3. `python readserial.py 120` resets the board and prints the `RESULT` lines
   (needs `pyserial`; edit the port in the script).

`sdkconfig.defaults` sets 32 MB flash, the large single-app partition, PSRAM at
200 MHz (needs `CONFIG_IDF_EXPERIMENTAL_FEATURES` in ESP-IDF 5.3) and the
CPU frequency. This ESP-IDF version runs a rev v1.3 chip at 360 MHz.

## What it does not show

- The output was not compared with the PC interpreter, so correctness on the
  device is unchecked.
- Only the model inference is timed: no image decoding, resizing, CTC decoding
  or box extraction.
- Whether ESP-NN's accelerated kernels are active for the P4 was not checked.
  The speed may improve.
- One board. These numbers do not carry over to an ESP32-S3.
