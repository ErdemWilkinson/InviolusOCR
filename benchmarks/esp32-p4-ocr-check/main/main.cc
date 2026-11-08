// Benchmark: run each TFLite model on the chip and report latency and arena use.
// Latency uses pseudo-random int8 input. check() then runs 24 real TS-TR test lines and prints the
// argmax per timestep, to be compared with the PC TFLite interpreter (p4_reference.json).
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <new>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#define EMB(n) extern const uint8_t n##_start[] asm("_binary_" #n "_tflite_start"); \
               extern const uint8_t n##_end[]   asm("_binary_" #n "_tflite_end");
EMB(ocr_t) EMB(ocr_d) EMB(ocr_p2) EMB(ocr_in)

static size_t bench(const char *name, const uint8_t *data, size_t len, const char *where, uint32_t caps, size_t arena_size) {
    uint8_t *arena = (uint8_t *)heap_caps_malloc(arena_size, caps | MALLOC_CAP_8BIT);
    if (!arena) { printf("RESULT model=%s mem=%s status=ARENA_ALLOC_FAILED asked=%u\n", name, where, (unsigned)arena_size); printf("  largest_free_block=%u\n", (unsigned)heap_caps_get_largest_free_block(caps | MALLOC_CAP_8BIT)); return 0; }
    const tflite::Model *model = tflite::GetModel(data);
    static tflite::MicroMutableOpResolver<16> r;
    static bool init = false;
    if (!init) {
        r.AddConv2D(); r.AddDepthwiseConv2D(); r.AddAdd(); r.AddMul(); r.AddMaxPool2D(); r.AddTranspose();
        r.AddReshape(); r.AddFullyConnected(); r.AddSoftmax(); r.AddResizeNearestNeighbor();
        r.AddLogistic(); r.AddQuantize(); r.AddConcatenation();
        init = true;
    }
    alignas(16) static uint8_t storage[sizeof(tflite::MicroInterpreter)];
    auto *interp = new (storage) tflite::MicroInterpreter(model, r, arena, arena_size);
    if (interp->AllocateTensors() != kTfLiteOk) {
        printf("RESULT model=%s mem=%s status=ALLOCATE_FAILED arena=%u\n", name, where, (unsigned)arena_size);
        interp->~MicroInterpreter(); heap_caps_free(arena); return 0;
    }
    TfLiteTensor *in = interp->input(0);
    uint32_t s = 12345;
    for (size_t i = 0; i < in->bytes; i++) { s = s * 1664525u + 1013904223u; in->data.int8[i] = (int8_t)(s >> 24); }
    for (int i = 0; i < 2; i++) interp->Invoke();
    const int N = 20; int64_t sum = 0, mn = INT64_MAX, mx = 0; bool ok = true;
    for (int i = 0; i < N; i++) {
        int64_t t0 = esp_timer_get_time();
        if (interp->Invoke() != kTfLiteOk) { ok = false; break; }
        int64_t dt = esp_timer_get_time() - t0; sum += dt; if (dt < mn) mn = dt; if (dt > mx) mx = dt;
        vTaskDelay(1);
    }
    if (!ok) printf("RESULT model=%s mem=%s status=INVOKE_FAILED\n", name, where);
    else printf("RESULT model=%s mem=%s status=OK model_bytes=%u arena_used=%u input_bytes=%u invoke_ms_mean=%.2f min=%.2f max=%.2f runs=%d\n",
                name, where, (unsigned)len, (unsigned)interp->arena_used_bytes(), (unsigned)in->bytes,
                sum / (double)N / 1000.0, mn / 1000.0, mx / 1000.0, N);
    size_t used = interp->arena_used_bytes();
    interp->~MicroInterpreter(); heap_caps_free(arena);
    return ok ? used : 0;
}

static void check(const char *name, const uint8_t *data) {
    const size_t arena_size = 2 * 1024 * 1024;
    uint8_t *arena = (uint8_t *)heap_caps_malloc(arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!arena) { printf("CHECK model=%s status=ARENA_ALLOC_FAILED\n", name); return; }
    static tflite::MicroMutableOpResolver<16> r;
    static bool init = false;
    if (!init) {
        r.AddConv2D(); r.AddDepthwiseConv2D(); r.AddAdd(); r.AddMul(); r.AddMaxPool2D(); r.AddTranspose();
        r.AddReshape(); r.AddFullyConnected(); r.AddSoftmax(); r.AddResizeNearestNeighbor();
        r.AddLogistic(); r.AddQuantize(); r.AddConcatenation();
        init = true;
    }
    alignas(16) static uint8_t storage[sizeof(tflite::MicroInterpreter)];
    auto *interp = new (storage) tflite::MicroInterpreter(tflite::GetModel(data), r, arena, arena_size);
    if (interp->AllocateTensors() != kTfLiteOk) { printf("CHECK model=%s status=ALLOCATE_FAILED\n", name); heap_caps_free(arena); return; }
    TfLiteTensor *in = interp->input(0), *out = interp->output(0);
    const int lines = (ocr_in_end - ocr_in_start) / in->bytes;
    const int steps = out->dims->data[1], classes = out->dims->data[2];
    for (int k = 0; k < lines; k++) {
        memcpy(in->data.int8, ocr_in_start + k * in->bytes, in->bytes);
        interp->Invoke();
        printf("IDS model=%s i=%d ", name, k);
        for (int t = 0; t < steps; t++) {
            const int8_t *row = out->data.int8 + t * classes; int best = 0;
            for (int c = 1; c < classes; c++) if (row[c] > row[best]) best = c;
            printf("%02x", best);
        }
        printf("\n");
        vTaskDelay(1);
    }
    interp->~MicroInterpreter(); heap_caps_free(arena);
}

extern "C" void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(1500));
    printf("INFO internal_free=%u spiram_total=%u spiram_free=%u cpu_mhz=%d\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_total_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    struct M { const char *n; const uint8_t *s, *e; } ms[] = {
        {"tiny", ocr_t_start, ocr_t_end}, {"D", ocr_d_start, ocr_d_end}, {"P2", ocr_p2_start, ocr_p2_end}};
    for (auto &m : ms) {
        size_t used = 0;
        if (heap_caps_get_total_size(MALLOC_CAP_SPIRAM) > 0) used = bench(m.n, m.s, m.e - m.s, "psram", MALLOC_CAP_SPIRAM, 2 * 1024 * 1024);
        size_t want = used ? ((used + 8192 + 15) & ~(size_t)15) : 420 * 1024;
        bench(m.n, m.s, m.e - m.s, "internal", MALLOC_CAP_INTERNAL, want);
    }
    for (auto &m : ms) check(m.n, m.s);
    printf("DONE\n");
    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}
