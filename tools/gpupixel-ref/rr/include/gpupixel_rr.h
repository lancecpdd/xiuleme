#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef GPUPIXEL_RR_EXPORTS
#define GPUPIXEL_RR_API __declspec(dllexport)
#else
#define GPUPIXEL_RR_API __declspec(dllimport)
#endif
#else
#define GPUPIXEL_RR_API __attribute__((visibility("default")))
#endif

typedef void* GpRrHandle;

/** Create an uninitialized session. */
GPUPIXEL_RR_API GpRrHandle gpupixel_rr_create(void);

/**
 * Initialize OpenGL context and filter graph.
 * resource_dir: directory containing face_pc[1.0.0].vnnmodel and lookup_*.png.
 *               On Windows the upstream loader resolves bare filenames from CWD;
 *               the implementation chdirs to resource_dir for the call.
 */
GPUPIXEL_RR_API int gpupixel_rr_initialize(GpRrHandle handle, const char* resource_dir);

/** Register landmark / pixel callbacks (must follow initialize). */
GPUPIXEL_RR_API void gpupixel_rr_set_callbacks(GpRrHandle handle);

/**
 * Levels are GPUPixel desktop 0..10 (not 0..100).
 * Mapping inside: beauty/10, white/20, thin/200, eye/100, lipstick/10, blusher/10,
 * eye_brighten/10, eye_redness/10.
 *
 * Legacy 6-arg callers: use gpupixel_rr_set_parameters_v1 or pass 0 for the last two.
 */
GPUPIXEL_RR_API void gpupixel_rr_set_parameters(
    GpRrHandle handle,
    float beauty,
    float white,
    float thin_face,
    float big_eye,
    float lipstick,
    float blusher,
    float eye_brighten,
    float eye_redness);

/** Backward-compatible 6-parameter setter (eye white intensities stay 0). */
GPUPIXEL_RR_API void gpupixel_rr_set_parameters_v1(
    GpRrHandle handle,
    float beauty,
    float white,
    float thin_face,
    float big_eye,
    float lipstick,
    float blusher);

/**
 * Process contiguous RGBA8 (height * width * 4).
 * Returns pointer owned by the session; copy immediately, then call
 * gpupixel_rr_release_output. Returns NULL on failure.
 */
GPUPIXEL_RR_API uint8_t* gpupixel_rr_run(
    GpRrHandle handle,
    const uint8_t* rgba,
    int width,
    int height,
    int channels);

/** Free the buffer returned by the last successful gpupixel_rr_run. */
GPUPIXEL_RR_API void gpupixel_rr_release_output(GpRrHandle handle);

/** Destroy the session. */
GPUPIXEL_RR_API void gpupixel_rr_destroy(GpRrHandle handle);

#ifdef __cplusplus
}
#endif
