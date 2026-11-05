// 只通过公开 C ABI 验证引擎行为，与宿主看到的接口一致。
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "portrait_engine.h"

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (last error: %s)\n", what, pe_last_error());
        ++failures;
    }
}

int32_t always_cancel(void*) { return 1; }

PeImage make_image(std::vector<uint8_t>& pixels, uint32_t width, uint32_t height) {
    pixels.assign(static_cast<size_t>(width) * height * 4, 128);
    PeImage image{};
    image.struct_size = sizeof(PeImage);
    image.format = PE_FORMAT_RGBA8;
    image.width = width;
    image.height = height;
    image.stride_bytes = width * 4;
    image.data = pixels.data();
    return image;
}

}  // namespace

int main() {
    check(pe_abi_version() == PE_ABI_VERSION, "abi version matches header");

    PeEngineDesc desc{};
    desc.struct_size = sizeof(PeEngineDesc);
    PeEngine* engine = nullptr;
    check(pe_engine_create(&desc, &engine) == PE_OK && engine != nullptr, "engine create");
    if (engine == nullptr) return 1;

    const std::string caps = pe_engine_capabilities(engine);
    check(caps.find("\"abi\":1") != std::string::npos, "capabilities report abi");
    check(caps.find("\"debug.identity\"") != std::string::npos, "capabilities list identity");
    check(caps.find("\"debug.pinch\"") != std::string::npos, "capabilities list pinch");
#if PE_HAS_OPENCV
    check(caps.find("\"body.slim_arms\"") != std::string::npos, "capabilities list slim arms");
#if PE_HAS_OPENCV
    check(caps.find("\"body.slim_waist\"") != std::string::npos, "capabilities list slim waist");
    check(caps.find("\"body.slim_legs\"") != std::string::npos, "capabilities list slim legs");
#endif
    check(caps.find("rtmpose-s-256x192.onnx") != std::string::npos, "capabilities list pose model");
#endif

    std::vector<uint8_t> pixels;
    PeImage image = make_image(pixels, 200, 120);

    PeAnalysis* identity = nullptr;
    check(pe_analyze(engine, "debug.identity", &image, nullptr, nullptr, &identity) == PE_OK,
          "identity analyze");
    check(std::strcmp(pe_analysis_feature(identity), "debug.identity") == 0, "analysis feature");
    check(std::strstr(pe_analysis_diagnostics(identity), "ready") != nullptr,
          "analysis diagnostics");

    uint8_t* blob = nullptr;
    size_t blob_size = 0;
    check(pe_analysis_serialize(identity, &blob, &blob_size) == PE_OK && blob_size > 0,
          "serialize");
    PeAnalysis* restored = nullptr;
    check(pe_analysis_deserialize(engine, blob, blob_size, &restored) == PE_OK,
          "deserialize round trip");
    PeAnalysis* truncated = nullptr;
    check(pe_analysis_deserialize(engine, blob, blob_size - 1, &truncated) ==
                  PE_ERR_INVALID_ARGUMENT &&
              truncated == nullptr,
          "truncated blob rejected");
    pe_buffer_free(blob);

    PeParam half{"debug.identity", "amount", 50.0};
    PeWarpResult* result = nullptr;
    const PeAnalysis* one[] = {identity};
    check(pe_solve_warp(engine, one, 1, &half, 1, nullptr, nullptr, &result) == PE_OK,
          "identity solve");
    check(pe_warp_region_count(result) == 0, "identity yields no regions");
    check(std::strstr(pe_warp_diagnostics(result), "\"identity\"") != nullptr,
          "identity diagnostics");
    pe_warp_result_release(result);

    PeParam unknown{"debug.identity", "strength", 1.0};
    check(pe_solve_warp(engine, one, 1, &unknown, 1, nullptr, nullptr, &result) ==
              PE_ERR_INVALID_ARGUMENT,
          "unknown parameter rejected");
    PeParam out_of_range{"debug.identity", "amount", 101.0};
    check(pe_solve_warp(engine, one, 1, &out_of_range, 1, nullptr, nullptr, &result) ==
              PE_ERR_INVALID_ARGUMENT,
          "out of range amount rejected");
    check(std::strlen(pe_last_error()) > 0, "last error is set");

    PeAnalysis* pinch = nullptr;
    check(pe_analyze(engine, "debug.pinch", &image, nullptr, nullptr, &pinch) == PE_OK,
          "pinch analyze");
    const PeAnalysis* both[] = {identity, pinch};
    PeParam full{"debug.pinch", "amount", 100.0};
    check(pe_solve_warp(engine, both, 2, &full, 1, nullptr, nullptr, &result) == PE_OK,
          "combined solve");
    check(pe_warp_region_count(result) == 1, "pinch yields one region");
    PeWarpRegion region{};
    region.struct_size = sizeof(PeWarpRegion);
    check(pe_warp_region(result, 0, &region) == PE_OK, "read region");
    const size_t columns = region.cells_x + 1;
    const size_t nodes = columns * (region.cells_y + 1);
    bool boundary_fixed = true;
    bool interior_moved = false;
    for (size_t n = 0; n < nodes; ++n) {
        const size_t i = n % columns;
        const size_t j = n / columns;
        const float tx = region.x0 + (region.x1 - region.x0) * i / region.cells_x;
        const float ty = region.y0 + (region.y1 - region.y0) * j / region.cells_y;
        const float dx = region.source_xy[2 * n] - tx;
        const float dy = region.source_xy[2 * n + 1] - ty;
        const bool edge = i == 0 || j == 0 || i == region.cells_x || j == region.cells_y;
        if (edge && (std::fabs(dx) > 1e-3f || std::fabs(dy) > 1e-3f)) boundary_fixed = false;
        if (!edge && (std::fabs(dx) > 0.5f || std::fabs(dy) > 0.5f)) interior_moved = true;
    }
    check(boundary_fixed, "pinch region boundary is identity");
    check(interior_moved, "pinch region interior moves");
    check(pe_warp_region(result, 1, &region) == PE_ERR_INVALID_ARGUMENT, "region index bound");
    pe_warp_result_release(result);

    const PeAnalysis* twice[] = {pinch, pinch};
    check(pe_solve_warp(engine, twice, 2, nullptr, 0, nullptr, nullptr, &result) ==
              PE_ERR_INVALID_ARGUMENT,
          "same feature twice rejected");
    check(pe_solve_warp(engine, both, 2, &full, 1, always_cancel, nullptr, &result) ==
              PE_CANCELLED,
          "solve cancellation");

    PeAnalysis* missing = nullptr;
    check(pe_analyze(engine, "body.unknown", &image, nullptr, nullptr, &missing) ==
              PE_ERR_UNSUPPORTED,
          "unknown feature rejected");
    image.stride_bytes = 10;
    check(pe_analyze(engine, "debug.identity", &image, nullptr, nullptr, &missing) ==
              PE_ERR_INVALID_ARGUMENT,
          "short stride rejected");
    check(pe_analyze(engine, "debug.identity", nullptr, nullptr, nullptr, &missing) ==
              PE_ERR_INVALID_ARGUMENT,
          "null image rejected");

    pe_analysis_release(restored);
    pe_analysis_release(pinch);
    pe_analysis_release(identity);
    pe_engine_destroy(engine);

    if (failures == 0) {
        std::printf("portrait engine self test passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
}
