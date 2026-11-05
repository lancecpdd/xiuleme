// Smoke-test body.slim_arms against real ONNX models without image-file codecs.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "portrait_engine.h"

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s (%s)\n", what, pe_last_error());
        ++failures;
    }
}

}  // namespace

int main() {
    const char* resources = std::getenv("PE_RESOURCES");
    const char* ort = std::getenv("PE_ORT");
    if (resources == nullptr || ort == nullptr) {
        std::fprintf(stderr, "skip: set PE_RESOURCES and PE_ORT to run ORT smoke test\n");
        return 0;
    }

    PeEngineDesc desc{};
    desc.struct_size = sizeof(desc);
    desc.resource_dir = resources;
    desc.onnxruntime_path = ort;
    PeEngine* engine = nullptr;
    check(pe_engine_create(&desc, &engine) == PE_OK && engine != nullptr, "create");
    if (engine == nullptr) return 1;

    const std::string caps = pe_engine_capabilities(engine);
    check(caps.find("body.slim_arms") != std::string::npos, "feature present");

    constexpr uint32_t width = 320;
    constexpr uint32_t height = 480;
    std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4, 90);
    // Paint a crude torso+arms blob so SCHP has something non-empty; still may bypass.
    for (uint32_t y = 80; y < 360; ++y) {
        for (uint32_t x = 120; x < 200; ++x) {
            const size_t i = (static_cast<size_t>(y) * width + x) * 4;
            rgba[i] = 200;
            rgba[i + 1] = 160;
            rgba[i + 2] = 140;
            rgba[i + 3] = 255;
        }
    }
    for (uint32_t y = 120; y < 280; ++y) {
        for (uint32_t x = 60; x < 100; ++x) {
            const size_t i = (static_cast<size_t>(y) * width + x) * 4;
            rgba[i] = 200;
            rgba[i + 1] = 160;
            rgba[i + 2] = 140;
            rgba[i + 3] = 255;
        }
        for (uint32_t x = 220; x < 260; ++x) {
            const size_t i = (static_cast<size_t>(y) * width + x) * 4;
            rgba[i] = 200;
            rgba[i + 1] = 160;
            rgba[i + 2] = 140;
            rgba[i + 3] = 255;
        }
    }

    PeImage image{};
    image.struct_size = sizeof(image);
    image.format = PE_FORMAT_RGBA8;
    image.width = width;
    image.height = height;
    image.stride_bytes = width * 4;
    image.data = rgba.data();

    PeAnalysis* analysis = nullptr;
    const PeStatus status =
        pe_analyze(engine, "body.slim_arms", &image, nullptr, nullptr, &analysis);
    check(status == PE_OK && analysis != nullptr, "analyze returns OK (ready or bypass)");
    if (analysis != nullptr) {
        const char* diagnostics = pe_analysis_diagnostics(analysis);
        check(diagnostics != nullptr && std::strstr(diagnostics, "status") != nullptr,
              "diagnostics json");
        std::printf("analysis: %s\n", diagnostics);

        PeParam param{"body.slim_arms", "amount", 50.0};
        const PeAnalysis* one[] = {analysis};
        PeWarpResult* result = nullptr;
        check(pe_solve_warp(engine, one, 1, &param, 1, nullptr, nullptr, &result) == PE_OK,
              "solve");
        if (result != nullptr) {
            std::printf("solve: %s\n", pe_warp_diagnostics(result));
            std::printf("regions: %zu\n", pe_warp_region_count(result));
            pe_warp_result_release(result);
        }
        pe_analysis_release(analysis);
    }

    pe_engine_destroy(engine);
    if (failures == 0) {
        std::printf("portrait engine ORT smoke passed\n");
        return 0;
    }
    return 1;
}
