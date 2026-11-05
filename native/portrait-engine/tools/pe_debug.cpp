/*
 * Debug CLI for body.slim_arms.
 * Reads/writes Windows BMP (no OpenCV codec plugins required for static builds).
 *
 * Usage:
 *   pe_debug --resources <dir> --ort <onnxruntime.dll> --image in.bmp --out-dir out [--amount 60]
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "portrait_engine.h"

namespace {

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "pe_debug: %s (%s)\n", message, pe_last_error());
    std::exit(1);
}

std::filesystem::path u8_path(const std::string& utf8) {
#if defined(_WIN32)
    return std::filesystem::u8path(utf8);
#else
    return std::filesystem::path(utf8);
#endif
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

#pragma pack(push, 1)
struct BmpFileHeader {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t off_bits;
};
struct BmpInfoHeader {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bit_count;
    uint32_t compression;
    uint32_t size_image;
    int32_t x_pels;
    int32_t y_pels;
    uint32_t clr_used;
    uint32_t clr_important;
};
#pragma pack(pop)

cv::Mat read_bmp(const std::string& path) {
    std::ifstream in(u8_path(path), std::ios::binary);
    if (!in) fail("cannot open image");
    BmpFileHeader file{};
    BmpInfoHeader info{};
    in.read(reinterpret_cast<char*>(&file), sizeof(file));
    in.read(reinterpret_cast<char*>(&info), sizeof(info));
    if (!in || file.type != 0x4D42 || info.bit_count != 24 || info.compression != 0) {
        fail("only uncompressed 24-bit BMP is supported");
    }
    const int width = info.width;
    const int height = std::abs(info.height);
    const bool bottom_up = info.height > 0;
    const int row_stride = ((width * 3 + 3) / 4) * 4;
    std::vector<uint8_t> rows(static_cast<size_t>(row_stride) * height);
    in.seekg(file.off_bits, std::ios::beg);
    in.read(reinterpret_cast<char*>(rows.data()), static_cast<std::streamsize>(rows.size()));
    if (!in) fail("truncated BMP");
    cv::Mat bgr(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        const int src_y = bottom_up ? (height - 1 - y) : y;
        std::memcpy(bgr.ptr(y), rows.data() + static_cast<size_t>(src_y) * row_stride,
                    static_cast<size_t>(width) * 3);
    }
    return bgr;
}

void write_bmp(const std::string& path, const cv::Mat& bgr) {
    CV_Assert(bgr.type() == CV_8UC3);
    const int width = bgr.cols;
    const int height = bgr.rows;
    const int row_stride = ((width * 3 + 3) / 4) * 4;
    BmpFileHeader file{};
    BmpInfoHeader info{};
    file.type = 0x4D42;
    file.off_bits = sizeof(file) + sizeof(info);
    file.size = file.off_bits + static_cast<uint32_t>(row_stride * height);
    info.size = sizeof(info);
    info.width = width;
    info.height = height;
    info.planes = 1;
    info.bit_count = 24;
    info.size_image = static_cast<uint32_t>(row_stride * height);
    std::ofstream out(u8_path(path), std::ios::binary);
    out.write(reinterpret_cast<const char*>(&file), sizeof(file));
    out.write(reinterpret_cast<const char*>(&info), sizeof(info));
    std::vector<uint8_t> row(static_cast<size_t>(row_stride), 0);
    for (int y = height - 1; y >= 0; --y) {
        std::memcpy(row.data(), bgr.ptr(y), static_cast<size_t>(width) * 3);
        out.write(reinterpret_cast<const char*>(row.data()), row_stride);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string resources;
    std::string ort;
    std::string image_path;
    std::string out_dir = "pe-debug-out";
    double amount = 60.0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](std::string& target) {
            if (i + 1 >= argc) fail("missing argument value");
            target = argv[++i];
        };
        if (arg == "--resources") next(resources);
        else if (arg == "--ort") next(ort);
        else if (arg == "--image") next(image_path);
        else if (arg == "--out-dir") next(out_dir);
        else if (arg == "--amount") {
            if (i + 1 >= argc) fail("missing amount");
            amount = std::atof(argv[++i]);
        } else {
            fail("unknown argument");
        }
    }
    if (resources.empty() || ort.empty() || image_path.empty()) {
        std::fprintf(stderr,
                     "usage: pe_debug --resources DIR --ort onnxruntime.dll --image in.bmp "
                     "[--out-dir DIR] [--amount 60]\n");
        return 2;
    }

    cv::Mat bgr = read_bmp(image_path);
    cv::Mat rgba;
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    std::filesystem::create_directories(u8_path(out_dir));

    PeEngineDesc desc{};
    desc.struct_size = sizeof(desc);
    desc.resource_dir = resources.c_str();
    desc.onnxruntime_path = ort.c_str();
    PeEngine* engine = nullptr;
    if (pe_engine_create(&desc, &engine) != PE_OK) fail("engine create");
    write_text(u8_path(out_dir) / "capabilities.json", pe_engine_capabilities(engine));

    PeImage image{};
    image.struct_size = sizeof(image);
    image.format = PE_FORMAT_RGBA8;
    image.width = static_cast<uint32_t>(rgba.cols);
    image.height = static_cast<uint32_t>(rgba.rows);
    image.stride_bytes = static_cast<uint32_t>(rgba.step);
    image.data = rgba.data;

    PeAnalysis* analysis = nullptr;
    if (pe_analyze(engine, "body.slim_arms", &image, nullptr, nullptr, &analysis) != PE_OK) {
        fail("analyze");
    }
    write_text(u8_path(out_dir) / "analysis.json", pe_analysis_diagnostics(analysis));

    PeParam param{"body.slim_arms", "amount", amount};
    const PeAnalysis* analyses[] = {analysis};
    PeWarpResult* result = nullptr;
    if (pe_solve_warp(engine, analyses, 1, &param, 1, nullptr, nullptr, &result) != PE_OK) {
        fail("solve");
    }
    write_text(u8_path(out_dir) / "solve.json", pe_warp_diagnostics(result));

    cv::Mat map_x(bgr.size(), CV_32FC1);
    cv::Mat map_y(bgr.size(), CV_32FC1);
    for (int y = 0; y < bgr.rows; ++y) {
        for (int x = 0; x < bgr.cols; ++x) {
            map_x.at<float>(y, x) = static_cast<float>(x);
            map_y.at<float>(y, x) = static_cast<float>(y);
        }
    }
    const size_t region_count = pe_warp_region_count(result);
    for (size_t r = 0; r < region_count; ++r) {
        PeWarpRegion region{};
        region.struct_size = sizeof(region);
        if (pe_warp_region(result, r, &region) != PE_OK) fail("region");
        for (uint32_t j = 0; j <= region.cells_y; ++j) {
            for (uint32_t i = 0; i <= region.cells_x; ++i) {
                const size_t index = static_cast<size_t>(j * (region.cells_x + 1) + i);
                const float tx = region.x0 + (region.x1 - region.x0) * i / region.cells_x;
                const float ty = region.y0 + (region.y1 - region.y0) * j / region.cells_y;
                const int x = std::clamp(static_cast<int>(std::floor(tx)), 0, bgr.cols - 1);
                const int y = std::clamp(static_cast<int>(std::floor(ty)), 0, bgr.rows - 1);
                map_x.at<float>(y, x) = region.source_xy[index * 2] - 0.5f;
                map_y.at<float>(y, x) = region.source_xy[index * 2 + 1] - 0.5f;
            }
        }
    }
    cv::Mat warped;
    cv::remap(bgr, warped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    write_bmp((u8_path(out_dir) / "input.bmp").u8string(), bgr);
    write_bmp((u8_path(out_dir) / "warped.bmp").u8string(), warped);

    pe_warp_result_release(result);
    pe_analysis_release(analysis);
    pe_engine_destroy(engine);
    std::printf("wrote %s (%zu regions)\n", out_dir.c_str(), region_count);
    return 0;
}
