#include "feature_body_narrow.h"

#include "feature_slim_arms.h"
#include "ort_runtime.h"
#include "sha256.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>

#if PE_HAS_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace pe {
namespace {

enum class NarrowKind { Waist, Legs };

constexpr char kPoseFile[] = "rtmpose-s-256x192.onnx";
constexpr char kParsingFile[] = "schp-pascal-7-int8.onnx";
constexpr char kPoseSha[] = "d869413ccfbc0ed027337c773f124d6362436239a83fd0a6e2d43b7a8377792e";
constexpr char kParsingSha[] = "66b12766d7f1ddbc3de972e67e8626be727507e7feeeca34e1b23b6f45e756d2";
constexpr int kGrid = 512;
constexpr int kPoseW = 192;
constexpr int kPoseH = 256;

using slim_arms_test::ArmSegment;
using slim_arms_test::Point;
using slim_arms_test::smooth;
using slim_arms_test::source_point;

#if PE_HAS_OPENCV

std::filesystem::path u8_path(const std::string& utf8) {
#if defined(_WIN32)
    return std::filesystem::u8path(utf8);
#else
    return std::filesystem::path(utf8);
#endif
}

std::string join_resource(const std::string& dir, const char* file) {
    return (u8_path(dir) / file).u8string();
}

void require_model(const std::string& resource_dir, const char* file, const char* expected_sha) {
    const std::string path = join_resource(resource_dir, file);
    if (!std::filesystem::is_regular_file(u8_path(path))) {
        throw Error(PE_ERR_MODEL, std::string("missing model: ") + file);
    }
    if (sha256_file(path) != expected_sha) {
        throw Error(PE_ERR_MODEL, std::string("model hash mismatch: ") + file);
    }
}

cv::Mat image_to_bgr(const ImageView& image) {
    if (image.format != PE_FORMAT_RGBA8) {
        throw Error(PE_ERR_UNSUPPORTED, "body narrow requires RGBA8 input");
    }
    cv::Mat rgba(static_cast<int>(image.height), static_cast<int>(image.width), CV_8UC4,
                 const_cast<uint8_t*>(image.data), image.stride_bytes);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    return bgr;
}

void nchw_from_rect(const cv::Mat& bgr, int out_w, int out_h, const cv::Rect2f& rect,
                    const cv::Scalar& mean, const cv::Scalar& stddev, std::vector<float>& out) {
    out.assign(static_cast<size_t>(3 * out_w * out_h), 0.0f);
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            const float px = rect.x + (x + 0.5f) * rect.width / out_w;
            const float py = rect.y + (y + 0.5f) * rect.height / out_h;
            const int ix = std::clamp(static_cast<int>(std::floor(px)), 0, bgr.cols - 1);
            const int iy = std::clamp(static_cast<int>(std::floor(py)), 0, bgr.rows - 1);
            const int ix1 = std::min(ix + 1, bgr.cols - 1);
            const int iy1 = std::min(iy + 1, bgr.rows - 1);
            const float fx = std::clamp(px - ix, 0.0f, 1.0f);
            const float fy = std::clamp(py - iy, 0.0f, 1.0f);
            const cv::Vec3b p00 = bgr.at<cv::Vec3b>(iy, ix);
            const cv::Vec3b p10 = bgr.at<cv::Vec3b>(iy, ix1);
            const cv::Vec3b p01 = bgr.at<cv::Vec3b>(iy1, ix);
            const cv::Vec3b p11 = bgr.at<cv::Vec3b>(iy1, ix1);
            for (int c = 0; c < 3; ++c) {
                const int src = 2 - c;
                const float v00 = p00[src] / 255.0f;
                const float v10 = p10[src] / 255.0f;
                const float v01 = p01[src] / 255.0f;
                const float v11 = p11[src] / 255.0f;
                const float value = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 +
                                   (1 - fx) * fy * v01 + fx * fy * v11;
                out[static_cast<size_t>(c * out_h * out_w + y * out_w + x)] =
                    static_cast<float>((value - mean[c]) / stddev[c]);
            }
        }
    }
}

struct PartMap {
    std::vector<uint8_t> labels;
    std::vector<float> confidence;
    int bounds[4] = {0, 0, kGrid, kGrid};

    std::pair<uint8_t, float> at_image(Point point, int width, int height) const {
        const int x = static_cast<int>(std::floor((point.x / std::max(width, 1)) * kGrid));
        const int y = static_cast<int>(std::floor((point.y / std::max(height, 1)) * kGrid));
        const size_t index =
            static_cast<size_t>(std::clamp(y, 0, kGrid - 1) * kGrid + std::clamp(x, 0, kGrid - 1));
        return {labels[index], confidence[index]};
    }
};

std::pair<int, float> simcc_peak(const float* values, int count) {
    int best = 0;
    float max_value = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < count; ++i) {
        if (values[i] > max_value) {
            max_value = values[i];
            best = i;
        }
    }
    return {best, max_value};
}

class BodySessions {
public:
    void ensure(const EngineConfig& config, const CancelToken& cancel) {
        if (pose_ && parsing_) {
            return;
        }
        cancel.check();
        if (config.resource_dir.empty()) {
            throw Error(PE_ERR_MODEL, "resource directory is empty");
        }
        require_model(config.resource_dir, kPoseFile, kPoseSha);
        require_model(config.resource_dir, kParsingFile, kParsingSha);
        OrtRuntime::instance().ensure_loaded(config.onnxruntime_path);
        const int threads = config.threads > 0 ? config.threads : 2;
        pose_ = std::make_unique<OrtRuntime::Session>(
            OrtRuntime::instance(), join_resource(config.resource_dir, kPoseFile), threads);
        parsing_ = std::make_unique<OrtRuntime::Session>(
            OrtRuntime::instance(), join_resource(config.resource_dir, kParsingFile), threads);
    }

    PartMap run_parsing(const cv::Mat& bgr, const CancelToken& cancel) {
        cancel.check();
        std::vector<float> input;
        nchw_from_rect(bgr, kGrid, kGrid,
                       cv::Rect2f(0, 0, static_cast<float>(bgr.cols), static_cast<float>(bgr.rows)),
                       cv::Scalar(0.485, 0.456, 0.406), cv::Scalar(0.229, 0.224, 0.225), input);
        std::vector<std::vector<int64_t>> shapes;
        const std::vector<int64_t> shape = {1, 3, kGrid, kGrid};
        const std::vector<const char*> outputs = {"logits"};
        const std::vector<float> flat =
            parsing_->run("pixel_values", shape, input.data(), input.size(), outputs, shapes);
        if (shapes.size() != 1 || shapes[0] != std::vector<int64_t>({1, 7, kGrid, kGrid})) {
            throw Error(PE_ERR_MODEL, "SCHP logits shape mismatch");
        }
        PartMap parts;
        parts.labels.resize(static_cast<size_t>(kGrid * kGrid));
        parts.confidence.resize(parts.labels.size());
        for (int y = 0; y < kGrid; ++y) {
            for (int x = 0; x < kGrid; ++x) {
                int best = 0;
                float max_value = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < 7; ++c) {
                    const float value =
                        flat[static_cast<size_t>(c * kGrid * kGrid + y * kGrid + x)];
                    if (value > max_value) {
                        max_value = value;
                        best = c;
                    }
                }
                float denom = 0.0f;
                for (int c = 0; c < 7; ++c) {
                    const float value =
                        flat[static_cast<size_t>(c * kGrid * kGrid + y * kGrid + x)];
                    denom += std::exp(value - max_value);
                }
                const size_t index = static_cast<size_t>(y * kGrid + x);
                parts.labels[index] = static_cast<uint8_t>(best);
                parts.confidence[index] = 1.0f / std::max(denom, 1e-12f);
            }
        }
        cv::Mat mask(kGrid, kGrid, CV_8UC1);
        for (int i = 0; i < kGrid * kGrid; ++i) {
            mask.data[i] = parts.labels[static_cast<size_t>(i)] == 0 ? 0 : 255;
        }
        cv::Mat labels_cc, stats, centroids;
        const int count = cv::connectedComponentsWithStats(mask, labels_cc, stats, centroids, 8);
        struct Component {
            int area = 0;
            int bounds[4] = {0, 0, 0, 0};
        };
        std::vector<Component> components;
        for (int i = 1; i < count; ++i) {
            Component component;
            component.area = stats.at<int>(i, cv::CC_STAT_AREA);
            const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            const int y = stats.at<int>(i, cv::CC_STAT_TOP);
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            component.bounds[0] = x;
            component.bounds[1] = y;
            component.bounds[2] = x + w;
            component.bounds[3] = y + h;
            components.push_back(component);
        }
        std::sort(components.begin(), components.end(),
                  [](const Component& a, const Component& b) { return a.area > b.area; });
        if (components.empty()) {
            throw Error(PE_ERR_INVALID_ARGUMENT, "no_primary_person");
        }
        if (components[0].area < 900 ||
            (components.size() > 1 &&
             components[1].area > static_cast<int>(components[0].area * 0.32f))) {
            throw Error(PE_ERR_INVALID_ARGUMENT, "multiple_people");
        }
        std::copy(std::begin(components[0].bounds), std::end(components[0].bounds), parts.bounds);
        return parts;
    }

    std::array<Point, 17> run_pose(const cv::Mat& bgr, const PartMap& parts,
                                   const CancelToken& cancel) {
        cancel.check();
        float x0 = parts.bounds[0] * static_cast<float>(bgr.cols) / kGrid;
        float y0 = parts.bounds[1] * static_cast<float>(bgr.rows) / kGrid;
        float x1 = parts.bounds[2] * static_cast<float>(bgr.cols) / kGrid;
        float y1 = parts.bounds[3] * static_cast<float>(bgr.rows) / kGrid;
        const Point center{(x0 + x1) * 0.5f, (y0 + y1) * 0.5f};
        float w = (x1 - x0) * 1.2f;
        float h = (y1 - y0) * 1.2f;
        const float aspect = static_cast<float>(kPoseW) / static_cast<float>(kPoseH);
        if (w / h < aspect) {
            w = h * aspect;
        } else {
            h = w / aspect;
        }
        const cv::Rect2f rect{center.x - w * 0.5f, center.y - h * 0.5f, w, h};
        std::vector<float> input;
        nchw_from_rect(bgr, kPoseW, kPoseH, rect,
                       cv::Scalar(123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0),
                       cv::Scalar(58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0), input);
        std::vector<std::vector<int64_t>> shapes;
        const std::vector<int64_t> shape = {1, 3, kPoseH, kPoseW};
        const std::vector<const char*> outputs = {"simcc_x", "simcc_y"};
        const std::vector<float> flat =
            pose_->run("input", shape, input.data(), input.size(), outputs, shapes);
        if (shapes.size() != 2 || shapes[0] != std::vector<int64_t>({1, 17, kPoseW * 2}) ||
            shapes[1] != std::vector<int64_t>({1, 17, kPoseH * 2})) {
            throw Error(PE_ERR_MODEL, "RTMPose SimCC shape mismatch");
        }
        const size_t x_count = static_cast<size_t>(1 * 17 * kPoseW * 2);
        const float* simcc_x = flat.data();
        const float* simcc_y = flat.data() + x_count;
        std::array<Point, 17> points{};
        for (int keypoint = 0; keypoint < 17; ++keypoint) {
            const auto [px, sx] = simcc_peak(simcc_x + keypoint * kPoseW * 2, kPoseW * 2);
            const auto [py, sy] = simcc_peak(simcc_y + keypoint * kPoseH * 2, kPoseH * 2);
            points[static_cast<size_t>(keypoint)] = {
                rect.x + (px * 0.5f / kPoseW) * rect.width,
                rect.y + (py * 0.5f / kPoseH) * rect.height,
            };
            points[static_cast<size_t>(keypoint)].x =
                std::isfinite(sx) && sx >= 0.15f ? points[static_cast<size_t>(keypoint)].x : -1.0f;
            points[static_cast<size_t>(keypoint)].y =
                std::isfinite(sy) && sy >= 0.15f ? points[static_cast<size_t>(keypoint)].y : -1.0f;
        }
        return points;
    }

private:
    std::unique_ptr<OrtRuntime::Session> pose_;
    std::unique_ptr<OrtRuntime::Session> parsing_;
};

std::optional<float> estimate_width(const PartMap& parts, int width, int height, Point start,
                                    Point end, uint8_t label_a, uint8_t label_b) {
    const Point axis = end.sub(start);
    const float length = axis.length();
    Point normal{-axis.y, axis.x};
    normal = normal.normalized();
    if (normal.x == 0 && normal.y == 0) {
        return std::nullopt;
    }
    const float max_radius =
        std::clamp(length * 0.35f, 4.0f, std::max(width, height) * 0.08f);
    const float step = std::max(std::max(width, height) / static_cast<float>(kGrid), 1.0f);
    std::vector<float> widths;
    for (int sample_index = 1; sample_index <= 5; ++sample_index) {
        const float t = sample_index / 6.0f;
        const Point center = start.add(axis.mul(t));
        float sides[2] = {0, 0};
        for (int side = 0; side < 2; ++side) {
            const float sign = side == 0 ? -1.0f : 1.0f;
            int misses = 0;
            for (float distance = step; distance <= max_radius; distance += step) {
                const Point point = center.add(normal.mul(sign * distance));
                const auto [part, confidence] = parts.at_image(point, width, height);
                if ((part == label_a || part == label_b) && confidence >= 0.3f) {
                    sides[side] = distance;
                    misses = 0;
                } else if (++misses >= 2) {
                    break;
                }
            }
        }
        if (sides[0] >= step * 2.0f && sides[1] >= step * 2.0f &&
            std::max(sides[0], sides[1]) <= std::min(sides[0], sides[1]) * 3.0f) {
            widths.push_back((sides[0] + sides[1]) * 0.5f);
        }
    }
    if (widths.size() < 3) {
        return std::nullopt;
    }
    std::nth_element(widths.begin(), widths.begin() + widths.size() / 2, widths.end());
    return std::clamp(widths[widths.size() / 2], length * 0.06f, length * 0.32f);
}

bool valid_point(Point point) { return point.x >= 0.0f && point.y >= 0.0f; }

void push_region(const std::vector<ArmSegment>& segments, float strength, int width, int height,
                 const PartMap& parts, uint8_t keep_a, uint8_t keep_b, std::vector<Region>& out) {
    if (segments.empty()) {
        return;
    }
    float min_x = static_cast<float>(width);
    float min_y = static_cast<float>(height);
    float max_x = 0.0f;
    float max_y = 0.0f;
    float radius = 0.0f;
    for (const auto& segment : segments) {
        radius = std::max(radius, segment.half_width * 1.6f);
        min_x = std::min({min_x, segment.start.x, segment.end.x});
        min_y = std::min({min_y, segment.start.y, segment.end.y});
        max_x = std::max({max_x, segment.start.x, segment.end.x});
        max_y = std::max({max_y, segment.start.y, segment.end.y});
    }
    min_x = std::max(0.0f, min_x - radius);
    min_y = std::max(0.0f, min_y - radius);
    max_x = std::min(static_cast<float>(width), max_x + radius);
    max_y = std::min(static_cast<float>(height), max_y + radius);
    if (!(max_x > min_x && max_y > min_y)) {
        return;
    }
    uint32_t cells_x =
        std::clamp(static_cast<uint32_t>(std::ceil((max_x - min_x) / 12.0f)), 8u, 96u);
    uint32_t cells_y =
        std::clamp(static_cast<uint32_t>(std::ceil((max_y - min_y) / 12.0f)), 8u, 96u);
    if ((cells_x + 1) * (cells_y + 1) == 217) {
        cells_x = std::min(96u, cells_x + 1);
    }
    Region region;
    region.x0 = min_x;
    region.y0 = min_y;
    region.x1 = max_x;
    region.y1 = max_y;
    region.cells_x = cells_x;
    region.cells_y = cells_y;
    region.source_xy.reserve(static_cast<size_t>(cells_x + 1) * (cells_y + 1) * 2);
    bool moved = false;
    for (uint32_t j = 0; j <= cells_y; ++j) {
        for (uint32_t i = 0; i <= cells_x; ++i) {
            const Point point{min_x + (max_x - min_x) * i / cells_x,
                              min_y + (max_y - min_y) * j / cells_y};
            const auto [label, confidence] = parts.at_image(point, width, height);
            Point source = point;
            if (confidence >= 0.25f && (label == keep_a || label == keep_b || label == 2)) {
                Point delta{};
                float total = 0.0f;
                for (const auto& segment : segments) {
                    bool ok = false;
                    const Point candidate = source_point(point, segment, strength, &ok);
                    if (!ok) {
                        continue;
                    }
                    const float displacement = candidate.sub(point).length();
                    const float weight = std::max(displacement, 1e-4f);
                    delta = delta.add(candidate.sub(point).mul(weight));
                    total += weight;
                }
                if (total > 0.0f) {
                    source = point.add(delta.mul(1.0f / total));
                }
            }
            region.source_xy.push_back(source.x);
            region.source_xy.push_back(source.y);
            if (std::fabs(source.x - point.x) > 1e-3f || std::fabs(source.y - point.y) > 1e-3f) {
                moved = true;
            }
        }
    }
    if (moved) {
        out.push_back(std::move(region));
    }
}

class NarrowAnalysisData final : public AnalysisData {
public:
    NarrowAnalysisData(uint32_t width, uint32_t height, NarrowKind kind)
        : AnalysisData(width, height), kind(kind) {}

    std::string diagnostics_json() const override {
        if (!bypass_reason.empty()) {
            return std::string(R"({"status":"bypass","reason":")") + json_escape(bypass_reason) +
                   "\"}";
        }
        return R"({"status":"ready"})";
    }

    std::vector<uint8_t> serialize_payload() const override {
        // Soft analyses are not snapshotted yet; empty ready payload is enough for ABI.
        std::vector<uint8_t> out;
        auto put_u32 = [&](uint32_t value) {
            for (int shift = 0; shift < 32; shift += 8) {
                out.push_back(static_cast<uint8_t>((value >> shift) & 0xff));
            }
        };
        put_u32(1);
        put_u32(static_cast<uint32_t>(bypass_reason.size()));
        out.insert(out.end(), bypass_reason.begin(), bypass_reason.end());
        put_u32(static_cast<uint32_t>(kind));
        put_u32(static_cast<uint32_t>(segments.size()));
        for (const auto& segment : segments) {
            auto put_f32 = [&](float value) {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                put_u32(bits);
            };
            put_f32(segment.start.x);
            put_f32(segment.start.y);
            put_f32(segment.end.x);
            put_f32(segment.end.y);
            put_f32(segment.half_width);
            put_u32(segment.fade_start ? 1u : 0u);
            put_u32(segment.fade_end ? 1u : 0u);
        }
        put_u32(static_cast<uint32_t>(parts.labels.size()));
        out.insert(out.end(), parts.labels.begin(), parts.labels.end());
        for (float value : parts.confidence) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            put_u32(bits);
        }
        return out;
    }

    NarrowKind kind;
    std::string bypass_reason;
    PartMap parts;
    std::vector<ArmSegment> segments;
};

std::unique_ptr<NarrowAnalysisData> deserialize_narrow(NarrowKind expected, uint32_t width,
                                                       uint32_t height, const uint8_t* data,
                                                       size_t size) {
    size_t offset = 0;
    auto need = [&](size_t count) {
        if (count > size - offset) {
            throw Error(PE_ERR_INVALID_ARGUMENT, "body narrow analysis blob truncated");
        }
    };
    auto u32 = [&]() {
        need(4);
        const uint8_t* bytes = data + offset;
        offset += 4;
        return static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 |
               static_cast<uint32_t>(bytes[2]) << 16 | static_cast<uint32_t>(bytes[3]) << 24;
    };
    auto f32 = [&]() {
        const uint32_t bits = u32();
        float value = 0;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    auto bytes = [&](size_t count) {
        need(count);
        const uint8_t* start = data + offset;
        offset += count;
        return start;
    };
    if (u32() != 1) {
        throw Error(PE_ERR_UNSUPPORTED, "unsupported body narrow analysis payload version");
    }
    auto analysis = std::make_unique<NarrowAnalysisData>(width, height, expected);
    const uint32_t reason_len = u32();
    const auto* reason = reinterpret_cast<const char*>(bytes(reason_len));
    analysis->bypass_reason.assign(reason, reason_len);
    analysis->kind = static_cast<NarrowKind>(u32());
    if (analysis->kind != expected) {
        throw Error(PE_ERR_INVALID_ARGUMENT, "body narrow analysis kind mismatch");
    }
    const uint32_t segment_count = u32();
    analysis->segments.resize(segment_count);
    for (auto& segment : analysis->segments) {
        segment.start.x = f32();
        segment.start.y = f32();
        segment.end.x = f32();
        segment.end.y = f32();
        segment.half_width = f32();
        segment.fade_start = u32() != 0;
        segment.fade_end = u32() != 0;
    }
    const uint32_t label_count = u32();
    const auto* labels = bytes(label_count);
    analysis->parts.labels.assign(labels, labels + label_count);
    analysis->parts.confidence.resize(label_count);
    for (uint32_t i = 0; i < label_count; ++i) {
        analysis->parts.confidence[i] = f32();
    }
    if (offset != size) {
        throw Error(PE_ERR_INVALID_ARGUMENT, "body narrow analysis blob has trailing bytes");
    }
    return analysis;
}

std::string map_bypass(const std::string& message) {
    if (message.find("no_primary_person") != std::string::npos) return "no_primary_person";
    if (message.find("multiple_people") != std::string::npos) return "multiple_people";
    if (message.find("low_keypoint_confidence") != std::string::npos)
        return "low_keypoint_confidence";
    if (message.find("incomplete_waist_boundary") != std::string::npos)
        return "incomplete_waist_boundary";
    if (message.find("incomplete_leg_boundary") != std::string::npos)
        return "incomplete_leg_boundary";
    return message;
}

class NarrowFeature final : public Feature {
public:
    NarrowFeature(EngineConfig config, NarrowKind kind, const char* id, const char* version)
        : config_(std::move(config)), kind_(kind), id_(id), version_(version) {}

    const char* id() const override { return id_; }
    const char* version() const override { return version_; }
    const char* kind() const override { return "warp"; }
    std::vector<std::string> param_names() const override { return {"amount"}; }
    std::vector<ModelInfo> models() const override {
        return {{kPoseFile, kPoseSha}, {kParsingFile, kParsingSha}};
    }

    std::unique_ptr<AnalysisData> analyze(const ImageView& image,
                                          const CancelToken& cancel) override {
        sessions_.ensure(config_, cancel);
        cancel.check();
        cv::Mat bgr = image_to_bgr(image);
        auto analysis = std::make_unique<NarrowAnalysisData>(image.width, image.height, kind_);
        try {
            analysis->parts = sessions_.run_parsing(bgr, cancel);
            const auto pose = sessions_.run_pose(bgr, analysis->parts, cancel);
            analysis->segments = build_segments(analysis->parts, bgr.cols, bgr.rows, pose);
        } catch (const Error& error) {
            if (error.status() == PE_ERR_MODEL || error.status() == PE_ERR_INTERNAL ||
                error.status() == PE_ERR_OUT_OF_MEMORY || error.status() == PE_ERR_UNSUPPORTED) {
                throw;
            }
            analysis->bypass_reason = map_bypass(error.what());
            analysis->segments.clear();
            analysis->parts = {};
        }
        return analysis;
    }

    std::unique_ptr<AnalysisData> deserialize(uint32_t width, uint32_t height, const uint8_t* payload,
                                              size_t size) const override {
        return deserialize_narrow(kind_, width, height, payload, size);
    }

    FeatureReport solve(const AnalysisData& analysis, const ParamView& params,
                        const CancelToken& cancel, std::vector<Region>& out) const override {
        const double amount = params.get("amount", 0.0);
        if (!std::isfinite(amount) || amount < 0.0 || amount > 100.0) {
            throw Error(PE_ERR_INVALID_ARGUMENT, "amount must be within [0, 100]");
        }
        const auto& body = dynamic_cast<const NarrowAnalysisData&>(analysis);
        if (!body.bypass_reason.empty()) {
            return {"bypass", body.bypass_reason};
        }
        if (amount == 0.0) {
            return {"identity", ""};
        }
        cancel.check();
        const float response =
            (kind_ == NarrowKind::Waist ? 0.12f : 0.14f) * smooth(static_cast<float>(amount / 100.0));
        const uint8_t keep_a = kind_ == NarrowKind::Waist ? 2 : 5;
        const uint8_t keep_b = kind_ == NarrowKind::Waist ? 2 : 6;
        const size_t before = out.size();
        push_region(body.segments, response, static_cast<int>(analysis.width),
                    static_cast<int>(analysis.height), body.parts, keep_a, keep_b, out);
        return out.size() > before ? FeatureReport{"applied", ""} : FeatureReport{"identity", ""};
    }

private:
    EngineConfig config_;
    NarrowKind kind_;
    const char* id_;
    const char* version_;
    mutable BodySessions sessions_;

    std::vector<ArmSegment> build_segments(const PartMap& parts, int width, int height,
                                           const std::array<Point, 17>& pose) const {
        std::vector<ArmSegment> segments;
        if (kind_ == NarrowKind::Waist) {
            // COCO: 5/6 shoulders, 11/12 hips.
            if (!valid_point(pose[5]) || !valid_point(pose[6]) || !valid_point(pose[11]) ||
                !valid_point(pose[12])) {
                throw Error(PE_ERR_INVALID_ARGUMENT, "low_keypoint_confidence");
            }
            const Point shoulder{(pose[5].x + pose[6].x) * 0.5f, (pose[5].y + pose[6].y) * 0.5f};
            const Point hip{(pose[11].x + pose[12].x) * 0.5f, (pose[11].y + pose[12].y) * 0.5f};
            const Point start = shoulder.add(hip.sub(shoulder).mul(0.35f));
            const Point end = shoulder.add(hip.sub(shoulder).mul(0.85f));
            const auto half = estimate_width(parts, width, height, start, end, 2, 2);
            if (!half) {
                throw Error(PE_ERR_INVALID_ARGUMENT, "incomplete_waist_boundary");
            }
            segments.push_back({start, end, *half, true, true});
            return segments;
        }
        // Legs: left 11-13-15, right 12-14-16.
        const int chains[2][3] = {{11, 13, 15}, {12, 14, 16}};
        for (const auto& chain : chains) {
            if (!valid_point(pose[chain[0]]) || !valid_point(pose[chain[1]]) ||
                !valid_point(pose[chain[2]])) {
                throw Error(PE_ERR_INVALID_ARGUMENT, "low_keypoint_confidence");
            }
            const struct {
                Point start;
                Point end;
                uint8_t label;
                bool fade_start;
                bool fade_end;
            } specs[] = {
                {pose[chain[0]], pose[chain[1]], 5, true, false},
                {pose[chain[1]], pose[chain[2]], 6, false, true},
            };
            for (const auto& spec : specs) {
                const auto half =
                    estimate_width(parts, width, height, spec.start, spec.end, spec.label, 5);
                if (!half) {
                    throw Error(PE_ERR_INVALID_ARGUMENT, "incomplete_leg_boundary");
                }
                segments.push_back(
                    {spec.start, spec.end, *half, spec.fade_start, spec.fade_end});
            }
        }
        return segments;
    }
};

#endif  // PE_HAS_OPENCV

}  // namespace

std::unique_ptr<Feature> create_slim_waist_feature(const EngineConfig& config) {
#if PE_HAS_OPENCV
    return std::make_unique<NarrowFeature>(config, NarrowKind::Waist, "body.slim_waist",
                                           "waist-warp-v1");
#else
    (void)config;
    return nullptr;
#endif
}

std::unique_ptr<Feature> create_slim_legs_feature(const EngineConfig& config) {
#if PE_HAS_OPENCV
    return std::make_unique<NarrowFeature>(config, NarrowKind::Legs, "body.slim_legs",
                                           "legs-warp-v1");
#else
    (void)config;
    return nullptr;
#endif
}

}  // namespace pe
