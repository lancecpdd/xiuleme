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

namespace slim_arms_test {

float Point::length() const { return std::sqrt(dot(*this)); }

Point Point::normalized() const {
    const float len = length();
    if (!(len > 1e-4f) || !std::isfinite(len)) {
        return {};
    }
    return mul(1.0f / len);
}

float smooth(float value) {
    const float t = std::clamp(value, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float segment_distance(Point a, Point b, Point p) {
    const Point axis = b.sub(a);
    const float denom = std::max(axis.dot(axis), 1e-6f);
    const float t = std::clamp(p.sub(a).dot(axis) / denom, 0.0f, 1.0f);
    return p.sub(a.add(axis.mul(t))).length();
}

float orient(Point a, Point b, Point c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool segments_intersect(Point a0, Point a1, Point b0, Point b1) {
    const float d1 = orient(a0, a1, b0);
    const float d2 = orient(a0, a1, b1);
    const float d3 = orient(b0, b1, a0);
    const float d4 = orient(b0, b1, a1);
    const bool ab = (d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0);
    const bool cd = (d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0);
    return ab && cd;
}

bool bones_intersect(const ArmSegment* segments) {
    for (int left = 0; left < 2; ++left) {
        for (int right = 2; right < 4; ++right) {
            if (segments_intersect(segments[left].start, segments[left].end, segments[right].start,
                                   segments[right].end)) {
                return true;
            }
        }
    }
    return false;
}

bool arms_near(const ArmSegment* segments) {
    for (int left = 0; left < 2; ++left) {
        for (int right = 2; right < 4; ++right) {
            const float distance = std::min({
                segment_distance(segments[left].start, segments[left].end, segments[right].start),
                segment_distance(segments[left].start, segments[left].end, segments[right].end),
                segment_distance(segments[right].start, segments[right].end, segments[left].start),
                segment_distance(segments[right].start, segments[right].end, segments[left].end),
            });
            if (distance < std::min(segments[left].half_width, segments[right].half_width) * 0.8f) {
                return true;
            }
        }
    }
    return false;
}

void ownership_counts(const uint8_t* assign, size_t cells, size_t& left, size_t& right,
                      size_t& overlap) {
    left = right = overlap = 0;
    for (size_t i = 0; i < cells; ++i) {
        if (assign[i] == 1) ++left;
        else if (assign[i] == 2) ++right;
        else if (assign[i] == 3) ++overlap;
    }
}

CrossLevel classify_cross(const ArmSegment* segments, const uint8_t* ownership, size_t cells) {
    size_t left = 0, right = 0, overlap = 0;
    ownership_counts(ownership, cells, left, right, overlap);
    const size_t arm = left + right + overlap;
    const float overlap_ratio =
        arm > 0 ? static_cast<float>(overlap) / static_cast<float>(arm) : 0.0f;
    if (bones_intersect(segments) || overlap_ratio >= 0.08f) {
        return CrossLevel::Cross;
    }
    if (arms_near(segments) || overlap_ratio >= 0.02f) {
        return CrossLevel::Contact;
    }
    return CrossLevel::None;
}

Point source_point(Point point, ArmSegment segment, float strength, bool* ok) {
    *ok = false;
    const Point axis = segment.end.sub(segment.start);
    const float length = axis.length();
    const Point direction = axis.normalized();
    if (direction.x == 0 && direction.y == 0) {
        return point;
    }
    const Point normal{-direction.y, direction.x};
    const Point relative = point.sub(segment.start);
    const float t = relative.dot(direction) / length;
    const float start_pad = segment.fade_start ? 0.08f : 0.3f;
    const float end_pad = segment.fade_end ? 0.08f : 0.3f;
    if (t < -start_pad || t > 1.0f + end_pad) {
        return point;
    }
    const float distance = relative.dot(normal);
    const float absolute = std::fabs(distance);
    const float outer = segment.half_width * 1.45f;
    if (absolute >= outer) {
        return point;
    }
    float envelope = 1.0f;
    if (segment.fade_start) {
        envelope *= smooth(t / 0.2f);
    }
    if (segment.fade_end) {
        envelope *= smooth((1.0f - t) / 0.2f);
    }
    envelope *= smooth((t + start_pad) / start_pad) * smooth((1.0f + end_pad - t) / end_pad);
    const float scale = std::clamp(1.0f - strength * envelope, 0.72f, 1.0f);
    const float core = segment.half_width * scale;
    const float source_distance =
        absolute <= core
            ? absolute / scale
            : absolute + (segment.half_width - core) *
                             smooth((outer - absolute) / std::max(outer - core, 1e-4f));
    const float displacement = (source_distance - absolute) * (distance < 0 ? -1.0f : 1.0f);
    *ok = true;
    return point.add(normal.mul(displacement));
}

}  // namespace slim_arms_test

namespace {

constexpr char kFeatureId[] = "body.slim_arms";
constexpr char kFeatureVersion[] = "arm-warp-v2";
constexpr char kPoseFile[] = "rtmpose-s-256x192.onnx";
constexpr char kParsingFile[] = "schp-pascal-7-int8.onnx";
constexpr char kPoseSha[] = "d869413ccfbc0ed027337c773f124d6362436239a83fd0a6e2d43b7a8377792e";
constexpr char kParsingSha[] = "66b12766d7f1ddbc3de972e67e8626be727507e7feeeca34e1b23b6f45e756d2";

constexpr int kGrid = 512;
constexpr int kPoseW = 192;
constexpr int kPoseH = 256;
constexpr uint8_t kOwnerNone = 0;
constexpr uint8_t kOwnerLeft = 1;
constexpr uint8_t kOwnerRight = 2;
constexpr uint8_t kOwnerOverlap = 3;
constexpr float kOwnershipRadius = 1.35f;
constexpr float kOverlapAmbiguity = 0.35f;

using slim_arms_test::ArmSegment;
using slim_arms_test::CrossLevel;
using slim_arms_test::Point;
using slim_arms_test::segment_distance;
using slim_arms_test::smooth;

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
    const std::string actual = sha256_file(path);
    if (actual != expected_sha) {
        throw Error(PE_ERR_MODEL, std::string("model hash mismatch: ") + file);
    }
}

cv::Mat image_to_bgr(const ImageView& image) {
    if (image.format != PE_FORMAT_RGBA8) {
        throw Error(PE_ERR_UNSUPPORTED, "slim arms currently requires RGBA8 input");
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
                // OpenCV BGR -> network RGB channel order.
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
    int bounds[4] = {0, 0, kGrid, kGrid};  // x0,y0,x1,y1 in grid cells

    std::pair<uint8_t, float> at_image(Point point, int width, int height) const {
        const int x = static_cast<int>(std::floor((point.x / std::max(width, 1)) * kGrid));
        const int y = static_cast<int>(std::floor((point.y / std::max(height, 1)) * kGrid));
        const int cx = std::clamp(x, 0, kGrid - 1);
        const int cy = std::clamp(y, 0, kGrid - 1);
        const size_t index = static_cast<size_t>(cy * kGrid + cx);
        return {labels[index], confidence[index]};
    }
};

struct Component {
    int area = 0;
    int bounds[4] = {0, 0, 0, 0};
};

std::vector<Component> foreground_components(const std::vector<uint8_t>& labels) {
    cv::Mat mask(kGrid, kGrid, CV_8UC1);
    for (int i = 0; i < kGrid * kGrid; ++i) {
        mask.data[i] = labels[static_cast<size_t>(i)] == 0 ? 0 : 255;
    }
    cv::Mat labels_cc, stats, centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels_cc, stats, centroids, 8);
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
    return components;
}

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

class ArmAnalysisData final : public AnalysisData {
public:
    ArmAnalysisData(uint32_t width, uint32_t height) : AnalysisData(width, height) {}

    std::string diagnostics_json() const override {
        if (!bypass_reason.empty()) {
            return std::string(R"({"status":"bypass","reason":")") + json_escape(bypass_reason) +
                   "\"}";
        }
        return R"({"status":"ready"})";
    }

    std::vector<uint8_t> serialize_payload() const override {
        // Payload version 1: bypass string OR full analysis fields.
        std::vector<uint8_t> out;
        auto put_u32 = [&](uint32_t value) {
            for (int shift = 0; shift < 32; shift += 8) {
                out.push_back(static_cast<uint8_t>((value >> shift) & 0xff));
            }
        };
        auto put_f32 = [&](float value) {
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(bits));
            put_u32(bits);
        };
        put_u32(1);
        put_u32(static_cast<uint32_t>(bypass_reason.size()));
        out.insert(out.end(), bypass_reason.begin(), bypass_reason.end());
        if (!bypass_reason.empty()) {
            return out;
        }
        put_u32(static_cast<uint32_t>(parts.labels.size()));
        out.insert(out.end(), parts.labels.begin(), parts.labels.end());
        for (float value : parts.confidence) put_f32(value);
        for (int i = 0; i < 4; ++i) put_u32(static_cast<uint32_t>(parts.bounds[i]));
        put_u32(static_cast<uint32_t>(ownership.size()));
        out.insert(out.end(), ownership.begin(), ownership.end());
        put_u32(static_cast<uint32_t>(edge_scale.size()));
        for (float value : edge_scale) put_f32(value);
        put_u32(static_cast<uint32_t>(cross));
        put_u32(front_arm < 0 ? 0xffffffffu : static_cast<uint32_t>(front_arm));
        for (const auto& segment : segments) {
            put_f32(segment.start.x);
            put_f32(segment.start.y);
            put_f32(segment.end.x);
            put_f32(segment.end.y);
            put_f32(segment.half_width);
            put_u32(segment.fade_start ? 1u : 0u);
            put_u32(segment.fade_end ? 1u : 0u);
        }
        for (const auto& point : pose_points) {
            put_f32(point.x);
            put_f32(point.y);
        }
        return out;
    }

    std::string bypass_reason;
    PartMap parts;
    std::vector<uint8_t> ownership;
    std::vector<float> edge_scale;
    std::vector<ArmSegment> segments;
    Point pose_points[6]{};
    CrossLevel cross = CrossLevel::None;
    int front_arm = -1;
};

// Full deserialize implementation (replaces stub above).
std::unique_ptr<ArmAnalysisData> deserialize_arm(uint32_t width, uint32_t height,
                                                 const uint8_t* data, size_t size) {
    size_t offset = 0;
    auto need = [&](size_t count) {
        if (count > size - offset) {
            throw Error(PE_ERR_INVALID_ARGUMENT, "slim arms analysis blob truncated");
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
        throw Error(PE_ERR_UNSUPPORTED, "unsupported slim arms analysis payload version");
    }
    auto analysis = std::make_unique<ArmAnalysisData>(width, height);
    const uint32_t reason_len = u32();
    const auto* reason = reinterpret_cast<const char*>(bytes(reason_len));
    analysis->bypass_reason.assign(reason, reason_len);
    if (!analysis->bypass_reason.empty()) {
        if (offset != size) {
            throw Error(PE_ERR_INVALID_ARGUMENT, "slim arms analysis blob has trailing bytes");
        }
        return analysis;
    }
    const uint32_t label_count = u32();
    if (label_count != static_cast<uint32_t>(kGrid * kGrid)) {
        throw Error(PE_ERR_INVALID_ARGUMENT, "slim arms label grid size mismatch");
    }
    const auto* labels = bytes(label_count);
    analysis->parts.labels.assign(labels, labels + label_count);
    analysis->parts.confidence.resize(label_count);
    for (uint32_t i = 0; i < label_count; ++i) {
        analysis->parts.confidence[i] = f32();
    }
    for (int i = 0; i < 4; ++i) {
        analysis->parts.bounds[i] = static_cast<int>(u32());
    }
    const uint32_t owner_count = u32();
    const auto* owners = bytes(owner_count);
    analysis->ownership.assign(owners, owners + owner_count);
    const uint32_t edge_count = u32();
    analysis->edge_scale.resize(edge_count);
    for (uint32_t i = 0; i < edge_count; ++i) {
        analysis->edge_scale[i] = f32();
    }
    analysis->cross = static_cast<CrossLevel>(u32());
    const uint32_t front = u32();
    analysis->front_arm = front == 0xffffffffu ? -1 : static_cast<int>(front);
    analysis->segments.resize(4);
    for (auto& segment : analysis->segments) {
        segment.start.x = f32();
        segment.start.y = f32();
        segment.end.x = f32();
        segment.end.y = f32();
        segment.half_width = f32();
        segment.fade_start = u32() != 0;
        segment.fade_end = u32() != 0;
    }
    for (auto& point : analysis->pose_points) {
        point.x = f32();
        point.y = f32();
    }
    if (offset != size) {
        throw Error(PE_ERR_INVALID_ARGUMENT, "slim arms analysis blob has trailing bytes");
    }
    return analysis;
}

class SlimArmsFeature final : public Feature {
public:
    explicit SlimArmsFeature(EngineConfig config) : config_(std::move(config)) {}

    const char* id() const override { return kFeatureId; }
    const char* version() const override { return kFeatureVersion; }
    const char* kind() const override { return "warp"; }
    std::vector<std::string> param_names() const override { return {"amount"}; }
    std::vector<ModelInfo> models() const override {
        return {{kPoseFile, kPoseSha}, {kParsingFile, kParsingSha}};
    }

    std::unique_ptr<AnalysisData> analyze(const ImageView& image,
                                          const CancelToken& cancel) override {
        ensure_sessions(cancel);
        cancel.check();
        cv::Mat bgr = image_to_bgr(image);
        auto analysis = std::make_unique<ArmAnalysisData>(image.width, image.height);
        try {
            run_pipeline(bgr, *analysis, cancel);
        } catch (const Error& error) {
            if (error.status() == PE_ERR_MODEL || error.status() == PE_ERR_INTERNAL ||
                error.status() == PE_ERR_OUT_OF_MEMORY || error.status() == PE_ERR_UNSUPPORTED) {
                throw;
            }
            // Geometry / detection failures become soft bypass for the product UI.
            analysis->bypass_reason = map_bypass(error.what());
            analysis->segments.clear();
            analysis->ownership.clear();
            analysis->edge_scale.clear();
            analysis->parts = {};
        }
        return analysis;
    }

    std::unique_ptr<AnalysisData> deserialize(uint32_t width, uint32_t height, const uint8_t* payload,
                                              size_t size) const override {
        return deserialize_arm(width, height, payload, size);
    }

    FeatureReport solve(const AnalysisData& analysis, const ParamView& params,
                        const CancelToken& cancel, std::vector<Region>& out) const override {
        const double amount = params.get("amount", 0.0);
        if (!std::isfinite(amount) || amount < 0.0 || amount > 100.0) {
            throw Error(PE_ERR_INVALID_ARGUMENT, "amount must be within [0, 100]");
        }
        const auto& arm = dynamic_cast<const ArmAnalysisData&>(analysis);
        if (!arm.bypass_reason.empty()) {
            return {"bypass", arm.bypass_reason};
        }
        if (amount == 0.0) {
            return {"identity", ""};
        }
        cancel.check();
        const float response = 0.15f * smooth(static_cast<float>(amount / 100.0));
        const int width = static_cast<int>(analysis.width);
        const int height = static_cast<int>(analysis.height);
        for (int arm_index = 0; arm_index < 2; ++arm_index) {
            cancel.check();
            build_arm_region(arm, arm_index, response, width, height, out);
        }
        return out.empty() ? FeatureReport{"identity", ""} : FeatureReport{"applied", ""};
    }

private:
    EngineConfig config_;
    std::unique_ptr<OrtRuntime::Session> pose_;
    std::unique_ptr<OrtRuntime::Session> parsing_;

    static std::string map_bypass(const std::string& message) {
        if (message.find("no_primary_person") != std::string::npos) return "no_primary_person";
        if (message.find("multiple_people") != std::string::npos) return "multiple_people";
        if (message.find("low_keypoint_confidence") != std::string::npos)
            return "low_keypoint_confidence";
        if (message.find("incomplete_arm_boundary") != std::string::npos)
            return "incomplete_arm_boundary";
        if (message.find("arms_crossed") != std::string::npos) return "arms_crossed";
        return message;
    }

    void ensure_sessions(const CancelToken& cancel) {
        if (pose_ && parsing_) {
            return;
        }
        cancel.check();
        if (config_.resource_dir.empty()) {
            throw Error(PE_ERR_MODEL, "resource directory is empty");
        }
        require_model(config_.resource_dir, kPoseFile, kPoseSha);
        require_model(config_.resource_dir, kParsingFile, kParsingSha);
        OrtRuntime::instance().ensure_loaded(config_.onnxruntime_path);
        const int threads = config_.threads > 0 ? config_.threads : 2;
        pose_ = std::make_unique<OrtRuntime::Session>(
            OrtRuntime::instance(), join_resource(config_.resource_dir, kPoseFile), threads);
        parsing_ = std::make_unique<OrtRuntime::Session>(
            OrtRuntime::instance(), join_resource(config_.resource_dir, kParsingFile), threads);
    }

    void run_pipeline(const cv::Mat& bgr, ArmAnalysisData& analysis, const CancelToken& cancel) {
        analysis.parts = run_parsing(bgr, cancel);
        analysis.pose_points[0] = {};  // filled by pose
        std::array<Point, 6> points = run_pose(bgr, analysis.parts, cancel);
        std::copy(points.begin(), points.end(), analysis.pose_points);
        analysis.segments = build_segments(analysis.parts, bgr.cols, bgr.rows, points);
        analysis.ownership =
            assign_ownership(analysis.parts, bgr.cols, bgr.rows, analysis.segments);
        analysis.cross = classify_cross(analysis.segments.data(), analysis.ownership.data(),
                                        analysis.ownership.size());
        if (analysis.cross == CrossLevel::Cross) {
            analysis.front_arm =
                choose_front_arm(analysis.parts, bgr.cols, bgr.rows, analysis.segments,
                                 analysis.ownership, points);
        }
        analysis.edge_scale = build_edge_scale(bgr);
    }

    PartMap run_parsing(const cv::Mat& bgr, const CancelToken& cancel) {
        cancel.check();
        std::vector<float> input;
        nchw_from_rect(bgr, kGrid, kGrid, cv::Rect2f(0, 0, static_cast<float>(bgr.cols),
                                                     static_cast<float>(bgr.rows)),
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
        const auto components = foreground_components(parts.labels);
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

    cv::Rect2f pose_rect(const PartMap& parts, int width, int height) const {
        float x0 = parts.bounds[0] * static_cast<float>(width) / kGrid;
        float y0 = parts.bounds[1] * static_cast<float>(height) / kGrid;
        float x1 = parts.bounds[2] * static_cast<float>(width) / kGrid;
        float y1 = parts.bounds[3] * static_cast<float>(height) / kGrid;
        const Point center{(x0 + x1) * 0.5f, (y0 + y1) * 0.5f};
        float w = (x1 - x0) * 1.2f;
        float h = (y1 - y0) * 1.2f;
        const float aspect = static_cast<float>(kPoseW) / static_cast<float>(kPoseH);
        if (w / h < aspect) {
            w = h * aspect;
        } else {
            h = w / aspect;
        }
        return {center.x - w * 0.5f, center.y - h * 0.5f, w, h};
    }

    std::array<Point, 6> run_pose(const cv::Mat& bgr, const PartMap& parts,
                                  const CancelToken& cancel) {
        cancel.check();
        const cv::Rect2f rect = pose_rect(parts, bgr.cols, bgr.rows);
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
        std::array<Point, 6> points{};
        const int keypoints[6] = {5, 7, 9, 6, 8, 10};
        for (int i = 0; i < 6; ++i) {
            const int keypoint = keypoints[i];
            const auto [px, sx] =
                simcc_peak(simcc_x + keypoint * kPoseW * 2, kPoseW * 2);
            const auto [py, sy] =
                simcc_peak(simcc_y + keypoint * kPoseH * 2, kPoseH * 2);
            if (!std::isfinite(sx) || !std::isfinite(sy) || std::min(sx, sy) < 0.2f) {
                throw Error(PE_ERR_INVALID_ARGUMENT, "low_keypoint_confidence");
            }
            points[static_cast<size_t>(i)] = {
                rect.x + (px * 0.5f / kPoseW) * rect.width,
                rect.y + (py * 0.5f / kPoseH) * rect.height,
            };
        }
        return points;
    }

    static std::optional<float> estimate_width(const PartMap& parts, int width, int height,
                                               Point start, Point end, uint8_t label) {
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
                    if ((part == label || part == 3 || part == 4) && confidence >= 0.3f) {
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
        return std::clamp(widths[widths.size() / 2], length * 0.06f, length * 0.28f);
    }

    static std::vector<ArmSegment> build_segments(const PartMap& parts, int width, int height,
                                                  const std::array<Point, 6>& points) {
        std::vector<ArmSegment> output;
        for (int arm = 0; arm < 2; ++arm) {
            const Point shoulder = points[static_cast<size_t>(arm * 3)];
            const Point elbow = points[static_cast<size_t>(arm * 3 + 1)];
            const Point wrist = points[static_cast<size_t>(arm * 3 + 2)];
            const struct {
                Point start;
                Point end;
                uint8_t label;
                bool fade_start;
                bool fade_end;
            } specs[] = {
                {shoulder, elbow, 3, true, false},
                {elbow, wrist, 4, false, true},
            };
            for (const auto& spec : specs) {
                const auto half_width =
                    estimate_width(parts, width, height, spec.start, spec.end, spec.label);
                if (!half_width) {
                    throw Error(PE_ERR_INVALID_ARGUMENT, "incomplete_arm_boundary");
                }
                output.push_back(
                    {spec.start, spec.end, *half_width, spec.fade_start, spec.fade_end});
            }
        }
        return output;
    }

    static std::optional<std::pair<int, float>> nearest_bone(Point point,
                                                             const std::vector<ArmSegment>& segments,
                                                             int begin, int end) {
        std::optional<std::pair<int, float>> best;
        for (int index = begin; index < end; ++index) {
            const float distance =
                segment_distance(segments[static_cast<size_t>(index)].start,
                                 segments[static_cast<size_t>(index)].end, point);
            const float limit = segments[static_cast<size_t>(index)].half_width * kOwnershipRadius;
            if (distance > limit) {
                continue;
            }
            if (!best || distance < best->second) {
                best = std::make_pair(index, distance);
            }
        }
        return best;
    }

    static std::vector<uint8_t> assign_ownership(const PartMap& parts, int width, int height,
                                                 const std::vector<ArmSegment>& segments) {
        std::vector<uint8_t> assign(static_cast<size_t>(kGrid * kGrid), kOwnerNone);
        const float sx = width / static_cast<float>(kGrid);
        const float sy = height / static_cast<float>(kGrid);
        for (int y = 0; y < kGrid; ++y) {
            for (int x = 0; x < kGrid; ++x) {
                const size_t index = static_cast<size_t>(y * kGrid + x);
                const uint8_t label = parts.labels[index];
                if (label != 3 && label != 4) {
                    continue;
                }
                const Point point{(x + 0.5f) * sx, (y + 0.5f) * sy};
                const auto left = nearest_bone(point, segments, 0, 2);
                const auto right = nearest_bone(point, segments, 2, 4);
                if (left && right) {
                    const float limit =
                        std::min(segments[0].half_width, segments[2].half_width) * kOverlapAmbiguity;
                    if (std::fabs(left->second - right->second) <= std::max(limit, 1.5f)) {
                        assign[index] = kOwnerOverlap;
                    } else {
                        assign[index] = left->second < right->second ? kOwnerLeft : kOwnerRight;
                    }
                } else if (left) {
                    assign[index] = kOwnerLeft;
                } else if (right) {
                    assign[index] = kOwnerRight;
                }
            }
        }
        return assign;
    }

    static int choose_front_arm(const PartMap& parts, int width, int height,
                                const std::vector<ArmSegment>& segments,
                                const std::vector<uint8_t>& ownership,
                                const std::array<Point, 6>& points) {
        const Point torso{
            (parts.bounds[0] + parts.bounds[2]) * 0.5f / kGrid * width,
            (parts.bounds[1] + parts.bounds[3]) * 0.5f / kGrid * height,
        };
        const float left_score = points[2].sub(torso).length() + points[2].y * 0.2f;
        const float right_score = points[5].sub(torso).length() + points[5].y * 0.2f;
        if (std::fabs(left_score - right_score) > 10.0f) {
            return right_score > left_score ? 1 : 0;
        }
        float conf[2] = {0, 0};
        int count[2] = {0, 0};
        const float sx = width / static_cast<float>(kGrid);
        const float sy = height / static_cast<float>(kGrid);
        for (int y = 0; y < kGrid; ++y) {
            for (int x = 0; x < kGrid; ++x) {
                const size_t index = static_cast<size_t>(y * kGrid + x);
                if (ownership[index] != kOwnerOverlap) {
                    continue;
                }
                const Point point{(x + 0.5f) * sx, (y + 0.5f) * sy};
                const auto left = nearest_bone(point, segments, 0, 2);
                const auto right = nearest_bone(point, segments, 2, 4);
                int side = -1;
                if (left && right) {
                    side = left->second <= right->second ? 0 : 1;
                } else if (left) {
                    side = 0;
                } else if (right) {
                    side = 1;
                }
                if (side < 0) {
                    continue;
                }
                conf[side] += parts.confidence[index];
                count[side] += 1;
            }
        }
        if (count[0] > 0 && count[1] > 0) {
            const float avg0 = conf[0] / count[0];
            const float avg1 = conf[1] / count[1];
            if (std::fabs(avg0 - avg1) > 0.04f) {
                return avg1 > avg0 ? 1 : 0;
            }
        }
        const float left_w = segments[0].half_width + segments[1].half_width;
        const float right_w = segments[2].half_width + segments[3].half_width;
        const float wider = std::max({left_w, right_w, 1e-3f});
        if (std::fabs(left_w - right_w) / wider > 0.08f) {
            return right_w > left_w ? 1 : 0;
        }
        throw Error(PE_ERR_INVALID_ARGUMENT, "arms_crossed");
    }

    static std::vector<float> build_edge_scale(const cv::Mat& bgr) {
        cv::Mat gray, gx, gy, mag;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32F, 1.0 / 255.0);
        cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
        cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, mag);
        cv::Mat small;
        cv::resize(mag, small, cv::Size(kGrid, kGrid), 0, 0, cv::INTER_AREA);
        std::vector<float> scale(static_cast<size_t>(kGrid * kGrid));
        for (int i = 0; i < kGrid * kGrid; ++i) {
            const float edge = small.at<float>(i / kGrid, i % kGrid);
            // Approximate the original 1px finite-difference scale with Sobel magnitude.
            scale[static_cast<size_t>(i)] = 1.0f - 0.85f * smooth((edge - 0.20f) / 0.16f);
        }
        return scale;
    }

    static bool arm_owns_point(const ArmAnalysisData& analysis, int arm, Point point, int width,
                               int height) {
        const int x = std::clamp(static_cast<int>(std::floor((point.x / width) * kGrid)), 0,
                                 kGrid - 1);
        const int y = std::clamp(static_cast<int>(std::floor((point.y / height) * kGrid)), 0,
                                 kGrid - 1);
        const uint8_t owner =
            analysis.ownership[static_cast<size_t>(y * kGrid + x)];
        const uint8_t mine = arm == 0 ? kOwnerLeft : kOwnerRight;
        const uint8_t other = arm == 0 ? kOwnerRight : kOwnerLeft;
        if (owner == mine) return true;
        if (owner == kOwnerOverlap) {
            if (analysis.cross == CrossLevel::Cross) {
                return analysis.front_arm == arm;
            }
            return true;
        }
        if (owner == other) return false;
        const int mine_begin = arm == 0 ? 0 : 2;
        const int other_begin = arm == 0 ? 2 : 0;
        const auto mine_dist = nearest_bone(point, analysis.segments, mine_begin, mine_begin + 2);
        const auto other_dist =
            nearest_bone(point, analysis.segments, other_begin, other_begin + 2);
        if (mine_dist && other_dist) return mine_dist->second <= other_dist->second + 1.0f;
        return static_cast<bool>(mine_dist);
    }

    static Point arm_source(const ArmAnalysisData& analysis, int arm, Point point, float strength,
                            int width, int height) {
        const ArmSegment* segments = analysis.segments.data() + arm * 2;
        if (!arm_owns_point(analysis, arm, point, width, height)) {
            return point;
        }
        const auto [label, _] = analysis.parts.at_image(point, width, height);
        auto protected_point = [&](Point sample) {
            const auto [part, confidence] = analysis.parts.at_image(sample, width, height);
            return confidence > 0.55f && (part == 1 || part == 2 || part == 5 || part == 6);
        };
        if (protected_point(point)) {
            return point;
        }
        Point delta{};
        float total = 0.0f;
        for (int i = 0; i < 2; ++i) {
            bool ok = false;
            const Point source = slim_arms_test::source_point(point, segments[i], strength, &ok);
            if (!ok) continue;
            const float displacement = source.sub(point).length();
            const float weight = std::max(displacement, 1e-4f);
            delta = delta.add(source.sub(point).mul(weight));
            total += weight;
        }
        if (total == 0.0f) {
            return point;
        }
        const float protection_radius =
            std::max({segments[0].half_width, segments[1].half_width, 4.0f});
        const Point directions[] = {{-1, 0}, {1, 0},  {0, -1}, {0, 1},
                                    {-0.707f, -0.707f}, {0.707f, -0.707f},
                                    {-0.707f, 0.707f},  {0.707f, 0.707f}};
        float protection_scale = 1.0f;
        constexpr int kRings = 24;
        for (int ring = 1; ring <= kRings; ++ring) {
            const float distance = protection_radius * ring / kRings;
            bool hit = false;
            for (const Point& direction : directions) {
                if (protected_point(point.add(direction.mul(distance)))) {
                    hit = true;
                    break;
                }
            }
            if (hit) {
                protection_scale = smooth(static_cast<float>(ring - 1) / kRings);
                break;
            }
        }
        Point source = point.add(delta.mul(protection_scale / total));
        const bool outside = segment_distance(segments[0].start, segments[0].end, point) >
                                 segments[0].half_width * 1.12f &&
                             segment_distance(segments[1].start, segments[1].end, point) >
                                 segments[1].half_width * 1.12f;
        if (label == 0 && outside) {
            const int gx = std::clamp(static_cast<int>(std::floor((point.x / width) * kGrid)), 0,
                                      kGrid - 1);
            const int gy = std::clamp(static_cast<int>(std::floor((point.y / height) * kGrid)), 0,
                                      kGrid - 1);
            const float edge = analysis.edge_scale[static_cast<size_t>(gy * kGrid + gx)];
            source = point.add(source.sub(point).mul(edge));
        }
        return source;
    }

    static void build_arm_region(const ArmAnalysisData& analysis, int arm, float strength,
                                 int width, int height, std::vector<Region>& out) {
        const ArmSegment* segments = analysis.segments.data() + arm * 2;
        const float radius =
            std::max(segments[0].half_width, segments[1].half_width) * 1.6f;
        const float min_x =
            std::max(0.0f, std::min({segments[0].start.x, segments[0].end.x, segments[1].start.x,
                                     segments[1].end.x}) -
                               radius);
        const float min_y =
            std::max(0.0f, std::min({segments[0].start.y, segments[0].end.y, segments[1].start.y,
                                     segments[1].end.y}) -
                               radius);
        const float max_x =
            std::min(static_cast<float>(width),
                     std::max({segments[0].start.x, segments[0].end.x, segments[1].start.x,
                               segments[1].end.x}) +
                         radius);
        const float max_y =
            std::min(static_cast<float>(height),
                     std::max({segments[0].start.y, segments[0].end.y, segments[1].start.y,
                               segments[1].end.y}) +
                         radius);
        if (!(max_x > min_x && max_y > min_y)) {
            return;
        }
        const uint32_t nx = std::clamp(static_cast<uint32_t>(std::ceil((max_x - min_x) / 12.0f)),
                                       8u, 96u);
        const uint32_t ny = std::clamp(static_cast<uint32_t>(std::ceil((max_y - min_y) / 12.0f)),
                                       8u, 96u);
        // Avoid colliding with the face ARAP control-net vertex count (217).
        uint32_t cells_x = nx;
        uint32_t cells_y = ny;
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
                const Point source = arm_source(analysis, arm, point, strength, width, height);
                region.source_xy.push_back(source.x);
                region.source_xy.push_back(source.y);
                if (std::fabs(source.x - point.x) > 1e-3f ||
                    std::fabs(source.y - point.y) > 1e-3f) {
                    moved = true;
                }
            }
        }
        if (moved) {
            out.push_back(std::move(region));
        }
    }
};

#endif  // PE_HAS_OPENCV

}  // namespace

std::unique_ptr<Feature> create_slim_arms_feature(const EngineConfig& config) {
#if PE_HAS_OPENCV
    return std::make_unique<SlimArmsFeature>(config);
#else
    (void)config;
    return nullptr;
#endif
}

}  // namespace pe
