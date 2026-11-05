#include <algorithm>
#include <cmath>

#include "feature.h"
#include "feature_body_narrow.h"
#include "feature_slim_arms.h"

namespace pe {
namespace {

class EmptyAnalysis final : public AnalysisData {
public:
    using AnalysisData::AnalysisData;
    std::string diagnostics_json() const override { return R"({"status":"ready"})"; }
    std::vector<uint8_t> serialize_payload() const override { return {}; }
};

double smooth(double value) {
    const double t = std::clamp(value, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

double checked_amount(const ParamView& params) {
    const double amount = params.get("amount", 0.0);
    if (!std::isfinite(amount) || amount < 0.0 || amount > 100.0) {
        throw Error(PE_ERR_INVALID_ARGUMENT, "amount must be within [0, 100]");
    }
    return amount;
}

class DebugFeature : public Feature {
public:
    const char* kind() const override { return "debug"; }
    std::vector<std::string> param_names() const override { return {"amount"}; }

    std::unique_ptr<AnalysisData> analyze(const ImageView& image,
                                          const CancelToken& cancel) override {
        cancel.check();
        return std::make_unique<EmptyAnalysis>(image.width, image.height);
    }

    std::unique_ptr<AnalysisData> deserialize(uint32_t width, uint32_t height, const uint8_t*,
                                              size_t size) const override {
        if (size != 0) {
            throw Error(PE_ERR_INVALID_ARGUMENT, "debug analysis payload must be empty");
        }
        return std::make_unique<EmptyAnalysis>(width, height);
    }
};

class IdentityFeature final : public DebugFeature {
public:
    const char* id() const override { return "debug.identity"; }
    const char* version() const override { return "debug-identity-v1"; }

    FeatureReport solve(const AnalysisData&, const ParamView& params, const CancelToken& cancel,
                        std::vector<Region>&) const override {
        checked_amount(params);
        cancel.check();
        return {"identity", ""};
    }
};

/* 以图像中心为圆心的径向收缩，只用于贯通宿主的「网格 → 规范场」链路。 */
class PinchFeature final : public DebugFeature {
public:
    const char* id() const override { return "debug.pinch"; }
    const char* version() const override { return "debug-pinch-v1"; }

    FeatureReport solve(const AnalysisData& analysis, const ParamView& params,
                        const CancelToken& cancel, std::vector<Region>& out) const override {
        const double amount = checked_amount(params);
        if (amount == 0.0) {
            return {"identity", ""};
        }
        const double cx = analysis.width * 0.5;
        const double cy = analysis.height * 0.5;
        const double radius = 0.25 * std::min(analysis.width, analysis.height);
        if (radius < 4.0) {
            return {"bypass", "image_too_small"};
        }
        // 源半径 r(1+s(1-t)²) 的导数下界为 1-s/3，s≤0.2 时不会翻折；t≥1 处精确恒等。
        const double strength = 0.2 * smooth(amount / 100.0);
        constexpr uint32_t cells = 16;
        Region region;
        region.x0 = static_cast<float>(cx - radius);
        region.y0 = static_cast<float>(cy - radius);
        region.x1 = static_cast<float>(cx + radius);
        region.y1 = static_cast<float>(cy + radius);
        region.cells_x = cells;
        region.cells_y = cells;
        region.source_xy.reserve(static_cast<size_t>(cells + 1) * (cells + 1) * 2);
        for (uint32_t j = 0; j <= cells; ++j) {
            cancel.check();
            const double py = region.y0 + (static_cast<double>(region.y1) - region.y0) * j / cells;
            for (uint32_t i = 0; i <= cells; ++i) {
                const double px =
                    region.x0 + (static_cast<double>(region.x1) - region.x0) * i / cells;
                const double t = std::hypot(px - cx, py - cy) / radius;
                const double scale = t >= 1.0 ? 1.0 : 1.0 + strength * (1.0 - t) * (1.0 - t);
                region.source_xy.push_back(static_cast<float>(cx + (px - cx) * scale));
                region.source_xy.push_back(static_cast<float>(cy + (py - cy) * scale));
            }
        }
        out.push_back(std::move(region));
        return {"applied", ""};
    }
};

}  // namespace

std::vector<std::unique_ptr<Feature>> create_debug_features() {
    std::vector<std::unique_ptr<Feature>> features;
    features.push_back(std::make_unique<IdentityFeature>());
    features.push_back(std::make_unique<PinchFeature>());
    return features;
}

std::vector<std::unique_ptr<Feature>> create_features(const EngineConfig& config) {
    std::vector<std::unique_ptr<Feature>> features = create_debug_features();
    if (auto slim_arms = create_slim_arms_feature(config)) {
        features.push_back(std::move(slim_arms));
    }
    if (auto slim_waist = create_slim_waist_feature(config)) {
        features.push_back(std::move(slim_waist));
    }
    if (auto slim_legs = create_slim_legs_feature(config)) {
        features.push_back(std::move(slim_legs));
    }
    return features;
}

}  // namespace pe
