#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "portrait_engine.h"

namespace pe {

class Error : public std::runtime_error {
public:
    Error(PeStatus status, const std::string& message)
        : std::runtime_error(message), status_(status) {}
    PeStatus status() const noexcept { return status_; }

private:
    PeStatus status_;
};

class Cancelled : public std::exception {
public:
    const char* what() const noexcept override { return "cancelled"; }
};

class CancelToken {
public:
    CancelToken(PeCancelFn fn, void* user) : fn_(fn), user_(user) {}
    void check() const {
        if (fn_ != nullptr && fn_(user_) != 0) {
            throw Cancelled();
        }
    }

private:
    PeCancelFn fn_;
    void* user_;
};

struct ImageView {
    PeFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t stride_bytes;
    const uint8_t* data;
};

struct EngineConfig {
    int threads;
    std::string resource_dir;
    std::string onnxruntime_path;
};

struct Region {
    float x0;
    float y0;
    float x1;
    float y1;
    uint32_t cells_x;
    uint32_t cells_y;
    std::vector<float> source_xy;
};

struct FeatureReport {
    std::string status;
    std::string reason;
};

class AnalysisData {
public:
    AnalysisData(uint32_t width, uint32_t height) : width(width), height(height) {}
    virtual ~AnalysisData() = default;
    virtual std::string diagnostics_json() const = 0;
    virtual std::vector<uint8_t> serialize_payload() const = 0;

    const uint32_t width;
    const uint32_t height;
};

/* 只含当前功能的参数；名称已由引擎按 Feature::param_names 校验。 */
class ParamView {
public:
    void add(const std::string& name, double value) { entries_.push_back({name, value}); }
    double get(const char* name, double fallback) const {
        for (const auto& entry : entries_) {
            if (entry.name == name) {
                return entry.value;
            }
        }
        return fallback;
    }

private:
    struct Entry {
        std::string name;
        double value;
    };
    std::vector<Entry> entries_;
};

struct ModelInfo {
    std::string file;
    std::string sha256;
};

class Feature {
public:
    virtual ~Feature() = default;
    virtual const char* id() const = 0;
    virtual const char* version() const = 0;
    virtual const char* kind() const = 0;
    virtual std::vector<std::string> param_names() const = 0;
    virtual std::vector<ModelInfo> models() const { return {}; }
    virtual std::unique_ptr<AnalysisData> analyze(const ImageView& image,
                                                  const CancelToken& cancel) = 0;
    virtual std::unique_ptr<AnalysisData> deserialize(uint32_t width, uint32_t height,
                                                      const uint8_t* payload,
                                                      size_t size) const = 0;
    virtual FeatureReport solve(const AnalysisData& analysis, const ParamView& params,
                                const CancelToken& cancel, std::vector<Region>& out) const = 0;
};

std::vector<std::unique_ptr<Feature>> create_debug_features();
std::vector<std::unique_ptr<Feature>> create_features(const EngineConfig& config);

std::string json_escape(const std::string& value);

}  // namespace pe
