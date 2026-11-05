#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "feature.h"

#if PE_HAS_OPENCV
#include <opencv2/core/version.hpp>
#endif

struct PeEngine {
    pe::EngineConfig config;
    std::vector<std::unique_ptr<pe::Feature>> features;
    std::string capabilities;
    std::mutex mutex;

    pe::Feature* find(const char* id) {
        for (const auto& feature : features) {
            if (std::strcmp(feature->id(), id) == 0) {
                return feature.get();
            }
        }
        return nullptr;
    }
};

struct PeAnalysis {
    const PeEngine* owner;
    const pe::Feature* feature;
    std::unique_ptr<pe::AnalysisData> data;
    std::string diagnostics;
};

struct PeWarpResult {
    std::vector<pe::Region> regions;
    std::string diagnostics;
};

namespace {

constexpr char kEngineVersion[] = "portrait-engine-v1";
constexpr uint8_t kMagic[4] = {'P', 'E', 'A', '1'};
constexpr uint32_t kMaxDimension = 65535;
constexpr uint32_t kMaxCells = 512;
/* 与宿主 SafeScene 单场景上限一致。 */
constexpr size_t kMaxRegions = 32;
constexpr size_t kMaxIdLength = 128;

thread_local std::string g_last_error;

template <class Body>
PeStatus guarded(Body&& body) noexcept {
    try {
        return body();
    } catch (const pe::Cancelled&) {
        g_last_error = "cancelled";
        return PE_CANCELLED;
    } catch (const pe::Error& error) {
        g_last_error = error.what();
        return error.status();
    } catch (const std::bad_alloc&) {
        g_last_error = "out of memory";
        return PE_ERR_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return PE_ERR_INTERNAL;
    } catch (...) {
        g_last_error = "unknown native exception";
        return PE_ERR_INTERNAL;
    }
}

void require(bool condition, PeStatus status, const std::string& message) {
    if (!condition) {
        throw pe::Error(status, message);
    }
}

pe::ImageView checked_image(const PeImage* image) {
    require(image != nullptr && image->struct_size >= sizeof(PeImage), PE_ERR_INVALID_ARGUMENT,
            "image descriptor is missing or too small");
    uint64_t bytes_per_pixel = 0;
    switch (image->format) {
        case PE_FORMAT_RGBA8:
            bytes_per_pixel = 4;
            break;
        case PE_FORMAT_RGBA32F:
            bytes_per_pixel = 16;
            break;
        default:
            throw pe::Error(PE_ERR_UNSUPPORTED, "unsupported image format");
    }
    require(image->width > 0 && image->height > 0 && image->width <= kMaxDimension &&
                image->height <= kMaxDimension,
            PE_ERR_INVALID_ARGUMENT, "image dimensions out of range");
    require(image->stride_bytes >= image->width * bytes_per_pixel, PE_ERR_INVALID_ARGUMENT,
            "image stride is smaller than one row");
    require(image->data != nullptr, PE_ERR_INVALID_ARGUMENT, "image data is null");
    return {image->format, image->width, image->height, image->stride_bytes,
            static_cast<const uint8_t*>(image->data)};
}

void require_id(const char* id, const char* what) {
    require(id != nullptr && id[0] != '\0' && std::strlen(id) <= kMaxIdLength,
            PE_ERR_INVALID_ARGUMENT, std::string(what) + " is missing or too long");
}

std::string capabilities_json(const PeEngine& engine) {
    std::string json = "{\"abi\":" + std::to_string(PE_ABI_VERSION) + ",\"engine\":\"" +
                       kEngineVersion + "\",\"opencv\":";
#if PE_HAS_OPENCV
    json += "\"" CV_VERSION "\"";
#else
    json += "null";
#endif
    json += ",\"features\":[";
    for (size_t i = 0; i < engine.features.size(); ++i) {
        const auto& feature = *engine.features[i];
        if (i > 0) json += ",";
        json += "{\"id\":\"" + pe::json_escape(feature.id()) + "\",\"version\":\"" +
                pe::json_escape(feature.version()) + "\",\"kind\":\"" +
                pe::json_escape(feature.kind()) + "\",\"params\":[";
        const auto params = feature.param_names();
        for (size_t p = 0; p < params.size(); ++p) {
            if (p > 0) json += ",";
            json += "\"" + pe::json_escape(params[p]) + "\"";
        }
        json += "],\"models\":[";
        const auto models = feature.models();
        for (size_t m = 0; m < models.size(); ++m) {
            if (m > 0) json += ",";
            json += "{\"file\":\"" + pe::json_escape(models[m].file) + "\",\"sha256\":\"" +
                    pe::json_escape(models[m].sha256) + "\"}";
        }
        json += "]}";
    }
    json += "]}";
    return json;
}

void put_u32(std::vector<uint8_t>& out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

class Reader {
public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    const uint8_t* take(size_t count) {
        require(count <= size_ - offset_, PE_ERR_INVALID_ARGUMENT, "analysis blob is truncated");
        const uint8_t* start = data_ + offset_;
        offset_ += count;
        return start;
    }
    uint32_t u32() {
        const uint8_t* bytes = take(4);
        return static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 |
               static_cast<uint32_t>(bytes[2]) << 16 | static_cast<uint32_t>(bytes[3]) << 24;
    }
    bool done() const { return offset_ == size_; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t offset_ = 0;
};

void validate_region(const pe::Region& region, uint32_t width, uint32_t height) {
    const bool finite_bounds = std::isfinite(region.x0) && std::isfinite(region.y0) &&
                               std::isfinite(region.x1) && std::isfinite(region.y1);
    require(finite_bounds && region.x0 < region.x1 && region.y0 < region.y1 &&
                region.x0 >= 0.0f && region.y0 >= 0.0f && region.x1 <= width &&
                region.y1 <= height,
            PE_ERR_INTERNAL, "feature produced a region outside the image");
    require(region.cells_x >= 1 && region.cells_y >= 1 && region.cells_x <= kMaxCells &&
                region.cells_y <= kMaxCells,
            PE_ERR_INTERNAL, "feature produced an invalid region grid");
    const size_t expected = static_cast<size_t>(region.cells_x + 1) * (region.cells_y + 1) * 2;
    require(region.source_xy.size() == expected, PE_ERR_INTERNAL,
            "feature produced a region with a wrong node count");
    for (float value : region.source_xy) {
        require(std::isfinite(value), PE_ERR_INTERNAL,
                "feature produced a non-finite source coordinate");
    }
}

bool aabb_overlap(const pe::Region& a, const pe::Region& b) {
    return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
}

void sample_region_source(const pe::Region& region, float x, float y, float& out_x, float& out_y) {
    if (!(x >= region.x0 && x <= region.x1 && y >= region.y0 && y <= region.y1)) {
        out_x = x;
        out_y = y;
        return;
    }
    const float u = (region.x1 > region.x0) ? (x - region.x0) / (region.x1 - region.x0) : 0.0f;
    const float v = (region.y1 > region.y0) ? (y - region.y0) / (region.y1 - region.y0) : 0.0f;
    const float fx = std::clamp(u * region.cells_x, 0.0f, static_cast<float>(region.cells_x));
    const float fy = std::clamp(v * region.cells_y, 0.0f, static_cast<float>(region.cells_y));
    const uint32_t x0 = static_cast<uint32_t>(std::floor(fx));
    const uint32_t y0 = static_cast<uint32_t>(std::floor(fy));
    const uint32_t x1 = std::min(x0 + 1, region.cells_x);
    const uint32_t y1 = std::min(y0 + 1, region.cells_y);
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const size_t stride = static_cast<size_t>(region.cells_x) + 1;
    auto node = [&](uint32_t ix, uint32_t iy) -> std::pair<float, float> {
        const size_t index = (static_cast<size_t>(iy) * stride + ix) * 2;
        return {region.source_xy[index], region.source_xy[index + 1]};
    };
    const auto n00 = node(x0, y0);
    const auto n10 = node(x1, y0);
    const auto n01 = node(x0, y1);
    const auto n11 = node(x1, y1);
    const float top_x = n00.first * (1.0f - tx) + n10.first * tx;
    const float top_y = n00.second * (1.0f - tx) + n10.second * tx;
    const float bottom_x = n01.first * (1.0f - tx) + n11.first * tx;
    const float bottom_y = n01.second * (1.0f - tx) + n11.second * tx;
    out_x = top_x * (1.0f - ty) + bottom_x * ty;
    out_y = top_y * (1.0f - ty) + bottom_y * ty;
}

pe::Region compose_regions(const std::vector<size_t>& indices,
                           const std::vector<pe::Region>& regions, uint32_t width,
                           uint32_t height) {
    float min_x = regions[indices[0]].x0;
    float min_y = regions[indices[0]].y0;
    float max_x = regions[indices[0]].x1;
    float max_y = regions[indices[0]].y1;
    for (size_t index : indices) {
        min_x = std::min(min_x, regions[index].x0);
        min_y = std::min(min_y, regions[index].y0);
        max_x = std::max(max_x, regions[index].x1);
        max_y = std::max(max_y, regions[index].y1);
    }
    min_x = std::max(0.0f, min_x);
    min_y = std::max(0.0f, min_y);
    max_x = std::min(static_cast<float>(width), max_x);
    max_y = std::min(static_cast<float>(height), max_y);
    require(max_x > min_x && max_y > min_y, PE_ERR_INTERNAL, "merged region bounds are empty");
    uint32_t cells_x =
        std::clamp(static_cast<uint32_t>(std::ceil((max_x - min_x) / 10.0f)), 8u, kMaxCells);
    uint32_t cells_y =
        std::clamp(static_cast<uint32_t>(std::ceil((max_y - min_y) / 10.0f)), 8u, kMaxCells);
    if ((cells_x + 1) * (cells_y + 1) == 217) {
        cells_x = std::min(kMaxCells, cells_x + 1);
    }
    pe::Region merged;
    merged.x0 = min_x;
    merged.y0 = min_y;
    merged.x1 = max_x;
    merged.y1 = max_y;
    merged.cells_x = cells_x;
    merged.cells_y = cells_y;
    merged.source_xy.reserve(static_cast<size_t>(cells_x + 1) * (cells_y + 1) * 2);
    for (uint32_t j = 0; j <= cells_y; ++j) {
        for (uint32_t i = 0; i <= cells_x; ++i) {
            float x = min_x + (max_x - min_x) * i / cells_x;
            float y = min_y + (max_y - min_y) * j / cells_y;
            // 按功能求解顺序合成：F_n ∘ … ∘ F_0。
            for (size_t index : indices) {
                float next_x = x;
                float next_y = y;
                sample_region_source(regions[index], x, y, next_x, next_y);
                x = next_x;
                y = next_y;
            }
            merged.source_xy.push_back(x);
            merged.source_xy.push_back(y);
        }
    }
    return merged;
}

void merge_overlapping_regions(std::vector<pe::Region>& regions, uint32_t width, uint32_t height) {
    if (regions.size() <= 1) {
        return;
    }
    std::vector<int> parent(regions.size());
    for (size_t i = 0; i < parent.size(); ++i) {
        parent[i] = static_cast<int>(i);
    }
    auto find = [&](int value) {
        while (parent[static_cast<size_t>(value)] != value) {
            parent[static_cast<size_t>(value)] = parent[static_cast<size_t>(parent[static_cast<size_t>(value)])];
            value = parent[static_cast<size_t>(value)];
        }
        return value;
    };
    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) {
            parent[static_cast<size_t>(b)] = a;
        }
    };
    for (size_t i = 0; i < regions.size(); ++i) {
        for (size_t j = i + 1; j < regions.size(); ++j) {
            if (aabb_overlap(regions[i], regions[j])) {
                unite(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }
    std::vector<std::vector<size_t>> groups(regions.size());
    for (size_t i = 0; i < regions.size(); ++i) {
        groups[static_cast<size_t>(find(static_cast<int>(i)))].push_back(i);
    }
    std::vector<pe::Region> merged;
    merged.reserve(regions.size());
    for (size_t root = 0; root < groups.size(); ++root) {
        auto& group = groups[root];
        if (group.empty()) {
            continue;
        }
        std::sort(group.begin(), group.end());
        if (group.size() == 1) {
            merged.push_back(std::move(regions[group[0]]));
        } else {
            pe::Region composed = compose_regions(group, regions, width, height);
            validate_region(composed, width, height);
            merged.push_back(std::move(composed));
        }
    }
    regions = std::move(merged);
}

}  // namespace

std::string pe::json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(c >> 4) & 0xf];
                    out += hex[c & 0xf];
                } else {
                    out += c;
                }
        }
    }
    return out;
}

extern "C" {

PE_API uint32_t pe_abi_version(void) { return PE_ABI_VERSION; }

PE_API const char* pe_last_error(void) { return g_last_error.c_str(); }

PE_API PeStatus pe_engine_create(const PeEngineDesc* desc, PeEngine** out_engine) {
    return guarded([&] {
        require(out_engine != nullptr, PE_ERR_INVALID_ARGUMENT, "out_engine is null");
        *out_engine = nullptr;
        require(desc != nullptr && desc->struct_size >= sizeof(PeEngineDesc),
                PE_ERR_INVALID_ARGUMENT, "engine descriptor is missing or too small");
        auto engine = std::make_unique<PeEngine>();
        engine->config.threads = desc->threads;
        engine->config.resource_dir = desc->resource_dir ? desc->resource_dir : "";
        engine->config.onnxruntime_path = desc->onnxruntime_path ? desc->onnxruntime_path : "";
        engine->features = pe::create_features(engine->config);
        engine->capabilities = capabilities_json(*engine);
        *out_engine = engine.release();
        return PE_OK;
    });
}

PE_API void pe_engine_destroy(PeEngine* engine) { delete engine; }

PE_API const char* pe_engine_capabilities(const PeEngine* engine) {
    return engine ? engine->capabilities.c_str() : "";
}

PE_API PeStatus pe_analyze(PeEngine* engine, const char* feature_id, const PeImage* image,
                           PeCancelFn cancel, void* cancel_user, PeAnalysis** out_analysis) {
    return guarded([&] {
        require(out_analysis != nullptr, PE_ERR_INVALID_ARGUMENT, "out_analysis is null");
        *out_analysis = nullptr;
        require(engine != nullptr, PE_ERR_INVALID_ARGUMENT, "engine is null");
        require_id(feature_id, "feature_id");
        const pe::ImageView view = checked_image(image);
        std::lock_guard<std::mutex> lock(engine->mutex);
        pe::Feature* feature = engine->find(feature_id);
        require(feature != nullptr, PE_ERR_UNSUPPORTED,
                std::string("unknown feature: ") + feature_id);
        const pe::CancelToken token(cancel, cancel_user);
        auto analysis = std::make_unique<PeAnalysis>();
        analysis->owner = engine;
        analysis->feature = feature;
        analysis->data = feature->analyze(view, token);
        require(analysis->data != nullptr && analysis->data->width == view.width &&
                    analysis->data->height == view.height,
                PE_ERR_INTERNAL, "feature returned an analysis for a different image");
        analysis->diagnostics = analysis->data->diagnostics_json();
        *out_analysis = analysis.release();
        return PE_OK;
    });
}

PE_API const char* pe_analysis_feature(const PeAnalysis* analysis) {
    return analysis ? analysis->feature->id() : "";
}

PE_API const char* pe_analysis_diagnostics(const PeAnalysis* analysis) {
    return analysis ? analysis->diagnostics.c_str() : "";
}

PE_API PeStatus pe_analysis_serialize(const PeAnalysis* analysis, uint8_t** out_data,
                                      size_t* out_size) {
    return guarded([&] {
        require(out_data != nullptr && out_size != nullptr, PE_ERR_INVALID_ARGUMENT,
                "output pointers are null");
        *out_data = nullptr;
        *out_size = 0;
        require(analysis != nullptr, PE_ERR_INVALID_ARGUMENT, "analysis is null");
        const std::string id = analysis->feature->id();
        const std::vector<uint8_t> payload = analysis->data->serialize_payload();
        require(payload.size() <= UINT32_MAX, PE_ERR_INTERNAL, "analysis payload is too large");
        std::vector<uint8_t> blob(kMagic, kMagic + 4);
        put_u32(blob, PE_ABI_VERSION);
        put_u32(blob, static_cast<uint32_t>(id.size()));
        blob.insert(blob.end(), id.begin(), id.end());
        put_u32(blob, analysis->data->width);
        put_u32(blob, analysis->data->height);
        put_u32(blob, static_cast<uint32_t>(payload.size()));
        blob.insert(blob.end(), payload.begin(), payload.end());
        auto* data = new uint8_t[blob.size()];
        std::memcpy(data, blob.data(), blob.size());
        *out_data = data;
        *out_size = blob.size();
        return PE_OK;
    });
}

PE_API PeStatus pe_analysis_deserialize(PeEngine* engine, const uint8_t* data, size_t size,
                                        PeAnalysis** out_analysis) {
    return guarded([&] {
        require(out_analysis != nullptr, PE_ERR_INVALID_ARGUMENT, "out_analysis is null");
        *out_analysis = nullptr;
        require(engine != nullptr && data != nullptr, PE_ERR_INVALID_ARGUMENT,
                "engine or data is null");
        Reader reader(data, size);
        require(std::memcmp(reader.take(4), kMagic, 4) == 0, PE_ERR_INVALID_ARGUMENT,
                "not a portrait engine analysis blob");
        require(reader.u32() == PE_ABI_VERSION, PE_ERR_UNSUPPORTED,
                "analysis blob was written by another ABI version");
        const uint32_t id_length = reader.u32();
        require(id_length > 0 && id_length <= kMaxIdLength, PE_ERR_INVALID_ARGUMENT,
                "analysis blob has an invalid feature id");
        const auto* id_bytes = reinterpret_cast<const char*>(reader.take(id_length));
        const std::string id(id_bytes, id_length);
        const uint32_t width = reader.u32();
        const uint32_t height = reader.u32();
        require(width > 0 && height > 0 && width <= kMaxDimension && height <= kMaxDimension,
                PE_ERR_INVALID_ARGUMENT, "analysis blob has invalid dimensions");
        const uint32_t payload_size = reader.u32();
        const uint8_t* payload = reader.take(payload_size);
        require(reader.done(), PE_ERR_INVALID_ARGUMENT, "analysis blob has trailing bytes");
        std::lock_guard<std::mutex> lock(engine->mutex);
        const pe::Feature* feature = engine->find(id.c_str());
        require(feature != nullptr, PE_ERR_UNSUPPORTED, "unknown feature: " + id);
        auto analysis = std::make_unique<PeAnalysis>();
        analysis->owner = engine;
        analysis->feature = feature;
        analysis->data = feature->deserialize(width, height, payload, payload_size);
        require(analysis->data != nullptr && analysis->data->width == width &&
                    analysis->data->height == height,
                PE_ERR_INTERNAL, "feature restored an analysis with different dimensions");
        analysis->diagnostics = analysis->data->diagnostics_json();
        *out_analysis = analysis.release();
        return PE_OK;
    });
}

PE_API void pe_analysis_release(PeAnalysis* analysis) { delete analysis; }

PE_API void pe_buffer_free(uint8_t* data) { delete[] data; }

PE_API PeStatus pe_solve_warp(PeEngine* engine, const PeAnalysis* const* analyses,
                              size_t analysis_count, const PeParam* params, size_t param_count,
                              PeCancelFn cancel, void* cancel_user, PeWarpResult** out_result) {
    return guarded([&] {
        require(out_result != nullptr, PE_ERR_INVALID_ARGUMENT, "out_result is null");
        *out_result = nullptr;
        require(engine != nullptr, PE_ERR_INVALID_ARGUMENT, "engine is null");
        require(analysis_count == 0 || analyses != nullptr, PE_ERR_INVALID_ARGUMENT,
                "analyses is null");
        require(param_count == 0 || params != nullptr, PE_ERR_INVALID_ARGUMENT, "params is null");
        std::lock_guard<std::mutex> lock(engine->mutex);

        uint32_t width = 0;
        uint32_t height = 0;
        for (size_t i = 0; i < analysis_count; ++i) {
            const PeAnalysis* analysis = analyses[i];
            require(analysis != nullptr && analysis->owner == engine, PE_ERR_INVALID_ARGUMENT,
                    "analysis is null or belongs to another engine");
            if (i == 0) {
                width = analysis->data->width;
                height = analysis->data->height;
            }
            require(analysis->data->width == width && analysis->data->height == height,
                    PE_ERR_INVALID_ARGUMENT, "analyses come from images of different sizes");
            for (size_t j = 0; j < i; ++j) {
                require(analyses[j]->feature != analysis->feature, PE_ERR_INVALID_ARGUMENT,
                        std::string("feature solved twice: ") + analysis->feature->id());
            }
        }

        std::vector<pe::ParamView> views(analysis_count);
        for (size_t p = 0; p < param_count; ++p) {
            const PeParam& param = params[p];
            require_id(param.feature_id, "param.feature_id");
            require_id(param.name, "param.name");
            require(std::isfinite(param.value), PE_ERR_INVALID_ARGUMENT,
                    std::string("non-finite value for ") + param.feature_id + "." + param.name);
            size_t target = analysis_count;
            for (size_t i = 0; i < analysis_count; ++i) {
                if (std::strcmp(analyses[i]->feature->id(), param.feature_id) == 0) {
                    target = i;
                }
            }
            require(target < analysis_count, PE_ERR_INVALID_ARGUMENT,
                    std::string("parameter for a feature that is not being solved: ") +
                        param.feature_id);
            bool known = false;
            for (const auto& name : analyses[target]->feature->param_names()) {
                known = known || name == param.name;
            }
            require(known, PE_ERR_INVALID_ARGUMENT,
                    std::string("unknown parameter ") + param.feature_id + "." + param.name);
            for (size_t q = 0; q < p; ++q) {
                require(std::strcmp(params[q].feature_id, param.feature_id) != 0 ||
                            std::strcmp(params[q].name, param.name) != 0,
                        PE_ERR_INVALID_ARGUMENT,
                        std::string("duplicate parameter ") + param.feature_id + "." + param.name);
            }
            views[target].add(param.name, param.value);
        }

        const pe::CancelToken token(cancel, cancel_user);
        auto result = std::make_unique<PeWarpResult>();
        std::string diagnostics = "{\"features\":[";
        for (size_t i = 0; i < analysis_count; ++i) {
            token.check();
            const pe::Feature& feature = *analyses[i]->feature;
            const size_t before = result->regions.size();
            const pe::FeatureReport report =
                feature.solve(*analyses[i]->data, views[i], token, result->regions);
            for (size_t r = before; r < result->regions.size(); ++r) {
                validate_region(result->regions[r], width, height);
            }
            if (i > 0) diagnostics += ",";
            diagnostics += "{\"id\":\"" + pe::json_escape(feature.id()) + "\",\"status\":\"" +
                           pe::json_escape(report.status) + "\"";
            if (!report.reason.empty()) {
                diagnostics += ",\"reason\":\"" + pe::json_escape(report.reason) + "\"";
            }
            diagnostics += "}";
        }
        diagnostics += "]}";
        merge_overlapping_regions(result->regions, width, height);
        require(result->regions.size() <= kMaxRegions, PE_ERR_UNSUPPORTED,
                "too many warp regions in one solve");
        result->diagnostics = std::move(diagnostics);
        *out_result = result.release();
        return PE_OK;
    });
}

PE_API size_t pe_warp_region_count(const PeWarpResult* result) {
    return result ? result->regions.size() : 0;
}

PE_API PeStatus pe_warp_region(const PeWarpResult* result, size_t index,
                               PeWarpRegion* out_region) {
    return guarded([&] {
        require(result != nullptr && out_region != nullptr &&
                    out_region->struct_size >= sizeof(PeWarpRegion),
                PE_ERR_INVALID_ARGUMENT, "result or region descriptor is invalid");
        require(index < result->regions.size(), PE_ERR_INVALID_ARGUMENT,
                "region index out of range");
        const pe::Region& region = result->regions[index];
        out_region->x0 = region.x0;
        out_region->y0 = region.y0;
        out_region->x1 = region.x1;
        out_region->y1 = region.y1;
        out_region->cells_x = region.cells_x;
        out_region->cells_y = region.cells_y;
        out_region->source_xy = region.source_xy.data();
        return PE_OK;
    });
}

PE_API const char* pe_warp_diagnostics(const PeWarpResult* result) {
    return result ? result->diagnostics.c_str() : "";
}

PE_API void pe_warp_result_release(PeWarpResult* result) { delete result; }

}  // extern "C"
