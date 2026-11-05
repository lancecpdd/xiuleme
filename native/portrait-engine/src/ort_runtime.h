#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace pe {

/* Dynamically loads the host's onnxruntime shared library and exposes a thin session wrapper. */
class OrtRuntime {
public:
    static OrtRuntime& instance();
    void ensure_loaded(const std::string& utf8_library_path);
    bool loaded() const;

    class Session {
    public:
        Session(OrtRuntime& runtime, const std::string& utf8_model_path, int threads);
        ~Session();
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;

        std::vector<float> run(const char* input_name, const std::vector<int64_t>& input_shape,
                               const float* input, size_t input_count,
                               const std::vector<const char*>& output_names,
                               std::vector<std::vector<int64_t>>& output_shapes) const;

    private:
        OrtRuntime* runtime_;
        void* session_ = nullptr;
        void* memory_info_ = nullptr;
    };

private:
    OrtRuntime() = default;
    void* library_ = nullptr;
    const void* api_ = nullptr;  // OrtApi*
    void* env_ = nullptr;
    std::mutex mutex_;
    std::string loaded_path_;
};

}  // namespace pe
