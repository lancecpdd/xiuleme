#include "ort_runtime.h"

#include "feature.h"

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#undef ORT_API_MANUAL_INIT

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstring>

namespace pe {
namespace {

using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)(void);

void* load_library(const std::string& utf8_path) {
#if defined(_WIN32)
    const int wide_count = MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, nullptr, 0);
    if (wide_count <= 0) {
        throw Error(PE_ERR_MODEL, "invalid ONNX Runtime path encoding");
    }
    std::wstring wide(static_cast<size_t>(wide_count - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, wide.data(), wide_count);
    HMODULE module = LoadLibraryW(wide.c_str());
    if (module == nullptr) {
        throw Error(PE_ERR_MODEL, "failed to load ONNX Runtime: " + utf8_path);
    }
    return module;
#else
    void* module = dlopen(utf8_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module == nullptr) {
        throw Error(PE_ERR_MODEL, std::string("failed to load ONNX Runtime: ") +
                                      (dlerror() ? dlerror() : utf8_path.c_str()));
    }
    return module;
#endif
}

void* library_symbol(void* library, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

const OrtApi* as_api(const void* api) { return static_cast<const OrtApi*>(api); }

void check_ort(const OrtApi* api, OrtStatus* status, const char* what) {
    if (status == nullptr) {
        return;
    }
    const char* message = api->GetErrorMessage(status);
    std::string text = std::string(what) + ": " + (message ? message : "unknown ORT error");
    api->ReleaseStatus(status);
    throw Error(PE_ERR_MODEL, text);
}

}  // namespace

OrtRuntime& OrtRuntime::instance() {
    static OrtRuntime runtime;
    return runtime;
}

bool OrtRuntime::loaded() const { return api_ != nullptr && env_ != nullptr; }

void OrtRuntime::ensure_loaded(const std::string& utf8_library_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (loaded()) {
        if (!loaded_path_.empty() && loaded_path_ != utf8_library_path) {
            throw Error(PE_ERR_MODEL, "ONNX Runtime already loaded from a different path");
        }
        return;
    }
    if (utf8_library_path.empty()) {
        throw Error(PE_ERR_MODEL, "ONNX Runtime path was not provided");
    }
    library_ = load_library(utf8_library_path);
    auto get_api_base = reinterpret_cast<GetApiBaseFn>(library_symbol(library_, "OrtGetApiBase"));
    if (get_api_base == nullptr) {
        throw Error(PE_ERR_MODEL, "OrtGetApiBase export missing");
    }
    const OrtApiBase* base = get_api_base();
    if (base == nullptr || base->GetApi == nullptr) {
        throw Error(PE_ERR_MODEL, "OrtGetApiBase returned null");
    }
    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
        throw Error(PE_ERR_MODEL, "ONNX Runtime API version mismatch");
    }
    api_ = api;
    OrtEnv* env = nullptr;
    check_ort(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "portrait_engine", &env),
              "CreateEnv");
    env_ = env;
    loaded_path_ = utf8_library_path;
}

OrtRuntime::Session::Session(OrtRuntime& runtime, const std::string& utf8_model_path, int threads)
    : runtime_(&runtime) {
    const OrtApi* api = as_api(runtime_->api_);
    OrtSessionOptions* options = nullptr;
    check_ort(api, api->CreateSessionOptions(&options), "CreateSessionOptions");
    const int intra = threads > 0 ? threads : 2;
    check_ort(api, api->SetIntraOpNumThreads(options, intra), "SetIntraOpNumThreads");
    check_ort(api, api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL),
              "SetSessionGraphOptimizationLevel");

    OrtSession* session = nullptr;
#if defined(_WIN32)
    const int wide_count = MultiByteToWideChar(CP_UTF8, 0, utf8_model_path.c_str(), -1, nullptr, 0);
    std::wstring wide(static_cast<size_t>(wide_count - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8_model_path.c_str(), -1, wide.data(), wide_count);
    OrtStatus* status =
        api->CreateSession(static_cast<OrtEnv*>(runtime_->env_), wide.c_str(), options, &session);
#else
    OrtStatus* status = api->CreateSession(static_cast<OrtEnv*>(runtime_->env_),
                                           utf8_model_path.c_str(), options, &session);
#endif
    api->ReleaseSessionOptions(options);
    check_ort(api, status, "CreateSession");
    session_ = session;

    OrtMemoryInfo* memory_info = nullptr;
    check_ort(api,
              api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info),
              "CreateCpuMemoryInfo");
    memory_info_ = memory_info;
}

OrtRuntime::Session::~Session() {
    if (runtime_ == nullptr || runtime_->api_ == nullptr) {
        return;
    }
    const OrtApi* api = as_api(runtime_->api_);
    if (session_ != nullptr) {
        api->ReleaseSession(static_cast<OrtSession*>(session_));
    }
    if (memory_info_ != nullptr) {
        api->ReleaseMemoryInfo(static_cast<OrtMemoryInfo*>(memory_info_));
    }
}

std::vector<float> OrtRuntime::Session::run(const char* input_name,
                                            const std::vector<int64_t>& input_shape,
                                            const float* input, size_t input_count,
                                            const std::vector<const char*>& output_names,
                                            std::vector<std::vector<int64_t>>& output_shapes) const {
    const OrtApi* api = as_api(runtime_->api_);
    OrtValue* input_tensor = nullptr;
    check_ort(api,
              api->CreateTensorWithDataAsOrtValue(static_cast<OrtMemoryInfo*>(memory_info_),
                                                  const_cast<float*>(input),
                                                  input_count * sizeof(float), input_shape.data(),
                                                  input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                  &input_tensor),
              "CreateTensorWithDataAsOrtValue");

    std::vector<OrtValue*> outputs(output_names.size(), nullptr);
    const char* input_names[] = {input_name};
    OrtStatus* status = api->Run(static_cast<OrtSession*>(session_), nullptr, input_names,
                                 const_cast<const OrtValue**>(&input_tensor), 1, output_names.data(),
                                 output_names.size(), outputs.data());
    api->ReleaseValue(input_tensor);
    check_ort(api, status, "Session::Run");

    std::vector<float> flat;
    output_shapes.assign(outputs.size(), {});
    for (size_t i = 0; i < outputs.size(); ++i) {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        check_ort(api, api->GetTensorTypeAndShape(outputs[i], &info), "GetTensorTypeAndShape");
        size_t dim_count = 0;
        check_ort(api, api->GetDimensionsCount(info, &dim_count), "GetDimensionsCount");
        output_shapes[i].assign(dim_count, 0);
        check_ort(api, api->GetDimensions(info, output_shapes[i].data(), dim_count),
                  "GetDimensions");
        size_t element_count = 0;
        check_ort(api, api->GetTensorShapeElementCount(info, &element_count),
                  "GetTensorShapeElementCount");
        api->ReleaseTensorTypeAndShapeInfo(info);

        float* data = nullptr;
        check_ort(api, api->GetTensorMutableData(outputs[i], reinterpret_cast<void**>(&data)),
                  "GetTensorMutableData");
        flat.insert(flat.end(), data, data + element_count);
        api->ReleaseValue(outputs[i]);
    }
    return flat;
}

}  // namespace pe
