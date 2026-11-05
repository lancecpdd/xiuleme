/*
 * XiuLeMe 人像处理引擎 C ABI。
 *
 * 引擎只做分析（模型推理＋前后处理）与形变求解，输出逆向采样网格；
 * 像素渲染、缓存、参数哈希与持久化由宿主（Rust）负责。
 * 设计说明见 docs/portrait-reshape/17-native-portrait-engine.md。
 *
 * 约定：
 * - 所有字符串为 UTF-8，以 NUL 结尾。
 * - 返回 PeStatus 的函数失败时，可在同一线程调用 pe_last_error() 取原因。
 * - 引擎实例内部串行执行；同一实例可被多个线程调用，但不会并行处理。
 * - 由引擎分配的对象只能用对应的 release／free 函数释放。
 * - 结构体首字段 struct_size 必须由调用方填写为 sizeof，用于向后兼容地追加字段。
 */
#ifndef XIULEME_PORTRAIT_ENGINE_H
#define XIULEME_PORTRAIT_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(PORTRAIT_ENGINE_EXPORTS)
#define PE_API __declspec(dllexport)
#else
#define PE_API __declspec(dllimport)
#endif
#else
#define PE_API __attribute__((visibility("default")))
#endif

/* 不兼容的 ABI 变更必须递增；宿主加载时精确比对。 */
#define PE_ABI_VERSION 1u

typedef int32_t PeStatus;
#define PE_OK 0
#define PE_CANCELLED 1
#define PE_ERR_INVALID_ARGUMENT 10
#define PE_ERR_UNSUPPORTED 11
#define PE_ERR_MODEL 12
#define PE_ERR_OUT_OF_MEMORY 13
#define PE_ERR_INTERNAL 14

typedef uint32_t PeFormat;
#define PE_FORMAT_RGBA8 1u
#define PE_FORMAT_RGBA32F 2u

typedef struct PeEngine PeEngine;
typedef struct PeAnalysis PeAnalysis;
typedef struct PeWarpResult PeWarpResult;

typedef struct PeEngineDesc {
    uint32_t struct_size;
    /* 推理线程数；<=0 时由引擎决定。 */
    int32_t threads;
    /* 模型与资源目录；引擎内部校验模型 SHA-256。可为 NULL（仅调试功能可用）。 */
    const char* resource_dir;
    /* 与宿主共用的 ONNX Runtime 动态库路径；可为 NULL（不需要推理的功能可用）。 */
    const char* onnxruntime_path;
} PeEngineDesc;

typedef struct PeImage {
    uint32_t struct_size;
    PeFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t stride_bytes;
    const void* data;
} PeImage;

/* 一条数值参数，如 { "body.slim_arms", "amount", 60.0 }。 */
typedef struct PeParam {
    const char* feature_id;
    const char* name;
    double value;
} PeParam;

/*
 * 一块形变区域：在 [x0,x1]×[y0,y1] 上均匀铺 (cells_x+1)×(cells_y+1) 个目标节点，
 * 按行优先给出每个节点的源坐标 (x,y)，即输出像素应从输入图哪里采样。
 * 坐标为输入图像素边界坐标；source_xy 在 PeWarpResult 释放前有效。
 */
typedef struct PeWarpRegion {
    uint32_t struct_size;
    float x0;
    float y0;
    float x1;
    float y1;
    uint32_t cells_x;
    uint32_t cells_y;
    const float* source_xy;
} PeWarpRegion;

/* 返回非 0 表示请求取消。引擎在各阶段之间及长循环内轮询。 */
typedef int32_t (*PeCancelFn)(void* user);

PE_API uint32_t pe_abi_version(void);

/* 当前线程最近一次失败的原因；成功调用不清空。返回值在同线程下次调用引擎前有效。 */
PE_API const char* pe_last_error(void);

PE_API PeStatus pe_engine_create(const PeEngineDesc* desc, PeEngine** out_engine);
PE_API void pe_engine_destroy(PeEngine* engine);

/*
 * 能力清单 JSON，由引擎持有，生命周期同 engine：
 * {"abi":1,"engine":"portrait-engine-v1","opencv":"4.x|null",
 *  "features":[{"id":"...","version":"...","kind":"warp","params":["amount"],"models":[{"file":"...","sha256":"..."}]}]}
 */
PE_API const char* pe_engine_capabilities(const PeEngine* engine);

/* 阶段一：分析（耗时，与强度无关，可由宿主按图像摘要缓存）。 */
PE_API PeStatus pe_analyze(PeEngine* engine, const char* feature_id, const PeImage* image,
                           PeCancelFn cancel, void* cancel_user, PeAnalysis** out_analysis);
PE_API const char* pe_analysis_feature(const PeAnalysis* analysis);
/* 分析诊断 JSON，如 {"status":"ready"} 或 {"status":"bypass","reason":"arms_crossed"}。 */
PE_API const char* pe_analysis_diagnostics(const PeAnalysis* analysis);
PE_API PeStatus pe_analysis_serialize(const PeAnalysis* analysis, uint8_t** out_data,
                                      size_t* out_size);
PE_API PeStatus pe_analysis_deserialize(PeEngine* engine, const uint8_t* data, size_t size,
                                        PeAnalysis** out_analysis);
PE_API void pe_analysis_release(PeAnalysis* analysis);
PE_API void pe_buffer_free(uint8_t* data);

/*
 * 阶段二：求解（快）。多个分析一次求解，引擎负责合并为互不重叠的区域；
 * 区域为 0 个表示恒等。所有分析必须来自同一尺寸的输入图。
 */
PE_API PeStatus pe_solve_warp(PeEngine* engine, const PeAnalysis* const* analyses,
                              size_t analysis_count, const PeParam* params, size_t param_count,
                              PeCancelFn cancel, void* cancel_user, PeWarpResult** out_result);
PE_API size_t pe_warp_region_count(const PeWarpResult* result);
PE_API PeStatus pe_warp_region(const PeWarpResult* result, size_t index, PeWarpRegion* out_region);
/* 求解诊断 JSON：{"features":[{"id":"...","status":"applied|identity|bypass","reason":"..."}]}。 */
PE_API const char* pe_warp_diagnostics(const PeWarpResult* result);
PE_API void pe_warp_result_release(PeWarpResult* result);

#ifdef __cplusplus
}
#endif

#endif /* XIULEME_PORTRAIT_ENGINE_H */
