# GPUPixel 产品运行时（Windows / macOS / Android）

构建 **GPUPixel 1.2.x（默认 `v1.2.x.8`）** 与产品 C ABI `libgpupixel_rr`，打包进
XiuLeMe Tauri（`engineVersion=gpupixel-runtime-v1`）：瘦脸、大眼、口红、腮红、提亮眼白、祛红血丝（经典磨皮／美白已下线，FFI 仍传 0）。
眼白在 `libgpupixel_rr` 内用 VNN 106 关键点对 RGBA 做 CPU 后处理；`gpupixel_rr_set_parameters` 现为 8 参（末两项 `eye_brighten` / `eye_redness`，0–10）。

Python 对照工具仍可用（`python/process_image.py`）；与产品共用同一上游滤镜链。

## 前置依赖

| 工具 | 说明 |
|------|------|
| CMake | ≥ 3.10 |
| MinGW-W64（Windows） | `MinGW_HOME` + PATH |
| Xcode CLT（macOS） | OpenGL + GLFW |
| Android NDK | arm64-v8a；EGL Session 见 `rr/src/gpupixel_rr_android.cc.stub` |
| Python 3.9+（可选对照） | `pip install -r python/requirements.txt` |

## 构建（Windows）

```powershell
cd tools\gpupixel-ref
.\scripts\bootstrap.ps1
.\scripts\build.ps1
.\scripts\package-rr.ps1   # → src-tauri/resources/gpupixel/
```

成功后目录含：`libgpupixel_rr.dll`、`libgpupixel.dll`、`glfw3.dll`、`vnn_*.dll`、
`face_pc[1.0.0].vnnmodel`、`lookup_*.png` 等。

修改 `rr/src` 或 `patches/target_raw_data_output.cc` 后必须重新执行
`bootstrap.ps1`（已有 vendor 时无需 `-Force`，用于重新覆盖补丁）、`build.ps1` 与
`package-rr.ps1`。PBO 补丁按当前帧同步读回，并修正奇数尺寸 I420 短分配；若只替换
Rust 代码而继续使用旧 DLL，瘦脸等首次触发 GPUPixel 的预览仍可能显示黑帧。Rust 侧
会对奇数宽高先裁成偶数再 bake，可在未重编 DLL 时避免 `STATUS_HEAP_CORRUPTION`。

## macOS

依赖：Xcode CLT、CMake（`brew install cmake`）。

```bash
cd tools/gpupixel-ref
./scripts/bootstrap.sh      # 拉取 v1.2.x.8 + 打补丁
./scripts/build-macos.sh    # 编译并打包 → src-tauri/resources/gpupixel/
```

成功后目录含：`libgpupixel_rr.dylib`、`gpupixel.framework`、
`vnn_*_osx.framework`、`face_pc[1.0.0].vnnmodel`、`lookup_*.png` 等。

然后重新启动 `tauri dev` / `cargo build`，使资源同步到
`src-tauri/target/*/resources/gpupixel/`。

## Android

```bash
./scripts/build-android.sh   # 需完成 EGL Session 后再链 so
```

## 产品接入

- Rust：`src-tauri/src/portrait_gpupixel.rs`（`libloading`）
- 渲染：`portrait_render` 在 `native_rgba` 前整图 bake；旁路 UV 场与自研美肤场
- UI：`beauty` / `white` / `slimFace` / `bigEye` / `lipstick` / `blusher` / `eyeWhiteBrighten` / `eyeRedness`

## 许可

见 [NOTICE.md](NOTICE.md)。上游 Apache-2.0；再分发保留 NOTICE／版权声明。
