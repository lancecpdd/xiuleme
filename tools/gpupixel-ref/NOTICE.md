# NOTICE — GPUPixel product runtime toolchain

This directory builds and stages the **GPUPixel** runtime used by XiuLeMe’s
portrait beauty path (`engineVersion=gpupixel-runtime-v1`).

## Third-party components

### GPUPixel (pixpark/gpupixel)

- License: Apache License 2.0
- Source: https://github.com/pixpark/gpupixel
- Pinned tag: **v1.2.x.8** (API: `SourceRawDataInput` / `TargetRawDataOutput`; includes `third_party/vnn`).

### gpupixel_pywrapper (quqixun)

- License: see [vendor/pywrapper/LICENSE](vendor/pywrapper/LICENSE)
- Source: https://github.com/quqixun/gpupixel_pywrapper
- Used as reference for the filter graph; product C ABI is `rr/` (`libgpupixel_rr`).

## XiuLeMe product path

XiuLeMe loads `libgpupixel_rr` at runtime (Windows / macOS resources; Android
jniLibs + assets). See `src-tauri/resources/portrait-NOTICE.txt` and
`src-tauri/src/portrait_gpupixel.rs`.
