GPUPixel runtime bundle (libgpupixel_rr + deps + VNN + lookup PNGs).

Build and stage with:
  cd tools/gpupixel-ref
  ./scripts/bootstrap.sh
  ./scripts/build-macos.sh   # macOS (packages into this folder)

Windows:
  .\scripts\bootstrap.ps1
  .\scripts\build.ps1
  .\scripts\package-rr.ps1

Android arm64: scripts/build-android.sh (EGL Session required)
