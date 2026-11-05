# Build portrait_engine with MSVC + Ninja, run the C ABI self test, and stage the
# library into src-tauri/resources/portrait-engine.
# OpenCV comes from vcpkg: vcpkg install opencv4[core]:x64-windows-static-md
param(
    [string]$Config = "Release",
    [string]$Triplet = "x64-windows-static-md",
    [switch]$NoOpenCV,
    [switch]$NoStage
)
$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$repo = Resolve-Path (Join-Path $root "..\..")
$build = Join-Path $root "build\$Config"
$stage = Join-Path $repo "src-tauri\resources\portrait-engine"

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio / Build Tools with the C++ toolset was not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"

$ortInclude = Join-Path $root "third_party\onnxruntime\include\onnxruntime_c_api.h"
if (-not $NoOpenCV -and -not (Test-Path $ortInclude)) {
    throw "ONNX Runtime headers missing at native/portrait-engine/third_party/onnxruntime/include. Extract the include/ folder from the official onnxruntime-win-x64-1.22.0.zip release."
}

$cmakeArgs = @("-S", "`"$root`"", "-B", "`"$build`"", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=$Config")
if ($NoOpenCV) {
    $cmakeArgs += "-DPE_WITH_OPENCV=OFF"
} else {
    $vcpkgRoot = $env:VCPKG_ROOT
    if (-not $vcpkgRoot) {
        $vcpkgExe = Get-Command vcpkg -ErrorAction SilentlyContinue
        if ($vcpkgExe) { $vcpkgRoot = Split-Path $vcpkgExe.Source }
    }
    if (-not $vcpkgRoot) { throw "vcpkg not found; set VCPKG_ROOT, or pass -NoOpenCV to build the skeleton only" }
    $toolchain = Join-Path $vcpkgRoot "scripts\buildsystems\vcpkg.cmake"
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=`"$toolchain`"", "-DVCPKG_TARGET_TRIPLET=$Triplet"
}
if (-not $NoStage) { $cmakeArgs += "-DPE_STAGE_DIR=`"$stage`"" }

$steps = @(
    "`"$vcvars`" >nul",
    "cmake $($cmakeArgs -join ' ')",
    "cmake --build `"$build`"",
    "ctest --test-dir `"$build`" --output-on-failure"
)
cmd /c ($steps -join " && ")
if ($LASTEXITCODE -ne 0) { throw "portrait_engine build or self test failed (exit $LASTEXITCODE)" }
if (-not $NoStage) { Write-Host "Staged into $stage" }
