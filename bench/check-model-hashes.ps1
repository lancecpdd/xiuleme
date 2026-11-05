$dir = Join-Path $PSScriptRoot "..\src-tauri\resources"
$expected = @{
    "sam_vit_b_01ec64_encoder.onnx" = "16ab73d9c824886f0de2938c19df22fb9ec3deebfd0de58e65177e479213d7d1"
    "sam_vit_b_01ec64_decoder.onnx" = "85d0d672cf5b7fe763edcde429e5533e62f674af4b15c7d688b7673b0ef00bf7"
    "u2net.onnx"                    = "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"
    "skyseg_u2net.onnx"             = "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39"
    "depth_anything_v2_vits.onnx"   = "d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be"
    "clip_model.onnx"               = "57879bb1c23cdeb350d23569dd251ed4b740a96d747c529e94a2bb8040ac5d00"
    "nind_denoise_utnet_684.onnx"   = "ee3586279d514df557ff3f7dec6df37fafc51ba5d3a3435b2cc9ac2d9017e7fe"
    "lama_fp16.onnx"                = "2d6be6277c400d6f1b91819737f7c3da935e5c63d1b521d393be1196a2bfa82c"
}
$allOk = $true
foreach ($name in $expected.Keys) {
    $path = Join-Path $dir $name
    if (-not (Test-Path $path)) {
        Write-Host "[MISSING] $name"
        $allOk = $false
        continue
    }
    $actual = (Get-FileHash -Path $path -Algorithm SHA256).Hash.ToLower()
    $match = $actual -eq $expected[$name]
    if (-not $match) { $allOk = $false }
    Write-Host ("[{0}] {1}" -f ($(if ($match) { "OK" } else { "FAIL" }), $name))
    if (-not $match) {
        Write-Host "   actual  : $actual"
        Write-Host "   expected: $($expected[$name])"
    }
}
Write-Host ("clip_tokenizer.json: no hash check required (exists={0})" -f (Test-Path (Join-Path $dir "clip_tokenizer.json")))
if ($allOk) { Write-Host "ALL HASHES OK" } else { Write-Host "SOME HASHES FAILED" }
