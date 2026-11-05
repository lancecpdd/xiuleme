$dir = Join-Path $PSScriptRoot "src\i18n\locales"
$targets = @("ca.json","cs.json","de.json","es.json","fr.json","it.json","ja.json","ko.json","nl.json","pl.json","pt.json","ru.json")
$old = @'
  "auth": {
    "accountPlaceholder": "Account",
    "confirming": "Confirming login…",
    "errorRequired": "Please enter both account and password.",
    "login": "Sign in",
    "logout": "Sign out",
    "passwordPlaceholder": "Password",
    "qrExpired": "QR code expired. Refresh to try again.",
    "qrHint": "Open the WeChat mini program and scan this code to confirm login.",
    "refreshQr": "Refresh",
    "simulateConfirm": "Simulate mini program confirmation",
    "tabAccount": "Account",
    "tabQr": "Scan to sign in",
    "title": "Sign in to continue",
    "waitingScan": "Waiting for scan confirmation",
    "wechatUser": "WeChat user"
  },
'@
$new = @'
  "auth": {
    "accountPlaceholder": "Account",
    "errorRequired": "Please enter both account and password.",
    "login": "Sign in",
    "logout": "Sign out",
    "passwordPlaceholder": "Password",
    "qrCreateFailed": "Failed to load QR code. Tap to retry.",
    "qrExpired": "QR code expired. Refresh to try again.",
    "qrHint": "Open the Paileme mini program on your phone and scan this code to confirm login.",
    "qrLoading": "Loading QR code…",
    "refreshQr": "Refresh",
    "retry": "Retry",
    "tabAccount": "Account",
    "tabQr": "Scan to sign in",
    "title": "Sign in to continue",
    "waitingScan": "Waiting for scan confirmation"
  },
'@
foreach ($t in $targets) {
  $path = Join-Path $dir $t
  $content = [System.IO.File]::ReadAllText($path)
  if ($content.Contains($old)) {
    $content = $content.Replace($old, $new)
    [System.IO.File]::WriteAllText($path, $content, (New-Object System.Text.UTF8Encoding($false)))
    Write-Output "UPDATED: $t"
  } else {
    Write-Output "NOT FOUND: $t"
  }
}
