# Worktree Web entry point. Never serve the repository tree or choose another port.
[CmdletBinding()]
param([int]$Port = 8870, [switch]$PreviewOnly, [string]$Python = 'python')
$ErrorActionPreference = 'Stop'
if ($Port -ne 8870) { throw 'Qolda worktree Web uses http://127.0.0.1:8870/ only. No server was started.' }
$root = Split-Path -Parent $PSScriptRoot
$url = 'http://127.0.0.1:8870/'
$listeners = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
if ($listeners | Where-Object { $_.Port -eq $Port }) {
    Add-Type -AssemblyName System.Net.Http
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.UseProxy = $false
    $handler.AllowAutoRedirect = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(3)
    $client.MaxResponseContentBufferSize = 1048576
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $health = $client.GetStringAsync($url + 'api/v1/health').GetAwaiter().GetResult() | ConvertFrom-Json
        if ($health.version -ne 'seriq.v1') { throw 'Unexpected health response.' }
        foreach ($path in 'index.html', 'src/main.js', 'styles/atlas.css') {
            $route = if ($path -eq 'index.html') { $url } else { $url + $path }
            $bytes = $client.GetByteArrayAsync($route).GetAwaiter().GetResult()
            $actual = [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '')
            $expected = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $root ('frontend/' + $path))).Hash
            if ($actual -ne $expected) { throw "Different file: $path" }
        }
    } catch { throw "Port 8870 is occupied but the Qolda worktree could not be confirmed. No process was changed. $($_.Exception.Message)" }
    finally { $sha.Dispose(); $client.Dispose() }
    Write-Output "Qolda worktree Web: $url (already running; model A: $($health.models.scratch))."
    return
}
if (-not $PreviewOnly) {
    throw "No worktree server at $url. Start the local GPU engine as in frontend/README.md, or use -PreviewOnly for an explicitly model-free preview on this same port."
}
Write-Output "Qolda preview WITHOUT models: $url. Press Ctrl+C to stop."
Push-Location -LiteralPath $root
try {
    & $Python -B -E -m backend.api --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Qolda preview exited with code $LASTEXITCODE. No alternate port was selected." }
} finally { Pop-Location }
