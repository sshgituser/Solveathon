[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$OutputDirectory)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$desktop = $PSScriptRoot
$repo = [IO.Path]::GetFullPath((Join-Path $desktop '..')).TrimEnd('\')
$destination = [IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\')
if ($destination.Equals($repo, [StringComparison]::OrdinalIgnoreCase) -or
    $destination.StartsWith($repo + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Choose an output directory outside the repository.'
}
if ((Test-Path -LiteralPath $destination) -and
    (!(Test-Path -LiteralPath $destination -PathType Container) -or
     (Get-ChildItem -LiteralPath $destination -Force | Select-Object -First 1))) {
    throw 'Choose a new or empty output directory; nothing will be overwritten.'
}

$cache = Join-Path $desktop '.nuget'
$env:DOTNET_CLI_HOME = $cache
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'
$env:DOTNET_NOLOGO = '1'
$env:NUGET_PACKAGES = Join-Path $cache 'packages'
$env:NUGET_HTTP_CACHE_PATH = Join-Path $cache 'http-cache'
$env:TEMP = Join-Path $cache 'temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$project = Join-Path $desktop 'Qolda.csproj'
& dotnet restore $project --locked-mode --nologo
if ($LASTEXITCODE) { throw "dotnet restore failed ($LASTEXITCODE)." }
& dotnet publish $project -c Release -r win-x64 --self-contained true --no-restore --nologo
if ($LASTEXITCODE) { throw "dotnet publish failed ($LASTEXITCODE)." }

$publish = Join-Path $desktop 'bin/Release/net10.0-windows10.0.17763.0/win-x64/publish'
$assets = Get-Content -LiteralPath (Join-Path $desktop 'obj/project.assets.json') -Raw | ConvertFrom-Json
$framework = $assets.project.frameworks.'net10.0-windows10.0.17763.0'
function Get-RuntimePackage([string]$Name) {
    $dependency = $framework.downloadDependencies | Where-Object { $_.name -eq $Name }
    if (!$dependency -or $dependency.version -notmatch '^\[(\d+\.\d+\.\d+), \1\]$') {
        throw "Cannot determine the resolved runtime package: $Name"
    }
    return Join-Path (Join-Path $env:NUGET_PACKAGES $Name.ToLowerInvariant()) $Matches[1]
}
$netCore = Get-RuntimePackage 'Microsoft.NETCore.App.Runtime.win-x64'
$winForms = Get-RuntimePackage 'Microsoft.WindowsDesktop.App.Runtime.win-x64'
$webPackage = $assets.libraries.PSObject.Properties | Where-Object { $_.Name -like 'Microsoft.Web.WebView2/*' }
$webView = Join-Path $env:NUGET_PACKAGES $webPackage.Value.path

# Only these public build inputs can enter the desktop archive.
$files = [ordered]@{
    'Qolda.exe' = (Join-Path $publish 'Qolda.exe')
    'WebView2Loader.dll' = (Join-Path $publish 'WebView2Loader.dll')
    'runtimes/win-x64/native/WebView2Loader.dll' = (Join-Path $publish 'runtimes/win-x64/native/WebView2Loader.dll')
    'README.md' = (Join-Path $desktop 'README.md')
    'qolda.example.json' = (Join-Path $desktop 'qolda.example.json')
    'licenses/dotnet-LICENSE.txt' = (Join-Path $netCore 'LICENSE.TXT')
    'licenses/dotnet-THIRD-PARTY-NOTICES.txt' = (Join-Path $netCore 'THIRD-PARTY-NOTICES.TXT')
    'licenses/windows-desktop-LICENSE.txt' = (Join-Path $winForms 'LICENSE')
    'licenses/webview2-LICENSE.txt' = (Join-Path $webView 'LICENSE.txt')
    'licenses/webview2-NOTICE.txt' = (Join-Path $webView 'NOTICE.txt')
}
foreach ($source in $files.Values) {
    if (!(Test-Path -LiteralPath $source -PathType Leaf)) { throw "Build input missing: $source" }
}
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$manifest = foreach ($name in ($files.Keys | Sort-Object)) {
    $file = Join-Path $destination $name
    New-Item -ItemType Directory -Path (Split-Path -Parent $file) -Force | Out-Null
    Copy-Item -LiteralPath $files[$name] -Destination $file
    $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $name"
}
$manifestName = 'SHA256SUMS.txt'
$manifest | Set-Content -LiteralPath (Join-Path $destination $manifestName) -Encoding Ascii

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = Join-Path $destination 'qolda-windows-win-x64.zip'
$zip = [IO.Compression.ZipFile]::Open($zipPath, 'Create')
try {
    foreach ($name in (@($files.Keys) + @($manifestName) | Sort-Object)) {
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, (Join-Path $destination $name), "windows/$name", 'Optimal') | Out-Null
    }
} finally { $zip.Dispose() }
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $destination 'Qolda.exe'),$zipPath
