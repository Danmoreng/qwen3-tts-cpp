# Qwen3-TTS-CPP Build Script
param (
    [switch]$Clean
)

# 1. Load Visual Studio Environment
function Import-VSEnv {
    Write-Host "Loading Visual Studio Environment..."
    $vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return }
    $vsroot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $vsroot) { return }
    $vcvars = Join-Path $vsroot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { return }
    $envDump = cmd /c "call `"$vcvars`" > nul && set PATH && set INCLUDE && set LIB"
    $envDump | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}
Import-VSEnv

# 2. Setup Directories
$ScriptDir = $PSScriptRoot
$BuildDir = Join-Path $ScriptDir "build"

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning build directory..."
    Remove-Item -Path $BuildDir -Recurse -Force
}

if (!(Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir
}
Set-Location $BuildDir

# 3. Check for Ninja
if (Get-Command ninja -ErrorAction SilentlyContinue) {
    Write-Host "Using Ninja generator..."
    $Generator = "-G Ninja"
} else {
    $Generator = ""
}

# 4. Configure CMake
Write-Host "Configuring CMake..."
# We are in cpp/build, and CMakeLists.txt is in cpp/ ($ScriptDir)
cmake $ScriptDir $Generator `
    -DGGML_CUDA=ON `
    -DCMAKE_CXX_STANDARD=17 `
    -DCMAKE_BUILD_TYPE=Release 

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

# 5. Build Targets

# A. Build llama (dependency)
Write-Host "Building llama/ggml dependencies..."
cmake --build . --config Release --target llama --parallel > build_llama.log 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Llama build failed!" -ForegroundColor Red
    Get-Content build_llama.log -Tail 20
    exit 1
}

# B. Build Qwen3-TTS CLI
Write-Host "Building Qwen3-TTS CLI..."
cmake --build . --config Release --target qwen3-tts-cli --parallel > build_cli.log 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Qwen3-TTS CLI build failed!" -ForegroundColor Red
    Get-Content build_cli.log -Tail 20
    exit 1
}

# C. Deploy DLLs to build root for easy execution
Write-Host "Deploying DLLs to build root..."
if (Test-Path "bin\*.dll") {
    Copy-Item -Path "bin\*.dll" -Destination "." -Force
}

Write-Host "Build success!" -ForegroundColor Green
Write-Host "Executable located at: $(Join-Path $BuildDir 'bin\Release\qwen3-tts-cli.exe') (or just bin/qwen3-tts-cli.exe depending on generator)"
