# Source installation/model preparation. Changes environment variables only in
# this process. No API keys, Codex credentials, global PATH or machine settings.
[CmdletBinding()]
param(
    [ValidateSet('Install', 'Translation', 'Models')][string]$Action = 'Install',
    [string]$ModelDirectory = '',
    [string]$DataDirectory = '',
    [string]$Python = '',
    [string]$TemporaryDirectory = '',
    [switch]$Gpu,
    [switch]$DownloadAsr,
    [switch]$DownloadTranslation,
    [ValidateSet('large-v3-turbo', 'large-v3', 'medium.en', 'small.en')]
    [string]$AsrModel = 'large-v3-turbo',
    [ValidateSet('argos', 'nllb')][string]$TranslationEngine = 'argos',
    [switch]$NoPrompt
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Executable" }
}

function Assert-FreeSpace {
    param([string]$Directory, [double]$RequiredGB)
    $absolute = [IO.Path]::GetFullPath($Directory)
    $root = [IO.Path]::GetPathRoot($absolute)
    try { $disk = [IO.DriveInfo]::new($root); $freeGB = $disk.AvailableFreeSpace / 1GB }
    catch { throw "Cannot check free space for $root. Choose a local drive with at least $RequiredGB GB free." }
    if ($freeGB -lt $RequiredGB) {
        throw ('Not enough space on {0}: {1:N1} GB free; at least {2:N1} GB required.' -f $root, $freeGB, $RequiredGB)
    }
    Write-Host ('Space check: {0} has {1:N1} GB free.' -f $root, $freeGB)
}

function Find-Python {
    if ($Python) { return (Get-Command $Python -ErrorAction Stop).Source }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @('-3.12', '-3.11')) {
            try {
                $found = @(& py $version -c 'import sys; print(sys.executable)' 2>$null)
                if ($LASTEXITCODE -eq 0 -and $found.Count) { return [string]$found[-1] }
            } catch { continue }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { return (Get-Command python).Source }
    throw 'Install CPython 3.12 or 3.11 x64 from https://www.python.org/downloads/windows/ and retry.'
}

function Resolve-ConfiguredPath {
    param([string]$Value)
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Value))
}

try {
    Write-Host 'Prism Interview - Windows x64 setup' -ForegroundColor Cyan
    $temporaryDirectory = Join-Path $PSScriptRoot 'work\temp'
    New-Item -ItemType Directory -Path $temporaryDirectory -Force | Out-Null
    $env:TEMP = $temporaryDirectory
    $env:TMP = $temporaryDirectory
    $env:PYTHONUTF8 = '1'
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

    $runtimePath = Join-Path $PSScriptRoot 'runtime.json'
    $existing = @{}
    if (Test-Path -LiteralPath $runtimePath) {
        $parsed = Get-Content -LiteralPath $runtimePath -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($property in $parsed.PSObject.Properties) { $existing[$property.Name] = $property.Value }
    }
    if (-not $ModelDirectory -and $existing.ContainsKey('model_dir') -and $existing['model_dir']) {
        $ModelDirectory = Resolve-ConfiguredPath ([string]$existing['model_dir'])
    }
    if (-not $ModelDirectory) {
        if ($NoPrompt) { throw 'Pass -ModelDirectory explicitly. Setup never silently puts model weights on C:.' }
        $suggestion = ''
        $systemRoot = $env:SystemDrive + '\'
        $otherDrive = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Root -ne $systemRoot -and $_.Free -gt 8GB } | Select-Object -First 1
        if ($otherDrive) { $suggestion = Join-Path $otherDrive.Root 'PrismInterview\models' }
        Write-Host 'Select the model directory. Prefer a non-system drive with 8 GB or more free.'
        if ($suggestion) {
            $selection = Read-Host "Model directory [Enter = $suggestion]"
            $ModelDirectory = if ($selection) { $selection } else { $suggestion }
        } else {
            $ModelDirectory = Read-Host 'Model directory (for example D:\PrismModels; required)'
        }
        if (-not $ModelDirectory) { throw 'No model directory selected.' }
    }
    $ModelDirectory = Resolve-ConfiguredPath $ModelDirectory
    Assert-FreeSpace $ModelDirectory 6
    # Some native wheel installers cannot handle Unicode in TEMP even though
    # Python itself can. Prefer a task-local ASCII path, never a machine setting.
    if (-not $TemporaryDirectory) {
        $TemporaryDirectory = Join-Path $PSScriptRoot 'work\temp'
        if ($TemporaryDirectory -match '[^\x00-\x7F]') {
            $TemporaryDirectory = Join-Path $ModelDirectory '.setup-temp'
        }
    }
    $TemporaryDirectory = Resolve-ConfiguredPath $TemporaryDirectory
    if ($TemporaryDirectory -match '[^\x00-\x7F]') {
        if ($NoPrompt) { throw 'Pass -TemporaryDirectory with an ASCII-only path, for example D:\PrismTemp.' }
        $TemporaryDirectory = Read-Host 'ASCII-only temporary directory required (for example D:\PrismTemp)'
        $TemporaryDirectory = Resolve-ConfiguredPath $TemporaryDirectory
        if ($TemporaryDirectory -match '[^\x00-\x7F]') { throw 'Temporary directory must use ASCII characters.' }
    }
    New-Item -ItemType Directory -Path $TemporaryDirectory -Force | Out-Null
    $env:TEMP = $TemporaryDirectory
    $env:TMP = $TemporaryDirectory
    if (-not $DataDirectory) {
        $DataDirectory = if ($existing.ContainsKey('data_dir') -and $existing['data_dir']) {
            Resolve-ConfiguredPath ([string]$existing['data_dir'])
        } else { Join-Path $PSScriptRoot 'data' }
    }
    $DataDirectory = Resolve-ConfiguredPath $DataDirectory
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

    $gpuInstall = $false
    if ($Action -eq 'Install') {
        $gpuInstall = [bool]$Gpu
        if (-not $NoPrompt -and -not $Gpu) {
            $choice = Read-Host 'Install optional NVIDIA CUDA runtime DLLs for RTX GPU? [Y/n]'
            $gpuInstall = $choice -notmatch '^[Nn]'
        }
        Assert-FreeSpace $PSScriptRoot $(if ($gpuInstall) { 8 } else { 4 })
        if (-not (Test-Path -LiteralPath $venvPython)) {
            $systemPython = Find-Python
            Invoke-Checked $systemPython @('-c', 'import sys,struct; assert sys.version_info[:2] in ((3,11),(3,12)), ''Python 3.11/3.12 required''; assert struct.calcsize(''P'')==8, ''64-bit Python required''; print(sys.version)')
            Invoke-Checked $systemPython @('-m', 'venv', (Join-Path $PSScriptRoot '.venv'))
        }
        Invoke-Checked $venvPython @('-c', 'import sys,struct; assert sys.version_info[:2] in ((3,11),(3,12)); assert struct.calcsize(''P'')==8')
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '--only-binary=:all:', '-r', (Join-Path $PSScriptRoot 'requirements.txt'))
        if ($gpuInstall) {
            Invoke-Checked $venvPython @('-m', 'pip', 'install', '--only-binary=:all:', '-r', (Join-Path $PSScriptRoot 'requirements-gpu.txt'))
        }
    } elseif (-not (Test-Path -LiteralPath $venvPython)) {
        throw 'Run install_windows.bat before preparing source-edition models.'
    }
    Invoke-Checked $venvPython @('-c', 'import sys,struct; assert sys.version_info[:2] in ((3,11),(3,12)); assert struct.calcsize(''P'')==8')

    New-Item -ItemType Directory -Path $ModelDirectory, $DataDirectory -Force | Out-Null
    $cudaPath = if (-not $gpuInstall -and $existing.ContainsKey('cuda_dir') -and $existing['cuda_dir']) {
        Resolve-ConfiguredPath ([string]$existing['cuda_dir'])
    } else { Join-Path $PSScriptRoot '.venv\Lib\site-packages' }
    $runtime = [ordered]@{ data_dir = $DataDirectory; model_dir = $ModelDirectory; cuda_dir = $cudaPath }
    [IO.File]::WriteAllText($runtimePath, ($runtime | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    $env:PRISM_INTERVIEW_HOME = $DataDirectory
    $env:PRISM_INTERVIEW_MODELS = $ModelDirectory
    $env:PRISM_INTERVIEW_CUDA_DIR = $cudaPath

    if (-not $NoPrompt -and $Action -eq 'Install' -and -not $DownloadAsr -and -not $DownloadTranslation) {
        $downloadChoice = Read-Host 'Download the Whisper ASR and English-Chinese model now? [Y/n]'
        if ($downloadChoice -notmatch '^[Nn]') { $DownloadAsr = $true; $DownloadTranslation = $true }
    }
    if ($Action -eq 'Models') { $DownloadAsr = $true; $DownloadTranslation = $true }
    if ($Action -eq 'Translation') { $DownloadTranslation = $true }
    if ($DownloadAsr) {
        Assert-FreeSpace $ModelDirectory 4
        Write-Host "Downloading ASR weights: $AsrModel (this can take several minutes)"
    }
    if ($DownloadTranslation) {
        Assert-FreeSpace $ModelDirectory 1
    }
    if ($DownloadAsr -or $DownloadTranslation) {
        $modelArguments = @('-m', 'zoom_copilot.setup_models', '--model-dir', $ModelDirectory, '--asr', $AsrModel)
        if (-not $DownloadAsr) { $modelArguments += '--skip-asr' }
        if ($DownloadTranslation) { $modelArguments += @('--translation', '--translation-engine', $TranslationEngine) }
        Invoke-Checked $venvPython $modelArguments
    }
    Invoke-Checked $venvPython @('-c', 'import PySide6, numpy, scipy, soundcard, faster_whisper, ctranslate2, sentencepiece, pypdf, av; assert int(av.__version__.split(''.'')[0])<16, ''PyAV must remain below 16''; print(''Core imports OK; CTranslate2'',ctranslate2.__version__,''PyAV'',av.__version__)')
    Write-Host ''
    Write-Host 'Setup complete. Start with start_windows.bat.' -ForegroundColor Green
    Write-Host "Models: $ModelDirectory"
    Write-Host "User index/preferences: $DataDirectory"
    Write-Host "Process-only TEMP: $TemporaryDirectory"
    Write-Host 'Codex is optional and separate: install the official Codex CLI, then run codex login on THIS computer.'
    Write-Host 'No cloud API is enabled, no credentials copied, and no API credit purchased by setup.'
    exit 0
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
