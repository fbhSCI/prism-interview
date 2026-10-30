# Build a Windows x64 onedir application and a source-only archive.
# Invoke deliberately; installation, model downloads and inference are separate.
[CmdletBinding()]
param(
    [string]$OutputDirectory = '',
    [string]$PythonExecutable = '',
    [string]$BuildDirectory = '',
    [string]$CudaDirectory = '',
    [switch]$InstallBuildDependencies,
    [switch]$IncludeGpuLibraries,
    [switch]$SkipSourceArchive,
    [switch]$Console
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot
$OutputEncoding = [Text.UTF8Encoding]::new($false)

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Executable" }
}

try {
    $pythonPath = if ($PythonExecutable) { [IO.Path]::GetFullPath($PythonExecutable) } else { Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create .venv with setup_windows.ps1 first.' }
    if (-not $OutputDirectory) { $OutputDirectory = Join-Path $PSScriptRoot 'dist' }
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    if (-not $BuildDirectory) { $BuildDirectory = Join-Path $PSScriptRoot 'work' }
    $BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
    $buildWork = Join-Path $BuildDirectory 'pyinstaller'
    $temporaryDirectory = Join-Path $BuildDirectory 'temp'
    New-Item -ItemType Directory -Path $buildWork, $temporaryDirectory -Force | Out-Null
    $env:TEMP = $temporaryDirectory
    $env:TMP = $temporaryDirectory
    $env:PYTHONUTF8 = '1'
    # Never inherit Poppler/Conda/other application DLL directories during
    # dependency analysis. Their ICU/OpenSSL DLLs can shadow Windows/Qt DLLs.
    $basePythonResult = @(& $pythonPath -c 'import sys; print(sys.base_prefix)')
    if ($LASTEXITCODE -ne 0 -or -not $basePythonResult.Count) { throw 'Cannot locate the base Python runtime.' }
    $basePython = [string]$basePythonResult[-1]
    $safeBuildDirectories = @((Split-Path -Parent $pythonPath), $basePython,
        (Join-Path $basePython 'DLLs'), (Join-Path $env:SystemRoot 'System32'), $env:SystemRoot)
    $env:PATH = (($safeBuildDirectories | Where-Object { Test-Path -LiteralPath $_ -PathType Container } | Select-Object -Unique) -join ';')
    if ($InstallBuildDependencies) {
        Invoke-Checked $pythonPath @('-m', 'pip', 'install', '-r', (Join-Path $PSScriptRoot 'requirements-dev.txt'))
    }
    Invoke-Checked $pythonPath @('-m', 'PyInstaller', '--version')
    $arguments = @('-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
        '--name', 'PrismInterview', '--contents-directory', '_internal',
        '--distpath', $OutputDirectory, '--workpath', $buildWork, '--specpath', $buildWork,
        '--collect-all', 'ctranslate2', '--collect-all', 'faster_whisper',
        '--collect-all', 'sentencepiece', '--collect-all', 'soundcard',
        '--collect-all', 'sherpa_onnx',
        '--collect-all', 'av', '--exclude-module', 'pymupdf', '--exclude-module', 'fitz',
        '--exclude-module', 'torch', '--exclude-module', 'stanza',
        '--exclude-module', 'argostranslate', '--exclude-module', 'tkinter',
        '--exclude-module', 'pytest', '--exclude-module', 'onnxruntime.tools',
        '--exclude-module', 'onnxruntime.quantization')
    if ($IncludeGpuLibraries -and -not $CudaDirectory) {
        Invoke-Checked $pythonPath @('-c', 'import nvidia.cublas, nvidia.cudnn, nvidia.cuda_nvrtc')
        foreach ($package in @('nvidia.cublas', 'nvidia.cudnn', 'nvidia.cuda_nvrtc', 'nvidia.cuda_runtime')) {
            $arguments += @('--collect-all', $package)
        }
    } else { $arguments += @('--exclude-module', 'nvidia') }
    $iconPath = Join-Path $PSScriptRoot 'docs\assets\app.ico'
    if (Test-Path -LiteralPath $iconPath) { $arguments += @('--icon', $iconPath) }
    $assetPath = Join-Path $PSScriptRoot 'zoom_copilot\assets'
    if (Test-Path -LiteralPath $assetPath) { $arguments += @('--add-data', ($assetPath + ';zoom_copilot/assets')) }
    $arguments += $(if ($Console) { '--console' } else { '--windowed' })
    $arguments += (Join-Path $PSScriptRoot 'run.py')
    Invoke-Checked $pythonPath $arguments

    $bundle = Join-Path $OutputDirectory 'PrismInterview'
    if ($IncludeGpuLibraries -and $CudaDirectory) {
        & (Join-Path $PSScriptRoot 'scripts\copy_cuda_runtime.ps1') -SourceDirectory $CudaDirectory -DestinationDirectory (Join-Path $bundle 'gpu')
    }
    foreach ($filename in @('README_中文.md', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'start_windows.bat', 'export_runtime.ps1')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $filename) -Destination $bundle -Force
    }
    $calibrationWave = Join-Path $PSScriptRoot 'examples\calibration.wav'
    if (Test-Path -LiteralPath $calibrationWave -PathType Leaf) {
        $exampleDirectory = Join-Path $bundle 'examples'
        New-Item -ItemType Directory -Path $exampleDirectory -Force | Out-Null
        Copy-Item -LiteralPath $calibrationWave -Destination $exampleDirectory
    }
    # Never copy source runtime.json: it can contain machine-specific paths.
    $runtime = [ordered]@{ data_dir = 'data'; model_dir = 'models'; cuda_dir = 'gpu' }
    [IO.File]::WriteAllText((Join-Path $bundle 'runtime.json'), ($runtime | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'docs') -Destination (Join-Path $bundle 'docs') -Recurse
    $licenseDirectory = Join-Path $bundle 'licenses'
    New-Item -ItemType Directory -Path $licenseDirectory -Force | Out-Null
    # Capture installed dependency versions and upstream license texts only.
    @'
import hashlib, importlib.metadata as m, json, pathlib, sys
dest = pathlib.Path(sys.argv[1]); rows=[]
for d in sorted(m.distributions(), key=lambda x: x.metadata.get('Name','').lower()):
    name=d.metadata.get('Name','unknown'); version=d.version
    if name.lower() in {'pip','setuptools','pytest','pyinstaller','pyinstaller-hooks-contrib','pymupdf'}: continue
    rows.append({'name':name,'version':version,'license':d.metadata.get('License-Expression') or d.metadata.get('License',''), 'home':d.metadata.get('Home-page','')})
    for entry in d.files or []:
        lower=str(entry).lower()
        if '.dist-info/' in lower and ('/licenses/' in lower or any(s in pathlib.PurePath(lower).name for s in ('license','copying','notice'))):
            source=pathlib.Path(d.locate_file(entry))
            if source.is_file() and source.stat().st_size < 3_000_000:
                digest=hashlib.sha256(str(entry).encode('utf-8')).hexdigest()[:10]
                target=dest/(name+'-'+digest+'-'+pathlib.PurePath(entry).name)
                target.write_bytes(source.read_bytes())
(dest/'installed-dependencies.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
'@ | & $pythonPath - $licenseDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Could not collect dependency license notices.' }

    # Exact whitelist: never package a venv, papers, session logs, user indexes,
    # runtime preferences, authentication files or API keys into the source ZIP.
    if (-not $SkipSourceArchive) {
        $archivePath = Join-Path $OutputDirectory 'PrismInterview_Source.zip'
        @'
import pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2])
files=['run.py','README.md','README_中文.md','LICENSE','THIRD_PARTY_NOTICES.md','CONTRIBUTING.md','SECURITY.md','.gitignore',
       'requirements.txt','requirements-translate.txt','requirements-gpu.txt','requirements-dev.txt',
       'install_windows.bat','start_windows.bat','install_translation_windows.bat',
       'setup_windows.ps1','build_windows.ps1','export_runtime.ps1','AGENTS.md','docs/.nojekyll']
for folder in ('zoom_copilot','tests','docs','scripts','.github'):
    files.extend(str(p.relative_to(root)) for p in (root/folder).rglob('*')
                 if p.is_file() and p.suffix in {'.py','.md','.json','.html','.css','.js','.svg','.png','.yml','.yaml','.ps1','.txt','.ico'} and '__pycache__' not in p.parts)
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
    for item in sorted(set(files)):
        p=root/item
        if p.is_file(): archive.write(p,str(pathlib.Path('PrismInterview')/item))
print(output)
'@ | & $pythonPath - $PSScriptRoot $archivePath
        if ($LASTEXITCODE -ne 0) { throw 'Could not build source archive.' }
    }
    Write-Host "Portable folder: $bundle" -ForegroundColor Green
    Write-Host 'Copy the ENTIRE folder, including _internal. Weights, GPU DLLs and Codex CLI are separate unless explicitly included.'
    Write-Host 'Review THIRD_PARTY_NOTICES.md before public redistribution (including Qt/FFmpeg and separate model licenses).'
    exit 0
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
