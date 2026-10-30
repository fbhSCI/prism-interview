# Copy only public model weights and NVIDIA runtime libraries into a portable
# bundle. No deletes, registry changes, credentials, papers or user indexes.
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$ModelDirectory,
    [Parameter(Mandatory = $true)][string]$CudaDirectory,
    [Parameter(Mandatory = $true)][string]$DestinationDirectory,
    [switch]$IncludeNllb,
    [switch]$IncludeStreamingModel,
    [switch]$IncludePunctuationModel,
    [switch]$IncludeSpeakerModel,
    [switch]$ReplaceRuntimeConfig
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-NoReparseAncestor {
    param([string]$Path)
    $candidate = [IO.Path]::GetFullPath($Path)
    while ($candidate) {
        if (Test-Path -LiteralPath $candidate) {
            $item = Get-Item -LiteralPath $candidate -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse points/symlinks are not accepted: $candidate"
            }
        }
        $candidateRoot = [IO.Path]::GetPathRoot($candidate)
        if ($candidate.TrimEnd('\', '/') -eq $candidateRoot.TrimEnd('\', '/')) { break }
        $parent = [IO.Path]::GetDirectoryName($candidate.TrimEnd('\', '/'))
        if ($parent -eq $candidate) { break }
        $candidate = $parent
    }
}

function Get-FullDirectory {
    param([string]$Value, [bool]$MustExist)
    $full = [IO.Path]::GetFullPath($Value).TrimEnd('\', '/')
    if ($MustExist -and -not (Test-Path -LiteralPath $full -PathType Container)) {
        throw "Directory does not exist: $full"
    }
    Assert-NoReparseAncestor $full
    return $full
}

function Test-Descendant {
    param([string]$Child, [string]$Parent)
    return $Child.Equals($Parent, [StringComparison]::OrdinalIgnoreCase) -or
        $Child.StartsWith($Parent.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar,
                          [StringComparison]::OrdinalIgnoreCase)
}

function Add-ExportFile {
    param([string]$Source, [string]$Relative, [string]$Kind, [bool]$Required = $false)
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        if ($Required) { throw "Required $Kind file missing: $Source" }
        return
    }
    Assert-NoReparseAncestor $Source
    $relativePath = $Relative.Replace('/', '\')
    $target = [IO.Path]::GetFullPath((Join-Path $script:TargetRoot $relativePath))
    if (-not (Test-Descendant $target $script:TargetRoot)) { throw 'Export target escaped its destination.' }
    Assert-NoReparseAncestor $target
    $item = Get-Item -LiteralPath $Source -Force
    if ($item.Length -le 0) { throw "Empty $Kind file: $Source" }
    $script:ExportFiles.Add([PSCustomObject]@{
        Source = $item.FullName; Relative = $relativePath.Replace('\', '/');
        Target = $target; Bytes = [long]$item.Length; Kind = $Kind
    })
}

function Add-KnownModelFiles {
    param([string]$SourceRoot, [string]$TargetPrefix, [string]$Kind,
          [string[]]$Required, [string[]]$Optional)
    foreach ($name in $Required) {
        Add-ExportFile (Join-Path $SourceRoot $name) "$TargetPrefix/$name" $Kind $true
    }
    foreach ($name in $Optional) {
        Add-ExportFile (Join-Path $SourceRoot $name) "$TargetPrefix/$name" $Kind $false
    }
}

function Get-SafeFiles {
    param([string]$Directory)
    Assert-NoReparseAncestor $Directory
    foreach ($item in Get-ChildItem -LiteralPath $Directory -Force) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point in selected runtime: $($item.FullName)" }
        if ($item.PSIsContainer) { Get-SafeFiles $item.FullName } else { $item }
    }
}

function Get-SHA256 {
    param([string]$Path)
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}

try {
    $sourceModels = Get-FullDirectory $ModelDirectory $true
    $sourceCuda = Get-FullDirectory $CudaDirectory $true
    $script:TargetRoot = Get-FullDirectory $DestinationDirectory $false
    $driveRoot = [IO.Path]::GetPathRoot($script:TargetRoot).TrimEnd('\', '/')
    if ($script:TargetRoot -eq $driveRoot) { throw 'Choose a dedicated portable application folder, not a drive root.' }
    foreach ($sourceRoot in @($sourceModels, $sourceCuda)) {
        if ((Test-Descendant $script:TargetRoot $sourceRoot) -or (Test-Descendant $sourceRoot $script:TargetRoot)) {
            throw 'Source and destination directories must be separate, never nested.'
        }
    }
    $configPath = Join-Path $script:TargetRoot 'runtime.json'
    if ((Test-Path -LiteralPath $configPath) -and -not $ReplaceRuntimeConfig) {
        throw 'runtime.json already exists. Use -ReplaceRuntimeConfig deliberately after reviewing the target folder.'
    }
    Assert-NoReparseAncestor $configPath
    $script:ExportFiles = [Collections.Generic.List[object]]::new()
    $asrRoot = Join-Path $sourceModels 'asr\large-v3-turbo'
    Add-KnownModelFiles $asrRoot 'runtime/models/asr/large-v3-turbo' 'asr' `
        @('model.bin', 'config.json', 'tokenizer.json') `
        @('vocabulary.json', 'vocabulary.txt', 'preprocessor_config.json', 'README.md', 'LICENSE', 'LICENSE.txt', 'WHISPER_LICENSE.txt', 'MODEL_SOURCE.txt')
    $asrConfig = Get-Content -LiteralPath (Join-Path $asrRoot 'config.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $asrConfig) { throw 'ASR config.json is invalid.' }

    $translationRoot = Join-Path $sourceModels 'translation'
    if (-not (Test-Path -LiteralPath $translationRoot -PathType Container)) { throw 'Argos translation directory is missing.' }
    $candidates = @($translationRoot)
    $candidates += @(Get-ChildItem -LiteralPath $translationRoot -Directory -Force | ForEach-Object { $_.FullName })
    $argosPackages = @()
    foreach ($candidate in $candidates) {
        Assert-NoReparseAncestor $candidate
        $metadataPath = Join-Path $candidate 'metadata.json'
        if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) { continue }
        $metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($metadata.from_code -eq 'en' -and $metadata.to_code -eq 'zh') { $argosPackages += $candidate }
    }
    if ($argosPackages.Count -ne 1) { throw 'Expected exactly one installed Argos en-to-zh package in translation/.' }
    $argosRoot = $argosPackages[0]
    $argosName = Split-Path -Leaf $argosRoot
    $argosPrefix = 'runtime/models/translation/' + $argosName
    Add-KnownModelFiles $argosRoot $argosPrefix 'argos' `
        @('metadata.json', 'sentencepiece.model', 'model/model.bin', 'model/config.json') `
        @('README.md', 'LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING', 'model/shared_vocabulary.json',
          'model/source_vocabulary.json', 'model/target_vocabulary.json')

    if ($IncludeNllb) {
        $nllbRoot = Join-Path $sourceModels 'translation-nllb'
        Add-KnownModelFiles $nllbRoot 'runtime/models/translation-nllb' 'nllb-noncommercial' `
            @('model.bin', 'tokenizer.json', 'shared_vocabulary.json', 'config.json',
              'tokenizer_config.json', 'special_tokens_map.json', 'README.md',
              'nllb-install.json', 'MODEL_LICENSE_NOTICE.txt') @()
        $nllbManifest = Get-Content -LiteralPath (Join-Path $nllbRoot 'nllb-install.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($nllbManifest.model_id -ne 'OpenNMT/nllb-200-distilled-1.3B-ct2-int8' -or
            $nllbManifest.revision -ne '70f572adafa4794890ce7826156a4209717855af') {
            throw 'NLLB manifest does not match the supported pinned model.'
        }
        Write-Host 'Including optional NLLB: CC-BY-NC-4.0, noncommercial research use only.' -ForegroundColor Yellow
    }

    if ($IncludeStreamingModel) {
        $streamingRoot = Join-Path $sourceModels 'streaming-asr\fastconformer-ctc-en-480ms'
        Add-KnownModelFiles $streamingRoot 'runtime/models/streaming-asr/fastconformer-ctc-en-480ms' 'streaming-asr' `
            @('model.int8.onnx', 'tokens.txt', 'model.json') `
            @('README.md', 'LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING')
        $streamingManifest = Get-Content -LiteralPath (Join-Path $streamingRoot 'model.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($streamingManifest.repository -ne 'csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms-int8' -or
            $streamingManifest.revision -ne '66e9b61a9fe334eba579700599e391c6ee093633' -or
            $streamingManifest.license -ne 'CC-BY-4.0') {
            throw 'Streaming ASR manifest does not match the supported pinned NVIDIA FastConformer model.'
        }
        Write-Host 'Including local FastConformer streaming drafts: NVIDIA CC-BY-4.0 model.'
    }

    if ($IncludePunctuationModel) {
        $punctuationRoot = Join-Path $sourceModels 'punctuation\edge-punct-en'
        Add-KnownModelFiles $punctuationRoot 'runtime/models/punctuation/edge-punct-en' 'punctuation' `
            @('model.int8.onnx', 'bpe.vocab', 'LICENSE', 'model.json') @('README.md')
        $punctuationManifest = Get-Content -LiteralPath (Join-Path $punctuationRoot 'model.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($punctuationManifest.archive_sha256 -ne '9f5e5a72c7d2829635bd074fce92b6bbd5b78da8a52e7ad8ed1be933f366b99d' -or
            $punctuationManifest.license -ne 'Apache-2.0') {
            throw 'Punctuation manifest does not match the supported pinned English punctuation model.'
        }
        Write-Host 'Including local English punctuation: Edge-Punct-Casing Apache-2.0 model.'
    }

    if ($IncludeSpeakerModel) {
        $speakerRoot = Join-Path $sourceModels 'speaker'
        Add-KnownModelFiles $speakerRoot 'runtime/models/speaker' 'speaker-model' `
            @('nemo_en_titanet_large.onnx', 'speaker-model.json', 'NVIDIA_TitaNet_model_card.md') `
            @('LICENSE', 'LICENSE.txt', 'MODEL_LICENSE_NOTICE.txt')
        $speakerMetadata = Get-Content -LiteralPath (Join-Path $speakerRoot 'speaker-model.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($speakerMetadata.license -ne 'CC-BY-4.0' -or
            $speakerMetadata.sha256 -ne 'd51abcf31717ef28162f26acb9d44dd4127c3d44c9b8624f699f3425daca8e77') {
            throw 'Speaker metadata does not match the supported NVIDIA TitaNet model.'
        }
    }

    # CudaDirectory may be a pip target or site-packages root. Only the known
    # NVIDIA DLL subtrees and package license texts are considered, never all of it.
    foreach ($library in @('cublas', 'cudnn', 'cuda_nvrtc', 'cuda_runtime')) {
        $binDirectory = Join-Path $sourceCuda "nvidia\$library\bin"
        if (-not (Test-Path -LiteralPath $binDirectory -PathType Container)) {
            if ($library -in @('cublas', 'cudnn', 'cuda_nvrtc')) { throw "Required NVIDIA runtime directory missing: $library/bin" }
            continue
        }
        $dlls = @(Get-SafeFiles $binDirectory | Where-Object { $_.Extension -ieq '.dll' })
        if (-not $dlls.Count) { throw "No DLLs in $library/bin" }
        foreach ($dll in $dlls) {
            $relative = $dll.FullName.Substring($sourceCuda.Length + 1)
            Add-ExportFile $dll.FullName ("runtime/gpu/" + $relative) 'nvidia-dll' $true
        }
    }
    foreach ($package in Get-ChildItem -LiteralPath $sourceCuda -Directory -Force) {
        if ($package.Name -notmatch '^nvidia_(cublas|cudnn|cuda_nvrtc|cuda_runtime)_cu12-[0-9.]+\.dist-info$') { continue }
        foreach ($file in Get-SafeFiles $package.FullName) {
            if ($file.Name -notmatch '(?i)license|copying|notice' -and $file.FullName -notmatch '(?i)[\\/]licenses[\\/]') { continue }
            if ($file.Length -gt 3MB) { throw 'Unexpectedly large NVIDIA license file.' }
            $relative = $file.FullName.Substring($sourceCuda.Length + 1)
            Add-ExportFile $file.FullName ("runtime/gpu/" + $relative) 'nvidia-license' $false
        }
    }
    $totalBytes = [long]0
    $existingBytes = [long]0
    foreach ($planned in $script:ExportFiles) {
        $totalBytes += $planned.Bytes
        if (Test-Path -LiteralPath $planned.Target -PathType Leaf) { $existingBytes += $planned.Bytes }
    }
    $drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($script:TargetRoot))
    $neededBytes = [long]($totalBytes - $existingBytes)
    if ($neededBytes -lt 0) { $neededBytes = [long]0 }
    $neededBytes += 256MB
    if ($drive.AvailableFreeSpace -lt $neededBytes) { throw 'Not enough free space on the destination drive.' }
    Write-Host ('Verified export plan: {0} whitelisted files, {1:N2} GiB.' -f $script:ExportFiles.Count, ($totalBytes / 1GB))
    if (-not $PSCmdlet.ShouldProcess($script:TargetRoot, 'Copy selected public models and NVIDIA DLLs; write relative runtime.json')) { exit 0 }

    $records = [Collections.Generic.List[object]]::new()
    $count = 0
    foreach ($file in $script:ExportFiles) {
        $count++
        Write-Host ("[{0}/{1}] {2}" -f $count, $script:ExportFiles.Count, $file.Relative)
        $sourceHash = Get-SHA256 $file.Source
        if (Test-Path -LiteralPath $file.Target) {
            if (-not (Test-Path -LiteralPath $file.Target -PathType Leaf)) { throw 'An export file collides with a directory.' }
            $targetHash = Get-SHA256 $file.Target
            if ($targetHash -ne $sourceHash) { throw "Existing destination differs; no overwrite performed: $($file.Relative)" }
        } else {
            $parentDirectory = Split-Path -Parent $file.Target
            New-Item -ItemType Directory -Path $parentDirectory -Force | Out-Null
            Assert-NoReparseAncestor $file.Target
            Copy-Item -LiteralPath $file.Source -Destination $file.Target
            $targetHash = Get-SHA256 $file.Target
            if ($targetHash -ne $sourceHash) { throw "Copy integrity failed: $($file.Relative)" }
        }
        $records.Add([ordered]@{ path = $file.Relative; bytes = $file.Bytes; sha256 = $sourceHash; kind = $file.Kind })
    }
    $manifestPath = Join-Path $script:TargetRoot 'runtime\manifest.json'
    Assert-NoReparseAncestor $manifestPath
    $manifest = [ordered]@{
        format = 1; generated_utc = [DateTime]::UtcNow.ToString('o');
        asr = 'large-v3-turbo'; translation = 'argos'; nllb_included = [bool]$IncludeNllb;
        streaming_model_included = [bool]$IncludeStreamingModel;
        punctuation_model_included = [bool]$IncludePunctuationModel;
        includes_private_data = $false; files = $records.ToArray()
    }
    [IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
    $runtime = [ordered]@{ data_dir = 'data'; model_dir = 'runtime/models'; cuda_dir = 'runtime/gpu' }
    [IO.File]::WriteAllText($configPath, ($runtime | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    Write-Host 'Export complete. All copied files were SHA256-verified; runtime paths are relative.' -ForegroundColor Green
    Write-Host 'No papers, indexes, settings, transcripts, API keys or Codex credentials were copied.'
    Write-Host 'Runtime copying is not an inference/Zoom test. Run target-machine acceptance checks separately.'
    exit 0
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
