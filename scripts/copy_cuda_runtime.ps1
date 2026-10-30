# Copy only NVIDIA runtime DLLs and their license notices into a new bundle.
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$SourceDirectory,
    [Parameter(Mandatory = $true)][string]$DestinationDirectory
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$source = [IO.Path]::GetFullPath($SourceDirectory).TrimEnd('\','/')
$destination = [IO.Path]::GetFullPath($DestinationDirectory).TrimEnd('\','/')
if ($destination -eq [IO.Path]::GetPathRoot($destination).TrimEnd('\','/')) { throw 'Choose a dedicated GPU folder.' }
if ($destination.StartsWith($source + '\', [StringComparison]::OrdinalIgnoreCase) -or
    $source.StartsWith($destination + '\', [StringComparison]::OrdinalIgnoreCase) -or $source -eq $destination) {
    throw 'CUDA source and destination must be separate, non-nested folders.'
}
function Assert-PlainPath([string]$Path) {
    $current = [IO.Path]::GetFullPath($Path)
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            if ((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw 'Reparse points are not accepted in runtime export.'
            }
        }
        $parent = [IO.Path]::GetDirectoryName($current.TrimEnd('\','/'))
        if ($parent -eq $current) { break }
        $current = $parent
    }
}
Assert-PlainPath $source
Assert-PlainPath $destination
$files = [Collections.Generic.List[object]]::new()
foreach ($library in @('cublas','cudnn','cuda_nvrtc','cuda_runtime')) {
    $folder = Join-Path $source "nvidia\$library\bin"
    if (-not (Test-Path -LiteralPath $folder)) {
        if ($library -ne 'cuda_runtime') { throw "Missing required NVIDIA runtime: $library" }
        continue
    }
    $dlls = @(Get-ChildItem -LiteralPath $folder -File -Filter '*.dll')
    if (-not $dlls.Count) { throw "No runtime DLLs found: $library" }
    foreach ($file in $dlls) { Assert-PlainPath $file.FullName; $files.Add($file) }
}
foreach ($package in Get-ChildItem -LiteralPath $source -Directory) {
    if ($package.Name -notmatch '^nvidia_(cublas|cudnn|cuda_nvrtc|cuda_runtime)_cu12-[0-9.]+\.dist-info$') { continue }
    Assert-PlainPath $package.FullName
    foreach ($file in Get-ChildItem -LiteralPath $package.FullName -Recurse -File) {
        if ($file.Name -match '(?i)license|copying|notice' -or $file.FullName -match '(?i)[\\/]licenses[\\/]') {
            Assert-PlainPath $file.FullName
            if ($file.Length -gt 3MB) { throw 'Unexpectedly large license notice.' }
            $files.Add($file)
        }
    }
}
$rows = @()
foreach ($file in $files) {
    $relative = $file.FullName.Substring($source.Length+1)
    $target = Join-Path $destination $relative
    Assert-PlainPath $target
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Test-Path -LiteralPath $target) -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hash) {
        throw 'Refusing to overwrite a different existing runtime file.'
    }
    $rows += [ordered]@{path=$relative.Replace('\','/');bytes=$file.Length;sha256=$hash}
}
if ($PSCmdlet.ShouldProcess($destination, 'Copy reviewed NVIDIA DLLs and license notices')) {
    foreach ($row in $rows) {
        $target = Join-Path $destination $row.path
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force
        if (-not (Test-Path -LiteralPath $target)) { Copy-Item -LiteralPath (Join-Path $source $row.path) -Destination $target }
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) { throw 'Copied CUDA file failed verification.' }
    }
    $manifest = [ordered]@{kind='public-nvidia-runtime';files=$rows}
    [IO.File]::WriteAllText((Join-Path $destination 'manifest.json'), ($manifest | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
}
Write-Host "NVIDIA runtime checked: $($rows.Count) files."
