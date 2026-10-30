# Install a new Prism Interview directory. Existing programs and user data are read-only.
[CmdletBinding()]
param(
    [string]$DestinationDirectory = '',
    [string]$BaseDirectory = '',
    [string]$RuntimeManifest = '',
    [string]$PayloadDirectory = '',
    [switch]$NoPrompt,
    [switch]$SkipShortcut
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$script:PrismInstallerRoot = $PSScriptRoot
$script:PrismVersion = '0.4.2'
$script:PrismBaseVersion = '0.4.0'
$script:PrismBaseUrl = 'https://github.com/fbhSCI/prism-interview/releases/download/v0.4.0/PrismInterview-0.4.0-Windows-x64.zip'
$script:PrismBaseHash = '8cf4cd726c7f685221d7ba0c203ad396756ceadeb99b34a0288fa63e5a945999'
$script:PrismBaseBytes = [long]1390833967

function Get-PrismFullPath([string]$Value) {
    if (-not $Value -or -not $Value.Trim()) { throw '文件夹路径不能为空。' }
    return [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($Value.Trim().Trim('"'))).TrimEnd('\','/')
}

function Assert-PrismPlainPath([string]$Path) {
    $current = [IO.Path]::GetFullPath($Path)
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            if ((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw '安装路径包含链接或重解析点；请改用普通文件夹。'
            }
        }
        $parent = [IO.Path]::GetDirectoryName($current.TrimEnd('\','/'))
        if ($parent -eq $current) { break }
        $current = $parent
    }
}

function ConvertTo-PrismRelativePath([string]$Path) {
    if (-not $Path -or [IO.Path]::IsPathRooted($Path) -or $Path -match '[:<>"|?*\x00-\x1f]') {
        throw '文件清单包含不安全的路径。'
    }
    $normalized = $Path.Replace('\','/').TrimEnd('/')
    $parts = $normalized.Split('/')
    foreach ($part in $parts) {
        if (-not $part -or $part -in @('.','..') -or $part.TrimEnd(' ','.') -ne $part -or
            $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?$') {
            throw '文件清单包含不安全的路径。'
        }
    }
    return $normalized
}

function Join-PrismContainedPath([string]$Root, [string]$Relative) {
    $relativePath = ConvertTo-PrismRelativePath $Relative
    $fullRoot = Get-PrismFullPath $Root
    $candidate = [IO.Path]::GetFullPath((Join-Path $fullRoot $relativePath))
    if (-not $candidate.StartsWith($fullRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw '文件路径超出安装目录。'
    }
    Assert-PrismPlainPath $candidate
    return $candidate
}

function Assert-PrismSeparatePaths([string]$Left, [string]$Right) {
    $a = Get-PrismFullPath $Left
    $b = Get-PrismFullPath $Right
    if ($a.Equals($b,[StringComparison]::OrdinalIgnoreCase) -or
        $a.StartsWith($b+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $b.StartsWith($a+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw '新安装目录必须与旧程序、安装文件分开放置，不能互相嵌套。'
    }
}

function Get-PrismHash([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}

function Read-PrismRuntimeManifest([string]$Path) {
    Assert-PrismPlainPath $Path
    $data = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($null -eq $data -or $data -is [Array] -or
        $data.PSObject.Properties.Name -notcontains 'schema_version' -or
        $data.PSObject.Properties.Name -notcontains 'base_version' -or
        $data.schema_version -ne 1 -or $data.base_version -ne $script:PrismBaseVersion -or
        $data.PSObject.Properties.Name -notcontains 'files' -or $data.files -isnot [Array] -or
        $data.files.Count -eq 0 -or $data.files.Count -gt 10000) {
        throw '运行库清单格式或基础版本不正确。请重新下载完整安装包。'
    }
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $rows = @()
    foreach ($item in $data.files) {
        if ($null -eq $item -or $item.PSObject.Properties.Name -notcontains 'path' -or
            $item.PSObject.Properties.Name -notcontains 'bytes' -or
            $item.PSObject.Properties.Name -notcontains 'sha256' -or $item.path -isnot [string]) {
            throw '运行库清单缺少必要字段。'
        }
        $relative = ConvertTo-PrismRelativePath $item.path
        if ($relative.Split('/')[0] -notin @('_internal','gpu','licenses') -or
            $relative -notmatch '/' -or -not $seen.Add($relative)) {
            throw '运行库清单包含未允许的目录或重复路径。'
        }
        if (($item.bytes -isnot [int] -and $item.bytes -isnot [long]) -or $item.bytes -lt 0 -or
            $item.sha256 -isnot [string] -or $item.sha256 -notmatch '^[a-fA-F0-9]{64}$') {
            throw '运行库清单的大小或校验值不正确。'
        }
        $rows += [PSCustomObject]@{path=$relative;bytes=[long]$item.bytes;sha256=$item.sha256.ToLowerInvariant()}
    }
    return $rows
}

function Get-PrismPayloadFiles([string]$Root) {
    Assert-PrismPlainPath $Root
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { throw '安装包缺少 payload 文件夹，请完整解压后再运行。' }
    $stack = [Collections.Generic.Stack[string]]::new()
    $stack.Push($Root)
    $rows = @()
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    while ($stack.Count) {
        $folder = $stack.Pop()
        foreach ($item in Get-ChildItem -LiteralPath $folder -Force) {
            Assert-PrismPlainPath $item.FullName
            $relative = ConvertTo-PrismRelativePath $item.FullName.Substring($Root.Length+1)
            $parts = $relative.Split('/')
            $isRuntimePatch = $parts[0] -eq 'runtime-patch'
            if ($isRuntimePatch) {
                # This release changes only Python's frozen standard library.
                # A reviewed patch has one explicit destination, never a general
                # permission to replace the old runtime or copy user data.
                if ($item.PSIsContainer) {
                    if ($relative -notin @('runtime-patch','runtime-patch/_internal')) {
                        throw "安装包包含未允许的运行库补丁目录：$relative"
                    }
                } elseif ($relative -ne 'runtime-patch/_internal/base_library.zip') {
                    throw "安装包包含未允许的运行库补丁：$relative"
                }
            } elseif ($parts[0] -notin @('docs','source','scripts') -and
                $relative -notin @('PrismInterview.exe','README.md','README_中文.md','README.en.md','LICENSE','THIRD_PARTY_NOTICES.md','AGENTS.md','start_windows.bat')) {
                throw "安装包包含未允许的内容：$relative"
            }
            foreach ($part in $parts) {
                if ($part -in @('.git','.venv','__pycache__','data','sessions','documents','models','gpu') -or
                    $part.StartsWith('data-',[StringComparison]::OrdinalIgnoreCase) -or $part.StartsWith('.env',[StringComparison]::OrdinalIgnoreCase)) {
                    throw "安装包包含用户数据或缓存目录：$relative"
                }
            }
            if ($item.PSIsContainer) { $stack.Push($item.FullName); continue }
            if ($item.Name -in @('runtime.json','auth.json','preferences.json','knowledge.json','research_facts.json','prepared_answers.json','portable_documents.json','session_api_keys.json') -or
                $item.Extension -in @('.pyc','.pyo','.wav','.mp3','.mp4')) { throw "安装包包含不应复制的记录或配置：$relative" }
            if ($isRuntimePatch) { $relative = '_internal/base_library.zip' }
            if (-not $seen.Add($relative)) { throw '安装包包含重复文件名。' }
            $rows += [PSCustomObject]@{path=$relative;source=$item.FullName;bytes=[long]$item.Length;sha256=(Get-PrismHash $item.FullName)}
        }
    }
    if (-not ($rows | Where-Object path -eq 'PrismInterview.exe')) { throw '安装包缺少 PrismInterview.exe。' }
    return $rows
}

function Assert-PrismFile([string]$Path, [long]$Bytes, [string]$Hash) {
    Assert-PrismPlainPath $Path
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or
        (Get-Item -LiteralPath $Path).Length -ne $Bytes -or
        (Get-PrismHash $Path) -ne $Hash) {
        throw "文件校验失败：$([IO.Path]::GetFileName($Path))。没有修改已有安装。"
    }
}

function Copy-PrismCheckedFile([string]$Source, [string]$Destination, [long]$Bytes, [string]$Hash) {
    Assert-PrismPlainPath $Source
    Assert-PrismPlainPath $Destination
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force
    [IO.File]::Copy($Source,$Destination,$false)
    Assert-PrismFile $Destination $Bytes $Hash
}

function Assert-PrismArchivePaths([string]$ZipPath) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        foreach ($entry in $archive.Entries) {
            $relative = ConvertTo-PrismRelativePath $entry.FullName
            if ($relative.Split('/')[0] -cne 'PrismInterview' -or -not $seen.Add($relative)) {
                throw '基础程序压缩包包含未知根目录或重复路径。'
            }
            $kind = ([long]$entry.ExternalAttributes -shr 16) -band 0xF000
            if ($kind -notin @(0,0x8000,0x4000) -or ($entry.ExternalAttributes -band 0x400)) {
                throw '基础程序压缩包包含链接或不支持的文件类型。'
            }
        }
    } finally { $archive.Dispose() }
}

function Assert-PrismFreeSpace([string]$Directory, [long]$Bytes) {
    try {
        $drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($Directory))
        if ($drive.AvailableFreeSpace -lt ($Bytes + 40MB)) { throw 'INSUFFICIENT_SPACE' }
    } catch {
        if ($_.Exception.Message -eq 'INSUFFICIENT_SPACE') { throw '目标磁盘空间不足。请选择其他安装目录；已有程序和数据不会被覆盖。' }
        if (-not $Directory.StartsWith('\\')) { throw }
        Write-Warning '无法预先查询网络磁盘空间；写入失败时不会提交安装。'
    }
}

function Get-PrismBaseArchive([string]$CacheRoot) {
    Assert-PrismPlainPath $CacheRoot
    $null = New-Item -ItemType Directory -Path $CacheRoot -Force
    $zipPath = Join-Path $CacheRoot 'PrismInterview-0.4.0-Windows-x64.zip'
    if (Test-Path -LiteralPath $zipPath) {
        Write-Host '正在检查已下载的基础程序…'
        try { Assert-PrismFile $zipPath $script:PrismBaseBytes $script:PrismBaseHash }
        catch { throw "下载缓存未通过校验。请移走此文件后重试，或指定旧程序文件夹：$zipPath" }
        return $zipPath
    }
    Assert-PrismFreeSpace $CacheRoot $script:PrismBaseBytes
    $partial = Join-Path $CacheRoot ('base-download-' + [Guid]::NewGuid().ToString('N') + '.partial')
    Write-Host '首次安装需要从官方 GitHub 下载约 1.4 GB 的基础运行库，可能需要数分钟。'
    Write-Host '以后更新可复用已安装程序，不必重复下载。请保持此窗口打开。'
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
            try { Start-BitsTransfer -Source $script:PrismBaseUrl -Destination $partial -DisplayName 'Prism Interview 基础程序' }
            catch {
                Write-Warning 'Windows 后台传输暂不可用，正在改用普通 HTTPS 下载。'
                # Keep an interrupted BITS file separate; never overwrite a file
                # that a background service might still be using.
                $partial = Join-Path $CacheRoot ('base-download-' + [Guid]::NewGuid().ToString('N') + '.partial')
                Invoke-WebRequest -Uri $script:PrismBaseUrl -OutFile $partial -UseBasicParsing
            }
        } else {
            Invoke-WebRequest -Uri $script:PrismBaseUrl -OutFile $partial -UseBasicParsing
        }
        Write-Host '下载完成，正在核对官方文件校验值…'
        Assert-PrismFile $partial $script:PrismBaseBytes $script:PrismBaseHash
        [IO.File]::Move($partial,$zipPath)
    } catch {
        throw "基础程序下载或校验失败。可稍后重试，或使用已安装的 0.4.0 文件夹。未修改已有程序。详细原因：$($_.Exception.Message)"
    }
    return $zipPath
}

function Install-PrismArchiveRuntime([string]$ZipPath, [string]$StageRoot, [object[]]$Rows) {
    Assert-PrismArchivePaths $ZipPath
    $archive = [IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($row in $Rows) {
            $entry = $archive.GetEntry('PrismInterview/' + $row.path)
            if ($null -eq $entry -or $entry.Length -ne $row.bytes -or $entry.FullName.EndsWith('/')) {
                throw "基础压缩包缺少匹配的运行库文件：$($row.path)"
            }
            $destination = Join-PrismContainedPath $StageRoot $row.path
            $null = New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force
            $input = $entry.Open()
            try {
                $output = [IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
                try { $input.CopyTo($output) } finally { $output.Dispose() }
            } finally { $input.Dispose() }
            Assert-PrismFile $destination $row.bytes $row.sha256
        }
    } finally { $archive.Dispose() }
}

function Invoke-PrismInstall {
    [CmdletBinding()]
    param([string]$DestinationDirectory='', [string]$BaseDirectory='', [string]$RuntimeManifest='',
          [string]$PayloadDirectory='', [switch]$NoPrompt, [switch]$SkipShortcut)
    $localData = [Environment]::GetEnvironmentVariable('LOCALAPPDATA')
    if (-not $localData) { $localData = [Environment]::GetFolderPath('LocalApplicationData') }
    if (-not $localData) { throw '无法定位当前用户的本地应用文件夹。' }
    if (-not $DestinationDirectory) { $DestinationDirectory = Join-Path $localData "Programs\PrismInterview\$script:PrismVersion" }
    $targetRoot = Get-PrismFullPath $DestinationDirectory
    if ($targetRoot -eq [IO.Path]::GetPathRoot($targetRoot).TrimEnd('\','/')) { throw '安装位置不能是磁盘根目录。' }
    Assert-PrismPlainPath $targetRoot
    if ((Test-Path -LiteralPath $targetRoot) -and
        (-not (Test-Path -LiteralPath $targetRoot -PathType Container) -or @(Get-ChildItem -LiteralPath $targetRoot -Force).Count)) {
        throw '目标文件夹已存在且不为空。请选择新目录，原有程序和数据保持不变。'
    }
    if (-not $RuntimeManifest) { $RuntimeManifest = Join-Path $script:PrismInstallerRoot 'trusted-runtime-manifest.json' }
    if (-not $PayloadDirectory) { $PayloadDirectory = Join-Path $script:PrismInstallerRoot 'payload' }
    $manifestPath = Get-PrismFullPath $RuntimeManifest
    $payloadRoot = Get-PrismFullPath $PayloadDirectory
    Assert-PrismSeparatePaths $targetRoot $payloadRoot
    $runtimeRows = @(Read-PrismRuntimeManifest $manifestPath)
    $payloadRows = @(Get-PrismPayloadFiles $payloadRoot)
    $payloadDestinations = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($row in $payloadRows) { $null = $payloadDestinations.Add($row.path) }
    # Replacements come solely from the new reviewed payload. Do not read,
    # validate or stage an old file that will not be used in this version.
    $runtimeRows = @($runtimeRows | Where-Object { -not $payloadDestinations.Contains($_.path) })
    Write-Host "Prism Interview $script:PrismVersion 安装"
    Write-Host "新程序位置：$targetRoot"
    Write-Host '安装器不会复制旧资料、设置、记录或密钥，也不会自动启动监听。'
    if (-not $BaseDirectory -and -not $NoPrompt) {
        Write-Host '如果已安装 0.4.0，可粘贴包含 PrismInterview.exe 和 _internal 的文件夹路径来复用运行库。'
        $BaseDirectory = Read-Host '旧程序文件夹（直接回车将从官方 GitHub 下载基础程序）'
    }
    $zipPath = ''
    if ($BaseDirectory) {
        $baseRoot = Get-PrismFullPath $BaseDirectory
        Assert-PrismPlainPath $baseRoot
        Assert-PrismSeparatePaths $targetRoot $baseRoot
        Write-Host '正在核对旧程序的公共运行库；只读取清单中的文件…'
        foreach ($row in $runtimeRows) { Assert-PrismFile (Join-PrismContainedPath $baseRoot $row.path) $row.bytes $row.sha256 }
    } else {
        $zipPath = Get-PrismBaseArchive (Join-Path $localData 'PrismInterviewInstallerCache')
        Assert-PrismArchivePaths $zipPath
    }
    $totalBytes = [long]0
    foreach ($row in @($runtimeRows)+@($payloadRows)) { $totalBytes += $row.bytes }
    Assert-PrismFreeSpace $targetRoot $totalBytes
    $parent = Split-Path -Parent $targetRoot
    Assert-PrismPlainPath $parent
    $null = New-Item -ItemType Directory -Path $parent -Force
    $stageRoot = Join-Path $parent ('.prism-' + [Guid]::NewGuid().ToString('N'))
    Assert-PrismPlainPath $stageRoot
    if (Test-Path -LiteralPath $stageRoot) { throw '无法创建独立临时安装目录。' }
    $null = New-Item -ItemType Directory -Path $stageRoot
    try {
        Write-Host '正在复制并校验运行库…'
        if ($zipPath) { Install-PrismArchiveRuntime $zipPath $stageRoot $runtimeRows }
        else {
            foreach ($row in $runtimeRows) {
                Copy-PrismCheckedFile (Join-PrismContainedPath $baseRoot $row.path) (Join-PrismContainedPath $stageRoot $row.path) $row.bytes $row.sha256
            }
        }
        Write-Host '正在安装新版本…'
        foreach ($row in $payloadRows) {
            Copy-PrismCheckedFile $row.source (Join-PrismContainedPath $stageRoot $row.path) $row.bytes $row.sha256
        }
        $runtime = [ordered]@{data_dir='data';model_dir='models';cuda_dir='gpu'}
        [IO.File]::WriteAllText((Join-Path $stageRoot 'runtime.json'),($runtime|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
        $receipt = [ordered]@{version=$script:PrismVersion;base_version=$script:PrismBaseVersion;runtime_files=$runtimeRows;payload_files=@($payloadRows|Select-Object path,bytes,sha256)}
        [IO.File]::WriteAllText((Join-Path $stageRoot 'installation-manifest.json'),($receipt|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
        Assert-PrismPlainPath $targetRoot
        if (Test-Path -LiteralPath $targetRoot) {
            if (-not (Test-Path -LiteralPath $targetRoot -PathType Container) -or @(Get-ChildItem -LiteralPath $targetRoot -Force).Count) {
                throw '安装过程中目标目录出现了其他文件，已停止以避免覆盖。'
            }
            # Non-recursive removal succeeds only while this directory is empty.
            [IO.Directory]::Delete($targetRoot,$false)
        }
        if ((Split-Path -Parent $stageRoot) -ne $parent -or (Split-Path -Parent $targetRoot) -ne $parent) { throw '安装目录校验失败。' }
        [IO.Directory]::Move($stageRoot,$targetRoot)
    } catch {
        Write-Warning "安装没有提交，已有程序保持不变。临时文件保留在：$stageRoot"
        throw
    }
    if (-not $SkipShortcut) {
        try {
            $desktop = [Environment]::GetFolderPath('Desktop')
            $shortcut = Join-Path $desktop "Prism Interview $script:PrismVersion.lnk"
            Assert-PrismPlainPath $shortcut
            if (-not (Test-Path -LiteralPath $shortcut)) {
                $shellObject = New-Object -ComObject WScript.Shell
                $link = $shellObject.CreateShortcut($shortcut)
                $link.TargetPath = Join-Path $targetRoot 'PrismInterview.exe'
                $link.WorkingDirectory = $targetRoot
                $link.IconLocation = $link.TargetPath + ',0'
                $link.Save()
            } else { Write-Host '已保留现有同名桌面入口；可从新程序位置打开。' }
        } catch { Write-Warning '程序已安装，但桌面入口创建失败；请从安装目录打开。' }
    }
    Write-Host '安装完成。程序不会自动运行。'
    Write-Host "请打开：$(Join-Path $targetRoot 'PrismInterview.exe')"
    Write-Host '首次使用请在软件里配置自己的 API Key，并点击“下载／检查语音模型”。'
    return $targetRoot
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        $OutputEncoding = [Text.UTF8Encoding]::new($false)
        try { [Console]::OutputEncoding = $OutputEncoding } catch {}
        $null = Invoke-PrismInstall @PSBoundParameters
        exit 0
    } catch {
        Write-Error $_ -ErrorAction Continue
        exit 1
    }
}
