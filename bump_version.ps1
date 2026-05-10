#!/usr/bin/env pwsh
# OmniLSS 版本号更新脚本
# 用法：
#   .\bump_version.ps1 0.2.1        # 设置为指定版本
#   .\bump_version.ps1 -Patch       # 增加补丁版本 (0.2.0 -> 0.2.1)
#   .\bump_version.ps1 -Minor       # 增加次要版本 (0.2.0 -> 0.3.0)
#   .\bump_version.ps1 -Major       # 增加主要版本 (0.2.0 -> 1.0.0)

param(
    [Parameter(Position=0)]
    [string]$NewVersion = "",
    [switch]$Patch = $false,
    [switch]$Minor = $false,
    [switch]$Major = $false,
    [switch]$DryRun = $false,
    [switch]$Help = $false
)

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message); Write-ColorOutput "✓ $Message" "Green" }
function Write-Error-Custom { param([string]$Message); Write-ColorOutput "✗ $Message" "Red" }
function Write-Warning-Custom { param([string]$Message); Write-ColorOutput "⚠ $Message" "Yellow" }
function Write-Info { param([string]$Message); Write-ColorOutput "ℹ $Message" "Cyan" }

if ($Help) {
    Write-ColorOutput @"

OmniLSS 版本号更新脚本

用法:
    .\bump_version.ps1 <新版本号>    # 设置为指定版本
    .\bump_version.ps1 -Patch        # 增加补丁版本 (0.2.0 -> 0.2.1)
    .\bump_version.ps1 -Minor        # 增加次要版本 (0.2.0 -> 0.3.0)
    .\bump_version.ps1 -Major        # 增加主要版本 (0.2.0 -> 1.0.0)

选项:
    -DryRun    仅显示将要进行的更改，不实际修改文件
    -Help      显示此帮助信息

示例:
    # 设置为特定版本
    .\bump_version.ps1 0.2.1

    # 增加补丁版本
    .\bump_version.ps1 -Patch

    # 预览更改（不实际修改）
    .\bump_version.ps1 -Patch -DryRun

"@ "White"
    exit 0
}

# 检查是否在项目根目录
if (-not (Test-Path "omnilss\pyproject.toml")) {
    Write-Error-Custom "请在项目根目录运行此脚本（应包含 omnilss\pyproject.toml）"
    exit 1
}

# 读取当前版本
$pyprojectPath = "omnilss\pyproject.toml"
$pyprojectContent = Get-Content $pyprojectPath -Raw

if ($pyprojectContent -match 'version\s*=\s*"([^"]+)"') {
    $currentVersion = $matches[1]
    Write-Info "当前版本: $currentVersion"
} else {
    Write-Error-Custom "无法从 pyproject.toml 读取版本号"
    exit 1
}

# 解析当前版本
if ($currentVersion -match '^(\d+)\.(\d+)\.(\d+)$') {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    $patch = [int]$matches[3]
} else {
    Write-Error-Custom "版本号格式不正确: $currentVersion（应为 x.y.z 格式）"
    exit 1
}

# 确定新版本号
if ($NewVersion) {
    # 验证版本号格式
    if ($NewVersion -notmatch '^\d+\.\d+\.\d+$') {
        Write-Error-Custom "版本号格式不正确: $NewVersion（应为 x.y.z 格式）"
        exit 1
    }
    $targetVersion = $NewVersion
} elseif ($Patch) {
    $targetVersion = "$major.$minor.$($patch + 1)"
} elseif ($Minor) {
    $targetVersion = "$major.$($minor + 1).0"
} elseif ($Major) {
    $targetVersion = "$($major + 1).0.0"
} else {
    Write-Error-Custom "请指定新版本号或使用 -Patch/-Minor/-Major 选项"
    Write-Info "运行 .\bump_version.ps1 -Help 查看帮助"
    exit 1
}

Write-Success "目标版本: $targetVersion"

if ($DryRun) {
    Write-Warning-Custom "`n[预览模式] 将要进行的更改:"
} else {
    Write-Info "`n将要进行的更改:"
}

# 更新 pyproject.toml
Write-Info "  1. omnilss\pyproject.toml: $currentVersion -> $targetVersion"

if (-not $DryRun) {
    $newContent = $pyprojectContent -replace 'version\s*=\s*"[^"]+"', "version = `"$targetVersion`""
    Set-Content -Path $pyprojectPath -Value $newContent -NoNewline
    Write-Success "     已更新 pyproject.toml"
}

# 检查并更新 CHANGELOG.md
$changelogPath = "CHANGELOG.md"
if (Test-Path $changelogPath) {
    Write-Info "  2. CHANGELOG.md: 添加新版本条目"
    
    if (-not $DryRun) {
        $changelogContent = Get-Content $changelogPath -Raw
        $date = Get-Date -Format "yyyy-MM-dd"
        
        # 在 ## [Unreleased] 后添加新版本
        if ($changelogContent -match '## \[Unreleased\]') {
            $newEntry = @"

## [$targetVersion] - $date

### Added
- 

### Changed
- 

### Fixed
- 

"@
            $newContent = $changelogContent -replace '(## \[Unreleased\])', "`$1$newEntry"
            Set-Content -Path $changelogPath -Value $newContent -NoNewline
            Write-Success "     已更新 CHANGELOG.md"
            Write-Warning-Custom "     请手动编辑 CHANGELOG.md 添加更新内容"
        } else {
            Write-Warning-Custom "     CHANGELOG.md 格式不标准，请手动更新"
        }
    }
} else {
    Write-Warning-Custom "  2. CHANGELOG.md 不存在，跳过"
}

# 提示后续步骤
if (-not $DryRun) {
    Write-Success "`n版本号已更新为 $targetVersion"
    Write-Info "`n后续步骤:"
    Write-Info "  1. 编辑 CHANGELOG.md，添加此版本的更新内容"
    Write-Info "  2. 提交更改:"
    Write-Info "     git add omnilss/pyproject.toml CHANGELOG.md"
    Write-Info "     git commit -m 'Bump version to $targetVersion'"
    Write-Info "  3. 创建 Git 标签:"
    Write-Info "     git tag v$targetVersion"
    Write-Info "     git push origin v$targetVersion"
    Write-Info "  4. 发布到 PyPI:"
    Write-Info "     .\publish.ps1 -TestPyPI  # 先测试"
    Write-Info "     .\publish.ps1            # 正式发布"
} else {
    Write-Info "`n这是预览模式，没有实际修改文件"
    Write-Info "移除 -DryRun 参数以实际执行更改"
}

Write-ColorOutput ""
