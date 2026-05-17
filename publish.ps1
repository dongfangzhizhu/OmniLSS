#!/usr/bin/env pwsh
# OmniLSS PyPI 发布脚本
# 用法：
#   .\publish.ps1                    # 发布到 PyPI（需要 token）
#   .\publish.ps1 -TestPyPI          # 发布到 Test PyPI（测试用）
#   .\publish.ps1 -SkipTests         # 跳过测试直接构建
#   .\publish.ps1 -Clean             # 仅清理 dist 目录

param(
    [switch]$TestPyPI = $false,      # 发布到 Test PyPI 而不是正式 PyPI
    [switch]$SkipTests = $false,     # 跳过测试
    [switch]$Clean = $false,         # 仅清理
    [switch]$BuildOnly = $false,     # 仅构建不发布
    [switch]$CheckVersion = $false,  # 仅检查版本是否存在
    [switch]$Help = $false           # 显示帮助
)

# 颜色输出函数
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message); Write-ColorOutput "✓ $Message" "Green" }
function Write-Error-Custom { param([string]$Message); Write-ColorOutput "✗ $Message" "Red" }
function Write-Warning-Custom { param([string]$Message); Write-ColorOutput "⚠ $Message" "Yellow" }
function Write-Info { param([string]$Message); Write-ColorOutput "ℹ $Message" "Cyan" }
function Write-Step { param([string]$Message); Write-ColorOutput "`n▶ $Message" "Magenta" }

# 显示帮助
if ($Help) {
    Write-ColorOutput @"

OmniLSS PyPI 发布脚本

用法:
    .\publish.ps1 [选项]

选项:
    -TestPyPI      发布到 Test PyPI (https://test.pypi.org)
    -SkipTests     跳过测试直接构建
    -BuildOnly     仅构建 wheel，不发布
    -Clean         仅清理 dist 目录
    -CheckVersion  检查版本是否已在 PyPI 上存在
    -Help          显示此帮助信息

示例:
    # 发布到正式 PyPI（交互式输入 token）
    .\publish.ps1

    # 先在 Test PyPI 上测试
    .\publish.ps1 -TestPyPI

    # 检查版本是否已存在
    .\publish.ps1 -CheckVersion

    # 仅构建 wheel 检查
    .\publish.ps1 -BuildOnly

    # 清理旧的构建文件
    .\publish.ps1 -Clean

环境变量:
    PYPI_TOKEN         PyPI API token (可选，否则提示输入)
    TEST_PYPI_TOKEN    Test PyPI API token (可选)

注意:
    - 首次发布需要在 PyPI 注册账号并创建项目
    - 建议先用 -TestPyPI 测试发布流程
    - 确保 CHANGELOG.md 和 pyproject.toml 版本号一致

"@ "White"
    exit 0
}

# 开始
Write-ColorOutput "`n═══════════════════════════════════════════════════════════════" "Cyan"
Write-ColorOutput "  OmniLSS PyPI 发布脚本" "Cyan"
Write-ColorOutput "═══════════════════════════════════════════════════════════════`n" "Cyan"

# 检查是否在项目根目录
if (-not (Test-Path "omnilss\pyproject.toml")) {
    Write-Error-Custom "请在项目根目录运行此脚本（应包含 omnilss\pyproject.toml）"
    exit 1
}

# 切换到 omnilss 目录
Push-Location omnilss

try {
    # 步骤 1: 清理旧的构建文件
    Write-Step "步骤 1/6: 清理旧的构建文件"

    @("dist", "build", "src\omnilss.egg-info") | ForEach-Object {
        if (Test-Path $_) {
            Remove-Item -Recurse -Force $_
            Write-Success "已删除 $_"
        }
    }

    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    Write-Success "已清理所有 __pycache__ 目录"

    if ($Clean) {
        Write-Success "`n清理完成！"
        Pop-Location
        exit 0
    }

    # 步骤 2: 读取版本号
    Write-Step "步骤 2/6: 读取版本信息"

    $pyprojectContent = Get-Content "pyproject.toml" -Raw
    if ($pyprojectContent -match 'version\s*=\s*"([^"]+)"') {
        $version = $matches[1]
        Write-Success "当前版本: v$version"
    } else {
        Write-Error-Custom "无法从 pyproject.toml 读取版本号"
        Pop-Location
        exit 1
    }

    # 检查版本是否已在 PyPI 上存在
    Write-Info "检查版本是否已存在..."
    
    if ($TestPyPI) {
        $pypiUrl = "https://test.pypi.org/pypi/omnilss/$version/json"
        $pypiName = "Test PyPI"
    } else {
        $pypiUrl = "https://pypi.org/pypi/omnilss/$version/json"
        $pypiName = "PyPI"
    }
    
    $versionExists = $false
    try {
        $response = Invoke-WebRequest -Uri $pypiUrl -Method Head -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $versionExists = $true
            Write-Warning-Custom "版本 $version 已存在于 $pypiName"
            
            if ($CheckVersion) {
                Write-Info "`n版本检查结果: 版本已存在"
                Write-Info "查看已发布版本: $(if ($TestPyPI) {'https://test.pypi.org/project/omnilss/'} else {'https://pypi.org/project/omnilss/'})"
                Pop-Location
                exit 0
            }
            
            Write-Info "PyPI 不允许覆盖已发布的版本"
            Write-Info "`n请更新版本号:"
            Write-Info "  1. 编辑 omnilss/pyproject.toml，修改 version 字段"
            
            # 建议新版本号
            if ($version -match '^(\d+)\.(\d+)\.(\d+)$') {
                $major = [int]$matches[1]
                $minor = [int]$matches[2]
                $patch = [int]$matches[3]
                Write-Info "     当前: $version"
                Write-Info "     建议: $major.$minor.$($patch + 1) (补丁版本)"
                Write-Info "     或:   $major.$($minor + 1).0 (次要版本)"
            }
            
            Write-Info "  2. 更新 CHANGELOG.md"
            Write-Info "  3. 提交更改: git commit -am 'Bump version to x.x.x'"
            Write-Info "  4. 创建标签: git tag vx.x.x"
            Write-Info "  5. 重新运行发布脚本"
            
            $continue = Read-Host "`n是否仍要继续（将会失败）？(y/N)"
            if ($continue -ne "y" -and $continue -ne "Y") {
                Write-Info "已取消发布"
                Pop-Location
                exit 0
            }
        }
    } catch {
        Write-Success "版本 $version 尚未发布（可以继续）"
        
        if ($CheckVersion) {
            Write-Info "`n版本检查结果: 版本可用"
            Write-Info "可以发布版本 $version 到 $pypiName"
            Pop-Location
            exit 0
        }
    }

    # 检查 Git 标签
    Push-Location ..
    $gitTag = git describe --tags --exact-match 2>$null
    Pop-Location

    if ($gitTag -eq "v$version") {
        Write-Success "Git 标签匹配: $gitTag"
    } else {
        Write-Warning-Custom "Git 标签不匹配或不存在 (当前版本: v$version, Git 标签: $gitTag)"
        $continue = Read-Host "是否继续？(y/N)"
        if ($continue -ne "y" -and $continue -ne "Y") {
            Write-Info "已取消发布"
            Pop-Location
            exit 0
        }
    }

    # 步骤 3: 运行测试（可选）
    if (-not $SkipTests) {
        Write-Step "步骤 3/6: 运行核心测试"
        Write-Info "运行快速测试套件..."

        $testResult = python -m pytest tests/test_rs_algorithm.py tests/test_cg_algorithm.py tests/test_mixed_algorithm.py tests/test_gcv.py tests/test_reml.py -q --tb=short 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Success "测试通过！"
        } else {
            Write-Error-Custom "测试失败！"
            Write-Info $testResult
            $continue = Read-Host "是否仍要继续构建？(y/N)"
            if ($continue -ne "y" -and $continue -ne "Y") {
                Pop-Location
                exit 1
            }
        }
    } else {
        Write-Step "步骤 3/6: 跳过测试 (-SkipTests)"
    }

    # 步骤 4: 检查并安装构建工具
    Write-Step "步骤 4/6: 检查构建工具"

    $buildInstalled = python -m pip show build 2>$null
    if (-not $buildInstalled) {
        Write-Info "安装 build 工具..."
        python -m pip install --upgrade build
    }
    Write-Success "build 工具已就绪"

    $twineInstalled = python -m pip show twine 2>$null
    if (-not $twineInstalled) {
        Write-Info "安装 twine 工具..."
        python -m pip install --upgrade twine
    }
    Write-Success "twine 工具已就绪"

    # 步骤 5: 构建分发包
    Write-Step "步骤 5/6: 构建分发包"

    Write-Info "构建 wheel 和 sdist..."
    python -m build

    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "构建失败！"
        Pop-Location
        exit 1
    }

    Write-Success "构建完成！"

    # 显示生成的文件
    $distFiles = Get-ChildItem dist
    Write-Info "生成的文件:"
    $distFiles | ForEach-Object { Write-Info "  - $($_.Name)" }

    # 检查包
    Write-Info "`n检查包完整性..."
    python -m twine check dist/*

    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "包检查失败！请修复后重试"
        Pop-Location
        exit 1
    }

    Write-Success "包检查通过！"

    if ($BuildOnly) {
        Write-Success "`n构建完成！包已保存在 omnilss\dist\ 目录"
        Write-Info "使用以下命令手动发布:"
        if ($TestPyPI) {
            Write-Info "  python -m twine upload --repository testpypi dist/*"
        } else {
            Write-Info "  python -m twine upload dist/*"
        }
        Pop-Location
        exit 0
    }

    # 步骤 6: 发布到 PyPI
    Write-Step "步骤 6/6: 发布到 PyPI"

    if ($TestPyPI) {
        Write-Warning-Custom "将发布到 Test PyPI (https://test.pypi.org)"
        $repository = "testpypi"
        $tokenEnvVar = "TEST_PYPI_TOKEN"
    } else {
        Write-Warning-Custom "将发布到正式 PyPI (https://pypi.org)"
        $repository = "pypi"
        $tokenEnvVar = "PYPI_TOKEN"
    }

    # 获取 token
    $token = [System.Environment]::GetEnvironmentVariable($tokenEnvVar)

    if (-not $token) {
        Write-Info "未找到环境变量 $tokenEnvVar"
        Write-Info "请输入 PyPI API token (以 'pypi-' 开头):"
        $secureToken = Read-Host -AsSecureString
        $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
        $token = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)
    }

    if (-not $token) {
        Write-Error-Custom "未提供 token，发布已取消"
        Pop-Location
        exit 1
    }

    # 最后确认
    Write-Warning-Custom "`n准备发布:"
    Write-Info "  版本: v$version"
    Write-Info "  目标: $repository"
    Write-Info "  文件: $($distFiles.Count) 个"

    $confirm = Read-Host "`n确认发布？(yes/NO)"
    if ($confirm -ne "yes") {
        Write-Info "已取消发布"
        Pop-Location
        exit 0
    }

    # 执行发布
    Write-Info "`n开始上传..."

    # 使用 --verbose 获取详细错误信息
    if ($TestPyPI) {
        $uploadOutput = python -m twine upload --repository testpypi dist/* --username __token__ --password $token --verbose 2>&1
    } else {
        $uploadOutput = python -m twine upload dist/* --username __token__ --password $token --verbose 2>&1
    }

    $uploadExitCode = $LASTEXITCODE

    if ($uploadExitCode -eq 0) {
        Write-Success "`n发布成功！ 🎉"
        Write-Info "`n安装测试:"
        if ($TestPyPI) {
            Write-Info "  pip install --index-url https://test.pypi.org/simple/ omnilss==$version"
        } else {
            Write-Info "  pip install omnilss==$version"
        }
        Write-Info "`n查看项目:"
        if ($TestPyPI) {
            Write-Info "  https://test.pypi.org/project/omnilss/$version/"
        } else {
            Write-Info "  https://pypi.org/project/omnilss/$version/"
        }
    } else {
        Write-Error-Custom "`n发布失败！"
        Write-Info "`n详细错误信息:"
        Write-Host $uploadOutput
        
        # 分析常见错误
        $outputStr = $uploadOutput | Out-String
        
        if ($outputStr -match "400.*Bad Request") {
            Write-Warning-Custom "`n可能的原因:"
            Write-Info "  1. 版本 $version 已存在于 PyPI（无法覆盖已发布的版本）"
            Write-Info "  2. 包名或元数据格式有问题"
            Write-Info "  3. README.md 包含不支持的内容"
            Write-Info "`n解决方案:"
            Write-Info "  - 如果版本已存在，请在 pyproject.toml 中更新版本号"
            Write-Info "  - 检查 pyproject.toml 中的元数据格式"
            Write-Info "  - 运行: python -m twine check dist/*"
        } elseif ($outputStr -match "403.*Forbidden") {
            Write-Warning-Custom "`n可能的原因:"
            Write-Info "  1. API token 无效或已过期"
            Write-Info "  2. 没有权限发布此包"
            Write-Info "  3. 包名已被其他用户占用"
            Write-Info "`n解决方案:"
            Write-Info "  - 检查 API token 是否正确"
            Write-Info "  - 确认在 PyPI 上有此包的发布权限"
            Write-Info "  - 如果是新包，确保包名未被占用"
        } elseif ($outputStr -match "401.*Unauthorized") {
            Write-Warning-Custom "`n可能的原因:"
            Write-Info "  1. API token 格式错误"
            Write-Info "  2. 使用了错误的 token（PyPI vs Test PyPI）"
            Write-Info "`n解决方案:"
            Write-Info "  - 确保 token 以 'pypi-' 开头"
            Write-Info "  - 确认使用了正确的 token（Test PyPI 和 PyPI 的 token 不同）"
        }
        
        Write-Info "`n手动上传命令:"
        if ($TestPyPI) {
            Write-Info "  python -m twine upload --repository testpypi dist/* --verbose"
        } else {
            Write-Info "  python -m twine upload dist/* --verbose"
        }
        
        Pop-Location
        exit 1
    }

} catch {
    Write-Error-Custom "发生错误: $_"
    Pop-Location
    exit 1
} finally {
    Pop-Location
}

Write-ColorOutput "`n═══════════════════════════════════════════════════════════════" "Cyan"
Write-Success "完成！"
Write-ColorOutput "═══════════════════════════════════════════════════════════════`n" "Cyan"
