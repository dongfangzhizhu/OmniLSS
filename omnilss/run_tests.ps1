# OmniLSS 测试运行脚本 (PowerShell)
# 用法: .\run_tests.ps1 [选项]

param(
    [switch]$All = $false,
    [switch]$Unit = $false,
    [switch]$Consistency = $false,
    [switch]$Smoothers = $false,
    [switch]$Quick = $false,
    [string[]]$Suite = @(),
    [string[]]$Module = @(),
    [switch]$ListSuites = $false,
    [switch]$FailFast = $false,
    [switch]$Verbose = $false,
    [switch]$Help = $false
)

# 错误时停止
$ErrorActionPreference = "Stop"

# 打印带颜色的消息
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Get-WorkspacePython {
    $sharedPython = Join-Path $PSScriptRoot "..\\.venv\\Scripts\\python.exe"
    if (Test-Path $sharedPython) {
        return $sharedPython
    }

    $localPython = Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe"
    if (Test-Path $localPython) {
        return $localPython
    }

    return $null
}

# 显示帮助信息
function Show-Help {
    Write-Host "OmniLSS 测试运行脚本"
    Write-Host ""
    Write-Host "用法: .\run_tests.ps1 [选项]"
    Write-Host ""
    Write-Host "选项:"
    Write-Host "  -All                 运行 all 套件（默认）"
    Write-Host "  -Unit                运行 unit_core 套件"
    Write-Host "  -Consistency         运行 consistency_all 套件（需要 R）"
    Write-Host "  -Smoothers           运行 unit_smoothers 套件"
    Write-Host "  -Quick               运行 quick 套件"
    Write-Host "  -Suite <name>        指定一个或多个套件，可重复传入"
    Write-Host "  -Module <module>     指定单个测试模块，可重复传入"
    Write-Host "  -ListSuites          列出所有可用套件"
    Write-Host "  -FailFast            首个失败后停止"
    Write-Host "  -Verbose             详细输出"
    Write-Host "  -Help                显示帮助信息"
    Write-Host ""
    Write-Host "示例:"
    Write-Host "  .\run_tests.ps1                            # 运行 all 套件"
    Write-Host "  .\run_tests.ps1 -Quick                     # 运行 quick 套件"
    Write-Host "  .\run_tests.ps1 -Suite consistency_advanced_fit"
    Write-Host "  .\run_tests.ps1 -Suite family_batches -Suite unit_smoothers"
    Write-Host "  .\run_tests.ps1 -Module tests.test_r_consistency_zip -Verbose"
    Write-Host "  .\run_tests.ps1 -ListSuites"
}

# 检查环境
function Test-Environment {
    Write-Info "检查环境..."
    
    $script:PythonExe = Get-WorkspacePython
    if (-not $script:PythonExe) {
        Write-Error "未找到可用 Python，请先在工作区根目录准备共享 .venv"
        exit 1
    }

    Write-Info "使用 Python: $script:PythonExe"
    
    # 检查 R（仅一致性测试需要）
    if ($script:NeedsR) {
        if (-not (Get-Command Rscript -ErrorAction SilentlyContinue)) {
            Write-Error "Rscript 未找到，一致性测试需要 R"
            exit 1
        }
        $rVersion = & Rscript --version 2>&1 | Select-Object -First 1
        Write-Info "R 版本: $rVersion"
    }
    
    Write-Success "环境检查完成"
}

# 设置环境变量
function Set-TestEnvironment {
    $env:PYTHONPATH = (Join-Path $PSScriptRoot "src")
    $env:JAX_PLATFORMS = "cpu"
    $env:JAX_ENABLE_X64 = "true"
    Write-Info "PYTHONPATH=$env:PYTHONPATH"
    Write-Info "JAX_PLATFORMS=$env:JAX_PLATFORMS"
    Write-Info "JAX_ENABLE_X64=$env:JAX_ENABLE_X64"
}

# 运行套件
function Invoke-TestRunner {
    param(
        [string[]]$Suites,
        [string[]]$Modules
    )

    $args = @("-m", "tests.run_suite")

    foreach ($suiteName in $Suites) {
        $args += @("--suite", $suiteName)
    }
    foreach ($moduleName in $Modules) {
        $args += @("--module", $moduleName)
    }
    if ($script:VerboseOutput) {
        $args += "--verbose"
    }
    if ($script:FailFastMode) {
        $args += "--failfast"
    }

    Write-Info ("执行: python {0}" -f ($args -join " "))

    try {
        & $script:PythonExe @args | Out-Host
        return [int]$LASTEXITCODE
    } catch {
        Write-Error "测试运行失败: $_"
        return 1
    }
}

# 主函数
function Main {
    # 显示帮助
    if ($Help) {
        Show-Help
        exit 0
    }
    
    # 设置默认值
    $script:VerboseOutput = $Verbose
    $script:FailFastMode = $FailFast
    $selectedSuites = @()
    $selectedModules = @($Module)

    if ($ListSuites) {
        $script:PythonExe = Get-WorkspacePython
        if (-not $script:PythonExe) {
            Write-Error "未找到可用 Python，请先在工作区根目录准备共享 .venv"
            exit 1
        }
        & $script:PythonExe -m tests.run_suite --list
        exit $LASTEXITCODE
    }

    if ($Suite.Count -gt 0) {
        $selectedSuites += $Suite
    }
    if ($Unit) {
        $selectedSuites += "unit_core"
    }
    if ($Consistency) {
        $selectedSuites += "consistency_all"
    }
    if ($Smoothers) {
        $selectedSuites += "unit_smoothers"
    }
    if ($Quick) {
        $selectedSuites += "quick"
    }
    if ($All -or ($selectedSuites.Count -eq 0 -and $selectedModules.Count -eq 0)) {
        $selectedSuites += "all"
    }

    $selectedSuites = @($selectedSuites | Select-Object -Unique)
    $selectedModules = @($selectedModules | Select-Object -Unique)
    $selectionText = @()
    if ($selectedSuites.Count -gt 0) {
        $selectionText += "套件: $($selectedSuites -join ', ')"
    }
    if ($selectedModules.Count -gt 0) {
        $selectionText += "模块: $($selectedModules -join ', ')"
    }

    $script:NeedsR = ($selectedSuites -match "^consistency") -or ($selectedSuites -contains "all")
    if (-not $script:NeedsR) {
        foreach ($moduleName in $selectedModules) {
            if ($moduleName -like "tests.test_consistency_*" -or $moduleName -like "tests.test_r_consistency_*") {
                $script:NeedsR = $true
                break
            }
        }
    }
    
    # 打印横幅
    Write-Host ""
    Write-Host "╔════════════════════════════════════════╗"
    Write-Host "║   OmniLSS 测试套件                 ║"
    Write-Host "╚════════════════════════════════════════╝"
    Write-Host ""
    
    # 检查环境
    Test-Environment
    
    # 设置环境变量
    Set-TestEnvironment
    
    Write-Host ""
    
    Write-Info ($selectionText -join " | ")
    $exitCode = Invoke-TestRunner -Suites $selectedSuites -Modules $selectedModules
    
    Write-Host ""
    
    # 打印总结
    if ($exitCode -eq 0) {
        Write-Success "╔════════════════════════════════════════╗"
        Write-Success "║   所选测试通过！ ✓                    ║"
        Write-Success "╚════════════════════════════════════════╝"
    } else {
        Write-Error "╔════════════════════════════════════════╗"
        Write-Error "║   所选测试失败 ✗                      ║"
        Write-Error "╚════════════════════════════════════════╝"
    }
    
    Write-Host ""
    
    exit $exitCode
}

# 运行主函数
Main
