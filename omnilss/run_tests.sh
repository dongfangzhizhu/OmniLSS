#!/bin/bash
# OmniLSS 测试运行脚本
# 用法: ./run_tests.sh [选项]

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

resolve_python() {
    if [ -x "$PWD/../.venv/Scripts/python.exe" ]; then
        echo "$PWD/../.venv/Scripts/python.exe"
        return 0
    fi
    if [ -x "$PWD/.venv/bin/python" ]; then
        echo "$PWD/.venv/bin/python"
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    return 1
}

# 显示帮助信息
show_help() {
    echo "OmniLSS 测试运行脚本"
    echo ""
    echo "用法: ./run_tests.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --all                 运行 all 套件（默认）"
    echo "  --unit                运行 unit_core 套件"
    echo "  --consistency         运行 consistency_all 套件（需要 R）"
    echo "  --smoothers           运行 unit_smoothers 套件"
    echo "  --quick               运行 quick 套件"
    echo "  --suite <name>        指定一个或多个套件，可重复传入"
    echo "  --module <module>     指定单个测试模块，可重复传入"
    echo "  --list-suites         列出所有可用套件"
    echo "  --failfast            首个失败后停止"
    echo "  --verbose             详细输出"
    echo "  --help                显示帮助信息"
    echo ""
    echo "示例:"
    echo "  ./run_tests.sh                            # 运行 all 套件"
    echo "  ./run_tests.sh --quick"
    echo "  ./run_tests.sh --suite consistency_advanced_fit"
    echo "  ./run_tests.sh --suite family_batches --suite unit_smoothers"
    echo "  ./run_tests.sh --module tests.test_r_consistency_zip --verbose"
    echo "  ./run_tests.sh --list-suites"
}

# 检查环境
check_environment() {
    print_info "检查环境..."
    
    PYTHON_EXE="$(resolve_python)" || {
        print_error "未找到可用 Python，请先在工作区根目录准备共享 .venv"
        exit 1
    }
    print_info "使用 Python: $PYTHON_EXE"
    
    # 检查 R（仅一致性测试需要）
    if [ "$NEEDS_R" = true ]; then
        if ! command -v Rscript &> /dev/null; then
            print_error "Rscript 未找到，一致性测试需要 R"
            exit 1
        fi
        print_info "R 版本: $(Rscript --version 2>&1 | head -n 1)"
    fi
    
    print_success "环境检查完成"
}

# 设置环境变量
setup_env() {
    export PYTHONPATH="$PWD/src"
    export JAX_PLATFORMS="cpu"
    export JAX_ENABLE_X64="true"
    print_info "PYTHONPATH=$PYTHONPATH"
    print_info "JAX_PLATFORMS=$JAX_PLATFORMS"
    print_info "JAX_ENABLE_X64=$JAX_ENABLE_X64"
}

run_named_tests() {
    local args=("-m" "tests.run_suite")
    for suite_name in "${SELECTED_SUITES[@]}"; do
        args+=("--suite" "$suite_name")
    done
    for module_name in "${SELECTED_MODULES[@]}"; do
        args+=("--module" "$module_name")
    done
    if [ -n "$VERBOSE_FLAG" ]; then
        args+=("$VERBOSE_FLAG")
    fi
    if [ "$FAIL_FAST" = true ]; then
        args+=("--failfast")
    fi

    print_info "执行: python ${args[*]}"
    "$PYTHON_EXE" "${args[@]}"
    return $?
}

# 主函数
main() {
    # 解析命令行参数
    RUN_ALL=false
    RUN_UNIT=false
    RUN_CONSISTENCY=false
    RUN_SMOOTHERS=false
    QUICK_MODE=false
    LIST_SUITES=false
    FAIL_FAST=false
    VERBOSE_FLAG=""
    SELECTED_SUITES=()
    SELECTED_MODULES=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                RUN_ALL=true
                shift
                ;;
            --unit)
                RUN_UNIT=true
                shift
                ;;
            --consistency)
                RUN_CONSISTENCY=true
                shift
                ;;
            --smoothers)
                RUN_SMOOTHERS=true
                shift
                ;;
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --suite)
                SELECTED_SUITES+=("$2")
                shift 2
                ;;
            --module)
                SELECTED_MODULES+=("$2")
                shift 2
                ;;
            --list-suites)
                LIST_SUITES=true
                shift
                ;;
            --failfast)
                FAIL_FAST=true
                shift
                ;;
            --verbose)
                VERBOSE_FLAG="-v"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done

    if [ "$LIST_SUITES" = true ]; then
        PYTHON_EXE="$(resolve_python)" || {
            print_error "未找到可用 Python，请先在工作区根目录准备共享 .venv"
            exit 1
        }
        "$PYTHON_EXE" -m tests.run_suite --list
        exit $?
    fi

    if [ "$RUN_UNIT" = true ]; then
        SELECTED_SUITES+=("unit_core")
    fi
    if [ "$RUN_CONSISTENCY" = true ]; then
        SELECTED_SUITES+=("consistency_all")
    fi
    if [ "$RUN_SMOOTHERS" = true ]; then
        SELECTED_SUITES+=("unit_smoothers")
    fi
    if [ "$QUICK_MODE" = true ]; then
        SELECTED_SUITES+=("quick")
    fi
    if [ "$RUN_ALL" = true ] || [ ${#SELECTED_SUITES[@]} -eq 0 -a ${#SELECTED_MODULES[@]} -eq 0 ]; then
        SELECTED_SUITES+=("all")
    fi

    NEEDS_R=false
    for suite_name in "${SELECTED_SUITES[@]}"; do
        if [[ "$suite_name" == consistency* ]] || [[ "$suite_name" == "all" ]]; then
            NEEDS_R=true
            break
        fi
    done
    if [ "$NEEDS_R" = false ]; then
        for module_name in "${SELECTED_MODULES[@]}"; do
            if [[ "$module_name" == tests.test_consistency_* ]] || [[ "$module_name" == tests.test_r_consistency_* ]]; then
                NEEDS_R=true
                break
            fi
        done
    fi
    
    # 打印横幅
    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║   OmniLSS 测试套件                 ║"
    echo "╚════════════════════════════════════════╝"
    echo ""
    
    # 检查环境
    check_environment
    
    # 设置环境变量
    setup_env
    
    echo ""
    
    if [ ${#SELECTED_SUITES[@]} -gt 0 ]; then
        print_info "套件: ${SELECTED_SUITES[*]}"
    fi
    if [ ${#SELECTED_MODULES[@]} -gt 0 ]; then
        print_info "模块: ${SELECTED_MODULES[*]}"
    fi

    # 运行测试
    local exit_code=0
    run_named_tests || exit_code=$?
    
    echo ""
    
    # 打印总结
    if [ $exit_code -eq 0 ]; then
        print_success "╔════════════════════════════════════════╗"
        print_success "║   所选测试通过！ ✓                    ║"
        print_success "╚════════════════════════════════════════╝"
    else
        print_error "╔════════════════════════════════════════╗"
        print_error "║   所选测试失败 ✗                      ║"
        print_error "╚════════════════════════════════════════╝"
    fi
    
    echo ""
    
    exit $exit_code
}

# 运行主函数
main "$@"
