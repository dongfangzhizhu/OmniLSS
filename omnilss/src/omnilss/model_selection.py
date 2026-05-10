"""模型选择和分布比较

实现自动分布选择功能，帮助用户找到最适合数据的分布族。

这是 GAMLSS 的强大功能之一：可以自动比较多个分布族，
选择最佳的分布来建模数据。

支持并行化以加速多个分布的比较。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os


def _fit_single_distribution(
    family_name: str,
    formula: str,
    data: dict,
    sigma_formula: str = "~1",
    parameter_formulas: Optional[Dict[str, str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """拟合单个分布（用于并行化）
    
    这是一个独立的函数，可以被多进程调用。
    
    Parameters
    ----------
    family_name : str
        分布族名称
    formula : str
        mu 参数的模型公式
    data : dict
        数据字典
    sigma_formula : str
        sigma 参数的公式
    parameter_formulas : dict, optional
        其他参数的公式
    **kwargs
        传递给 gamlss() 的其他参数
    
    Returns
    -------
    result : dict
        拟合结果字典
    """
    from .fitting import gamlss
    
    try:
        # 拟合模型
        model = gamlss(
            formula,
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas,
            family=family_name,
            data=data,
            **kwargs
        )
        
        # 提取信息
        result = {
            "family": family_name,
            "deviance": float(model.g_dev),
            "df": float(model.df_fit),
            "AIC": float(model.additional_slots.get("aic", np.inf)),
            "BIC": float(model.additional_slots.get("sbc", np.inf)),
            "converged": bool(model.additional_slots.get("converged", False)),
            "n_iter": int(model.iter),
            "success": True
        }
        
    except Exception as e:
        # 拟合失败
        result = {
            "family": family_name,
            "deviance": np.inf,
            "df": np.nan,
            "AIC": np.inf,
            "BIC": np.inf,
            "converged": False,
            "n_iter": 0,
            "error": str(e),
            "success": False
        }
    
    return result


def compare_distributions(
    formula: str,
    families: List[str],
    data: dict,
    sigma_formula: str = "~1",
    parameter_formulas: Optional[Dict[str, str]] = None,
    criterion: str = "AIC",
    verbose: bool = False,
    parallel: bool = False,
    n_jobs: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """比较多个分布族
    
    对同一个公式拟合多个分布族，并根据信息准则排序。
    
    Parameters
    ----------
    formula : str
        mu 参数的模型公式
    families : list of str
        要比较的分布族列表，如 ["NO", "GA", "LOGNO", "WEI"]
    data : dict
        数据字典 {变量名: 值数组}
    sigma_formula : str, default="~1"
        sigma 参数的公式
    parameter_formulas : dict, optional
        其他参数的公式
    criterion : str, default="AIC"
        排序准则: "AIC", "BIC", "deviance"
    verbose : bool, default=False
        是否打印拟合进度
    parallel : bool, default=False
        是否使用并行化（多进程）
    n_jobs : int, optional
        并行进程数，默认为 CPU 核心数
    **kwargs
        传递给 gamlss() 的其他参数
    
    Returns
    -------
    results : pd.DataFrame
        比较结果，按 criterion 排序
        列包括: family, deviance, df, AIC, BIC, converged
    
    Examples
    --------
    >>> import numpy as np
    >>> from omnilss.model_selection import compare_distributions
    >>> 
    >>> # 生成数据
    >>> np.random.seed(42)
    >>> n = 100
    >>> x = np.random.randn(n)
    >>> y = np.exp(2 + 0.5*x + np.random.randn(n)*0.3)  # Log-normal data
    >>> data = {"y": y, "x": x}
    >>> 
    >>> # 串行比较
    >>> results = compare_distributions(
    ...     "y ~ x",
    ...     families=["NO", "GA", "LOGNO", "WEI"],
    ...     data=data,
    ...     criterion="AIC"
    ... )
    >>> 
    >>> # 并行比较（更快）
    >>> results = compare_distributions(
    ...     "y ~ x",
    ...     families=["NO", "GA", "LOGNO", "WEI"],
    ...     data=data,
    ...     criterion="AIC",
    ...     parallel=True,
    ...     n_jobs=4
    ... )
    >>> 
    >>> # 查看结果
    >>> print(results)
    >>> print(f"Best family: {results.iloc[0]['family']}")
    
    Notes
    -----
    这个函数会尝试拟合所有指定的分布族。如果某个分布拟合失败，
    会记录错误信息但不会中断整个比较过程。
    
    信息准则越小越好：
    - AIC = deviance + 2 * df
    - BIC = deviance + log(n) * df
    
    并行化说明：
    - 当 parallel=True 时，使用多进程并行拟合分布
    - 对于大量分布（>4）或复杂模型，并行化可以显著加速
    - n_jobs 默认为 CPU 核心数，可以手动指定
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"比较 {len(families)} 个分布族")
        if parallel:
            n_jobs_actual = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
            print(f"使用并行化: {n_jobs_actual} 个进程")
        print(f"{'='*70}\n")
    
    if parallel:
        # 并行化执行
        results = _compare_distributions_parallel(
            formula, families, data, sigma_formula, parameter_formulas,
            verbose, n_jobs, **kwargs
        )
    else:
        # 串行执行
        results = _compare_distributions_serial(
            formula, families, data, sigma_formula, parameter_formulas,
            verbose, **kwargs
        )
    
    # 转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 排序
    if criterion in df.columns:
        df = df.sort_values(criterion, ascending=True).reset_index(drop=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print("比较完成")
        print(f"{'='*70}\n")
    
    return df


def _compare_distributions_serial(
    formula: str,
    families: List[str],
    data: dict,
    sigma_formula: str,
    parameter_formulas: Optional[Dict[str, str]],
    verbose: bool,
    **kwargs
) -> List[Dict[str, Any]]:
    """串行比较分布（原始实现）"""
    from .fitting import gamlss
    
    results = []
    
    for i, family_name in enumerate(families, 1):
        if verbose:
            print(f"[{i}/{len(families)}] 拟合 {family_name}...", end=" ")
        
        result = _fit_single_distribution(
            family_name, formula, data, sigma_formula,
            parameter_formulas, **kwargs
        )
        
        if verbose:
            if result["success"]:
                print(f"✓ (AIC={result['AIC']:.2f}, BIC={result['BIC']:.2f})")
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"✗ ({error_msg[:50]})")
        
        results.append(result)
    
    return results


def _compare_distributions_parallel(
    formula: str,
    families: List[str],
    data: dict,
    sigma_formula: str,
    parameter_formulas: Optional[Dict[str, str]],
    verbose: bool,
    n_jobs: Optional[int],
    **kwargs
) -> List[Dict[str, Any]]:
    """并行比较分布"""
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    results = []
    completed_count = 0
    total_count = len(families)
    
    # 使用 ProcessPoolExecutor 并行执行
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # 提交所有任务
        future_to_family = {
            executor.submit(
                _fit_single_distribution,
                family_name, formula, data, sigma_formula,
                parameter_formulas, **kwargs
            ): family_name
            for family_name in families
        }
        
        # 收集结果
        for future in as_completed(future_to_family):
            family_name = future_to_family[future]
            completed_count += 1
            
            try:
                result = future.result()
                
                if verbose:
                    if result["success"]:
                        print(f"[{completed_count}/{total_count}] {family_name}: ✓ (AIC={result['AIC']:.2f}, BIC={result['BIC']:.2f})")
                    else:
                        error_msg = result.get("error", "Unknown error")
                        print(f"[{completed_count}/{total_count}] {family_name}: ✗ ({error_msg[:50]})")
                
                results.append(result)
                
            except Exception as e:
                if verbose:
                    print(f"[{completed_count}/{total_count}] {family_name}: ✗ (Execution error: {str(e)[:50]})")
                
                results.append({
                    "family": family_name,
                    "deviance": np.inf,
                    "df": np.nan,
                    "AIC": np.inf,
                    "BIC": np.inf,
                    "converged": False,
                    "n_iter": 0,
                    "error": str(e),
                    "success": False
                })
    
    return results


def select_best_distribution(
    formula: str,
    families: List[str],
    data: dict,
    sigma_formula: str = "~1",
    parameter_formulas: Optional[Dict[str, str]] = None,
    criterion: str = "AIC",
    verbose: bool = False,
    **kwargs
) -> tuple:
    """自动选择最佳分布
    
    比较多个分布族，返回最佳模型和比较结果。
    
    Parameters
    ----------
    formula : str
        mu 参数的模型公式
    families : list of str
        候选分布族列表
    data : dict
        数据字典
    sigma_formula : str, default="~1"
        sigma 参数的公式
    parameter_formulas : dict, optional
        其他参数的公式
    criterion : str, default="AIC"
        选择准则: "AIC", "BIC", "deviance"
    verbose : bool, default=False
        是否打印详细信息
    **kwargs
        传递给 gamlss() 的其他参数
    
    Returns
    -------
    best_model : GAMLSSModel
        最佳模型（根据 criterion）
    comparison : pd.DataFrame
        所有模型的比较结果
    
    Examples
    --------
    >>> import numpy as np
    >>> from omnilss.model_selection import select_best_distribution
    >>> 
    >>> # 生成数据
    >>> np.random.seed(42)
    >>> n = 100
    >>> x = np.random.randn(n)
    >>> y = 2 + 3*x + np.random.randn(n)
    >>> data = {"y": y, "x": x}
    >>> 
    >>> # 自动选择最佳分布
    >>> best_model, comparison = select_best_distribution(
    ...     "y ~ x",
    ...     families=["NO", "GA", "LOGNO"],
    ...     data=data,
    ...     verbose=True
    ... )
    >>> 
    >>> # 使用最佳模型进行预测
    >>> newdata = {"x": np.array([0, 1, 2])}
    >>> predictions = best_model.predict_quantiles(newdata, [0.05, 0.5, 0.95])
    
    Notes
    -----
    这个函数首先比较所有候选分布，然后重新拟合最佳分布并返回。
    这确保返回的模型是完整的，可以用于后续的预测和诊断。
    """
    from .fitting import gamlss
    
    # 比较所有分布
    comparison = compare_distributions(
        formula=formula,
        families=families,
        data=data,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
        criterion=criterion,
        verbose=verbose,
        **kwargs
    )
    
    # 获取最佳分布
    best_family = comparison.iloc[0]["family"]
    
    if verbose:
        print(f"\n重新拟合最佳分布: {best_family}")
    
    # 重新拟合最佳模型
    best_model = gamlss(
        formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
        family=best_family,
        data=data,
        **kwargs
    )
    
    return best_model, comparison


def stepwise_distribution_selection(
    formula: str,
    data: dict,
    family_groups: Optional[Dict[str, List[str]]] = None,
    sigma_formula: str = "~1",
    parameter_formulas: Optional[Dict[str, str]] = None,
    criterion: str = "AIC",
    verbose: bool = False,
    **kwargs
) -> tuple:
    """逐步分布选择
    
    按照分布族的类型分组进行逐步选择，先选择最佳类型，
    再在该类型中选择最佳分布。
    
    Parameters
    ----------
    formula : str
        模型公式
    data : dict
        数据字典
    family_groups : dict, optional
        分布族分组，如 {
            "Normal": ["NO", "LOGNO"],
            "Gamma": ["GA", "GG"],
            "Poisson": ["PO", "NBI"]
        }
        如果未指定，使用默认分组
    sigma_formula : str, default="~1"
        sigma 参数的公式
    parameter_formulas : dict, optional
        其他参数的公式
    criterion : str, default="AIC"
        选择准则
    verbose : bool, default=False
        是否打印详细信息
    **kwargs
        传递给 gamlss() 的其他参数
    
    Returns
    -------
    best_model : GAMLSSModel
        最佳模型
    group_comparison : pd.DataFrame
        分组比较结果
    final_comparison : pd.DataFrame
        最佳组内的详细比较结果
    
    Examples
    --------
    >>> best_model, group_comp, final_comp = stepwise_distribution_selection(
    ...     "y ~ x",
    ...     data=data,
    ...     verbose=True
    ... )
    """
    # 默认分组
    if family_groups is None:
        family_groups = {
            "Normal": ["NO", "LOGNO"],
            "Gamma": ["GA"],
            "Poisson": ["PO", "NBI"],
            "Binomial": ["BI"],
            "Beta": ["BE"],
        }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"逐步分布选择")
        print(f"{'='*70}\n")
        print(f"第 1 步: 比较 {len(family_groups)} 个分布族组")
    
    # 第 1 步：每组选择一个代表
    group_results = []
    for group_name, families in family_groups.items():
        if verbose:
            print(f"\n  组 '{group_name}': {families}")
        
        # 比较该组内的分布
        comparison = compare_distributions(
            formula=formula,
            families=families,
            data=data,
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas,
            criterion=criterion,
            verbose=False,
            **kwargs
        )
        
        # 选择该组最佳
        best_in_group = comparison.iloc[0]
        group_results.append({
            "group": group_name,
            "best_family": best_in_group["family"],
            criterion: best_in_group[criterion]
        })
        
        if verbose:
            print(f"    最佳: {best_in_group['family']} ({criterion}={best_in_group[criterion]:.2f})")
    
    # 转换为 DataFrame
    group_comparison = pd.DataFrame(group_results).sort_values(criterion)
    
    if verbose:
        print(f"\n第 2 步: 选择最佳组")
        print(group_comparison)
    
    # 第 2 步：在最佳组中详细比较
    best_group = group_comparison.iloc[0]["group"]
    best_families = family_groups[best_group]
    
    if verbose:
        print(f"\n第 3 步: 在最佳组 '{best_group}' 中详细比较")
    
    # 详细比较
    best_model, final_comparison = select_best_distribution(
        formula=formula,
        families=best_families,
        data=data,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas,
        criterion=criterion,
        verbose=verbose,
        **kwargs
    )
    
    return best_model, group_comparison, final_comparison


# 预定义的常用分布组合
COMMON_FAMILIES = {
    "continuous_positive": ["GA", "LOGNO", "WEI", "IG"],
    "continuous_real": ["NO", "TF", "LO"],
    "count": ["PO", "NBI", "ZIP"],
    "proportion": ["BE", "BEINF"],
    "all_common": ["NO", "GA", "LOGNO", "PO", "BI", "BE", "NBI", "WEI"]
}


def quick_distribution_search(
    formula: str,
    data: dict,
    data_type: str = "auto",
    sigma_formula: str = "~1",
    criterion: str = "AIC",
    verbose: bool = False,
    **kwargs
) -> tuple:
    """快速分布搜索
    
    根据数据类型自动选择候选分布族进行比较。
    
    Parameters
    ----------
    formula : str
        模型公式
    data : dict
        数据字典
    data_type : str, default="auto"
        数据类型: "auto", "continuous_positive", "continuous_real", 
        "count", "proportion"
    sigma_formula : str, default="~1"
        sigma 参数的公式
    criterion : str, default="AIC"
        选择准则
    verbose : bool, default=False
        是否打印详细信息
    **kwargs
        传递给 gamlss() 的其他参数
    
    Returns
    -------
    best_model : GAMLSSModel
        最佳模型
    comparison : pd.DataFrame
        比较结果
    
    Examples
    --------
    >>> # 自动检测数据类型并选择分布
    >>> best_model, comparison = quick_distribution_search(
    ...     "y ~ x",
    ...     data=data,
    ...     data_type="auto",
    ...     verbose=True
    ... )
    """
    from .fitting import _parse_formula
    
    # 自动检测数据类型
    if data_type == "auto":
        response_name, _ = _parse_formula(formula)
        y = np.asarray(data[response_name])
        
        # 检测数据类型
        if np.all(y >= 0):
            if np.all(y == np.floor(y)):
                data_type = "count"
            elif np.all((y >= 0) & (y <= 1)):
                data_type = "proportion"
            else:
                data_type = "continuous_positive"
        else:
            data_type = "continuous_real"
        
        if verbose:
            print(f"自动检测数据类型: {data_type}")
    
    # 选择候选分布
    if data_type in COMMON_FAMILIES:
        families = COMMON_FAMILIES[data_type]
    else:
        families = COMMON_FAMILIES["all_common"]
    
    if verbose:
        print(f"候选分布: {families}")
    
    # 选择最佳分布
    return select_best_distribution(
        formula=formula,
        families=families,
        data=data,
        sigma_formula=sigma_formula,
        criterion=criterion,
        verbose=verbose,
        **kwargs
    )
