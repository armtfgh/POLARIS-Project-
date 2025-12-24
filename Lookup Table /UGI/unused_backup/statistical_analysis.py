#!/usr/bin/env python3
"""
statistical_analysis.py
=======================

Statistical rigor for BO benchmarking.

Provides:
- Significance testing (Wilcoxon signed-rank, t-tests)
- Effect size computation (Cohen's d, rank-biserial correlation)
- Bootstrap confidence intervals
- Multiple comparison correction (Bonferroni, Holm)
- Power analysis

Usage:
    results = statistical_comparison(hist_df, "hybrid_best", "baseline_ei")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Significant: {results['significant']}")
    print(f"Effect size: {results['cohens_d']:.3f} ({results['interpretation']})")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt


def statistical_comparison(
    hist_df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = "best_so_far",
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> Dict:
    """
    Compare two methods with rigorous statistical testing.

    Args:
        hist_df: Long-form DataFrame with columns [method, seed, iter, best_so_far]
        method_a: First method name
        method_b: Second method name
        metric: Column to compare (default: "best_so_far")
        alpha: Significance level (default: 0.05)
        n_bootstrap: Bootstrap iterations for CI

    Returns:
        Dictionary with:
        - p_value: Wilcoxon signed-rank test p-value
        - significant: Whether difference is significant
        - cohens_d: Effect size
        - interpretation: Effect size interpretation
        - mean_diff: Mean difference
        - ci_95: Bootstrap 95% confidence interval
        - power: Statistical power (if significant)
    """
    # Get final performance for each seed
    final_a = hist_df[hist_df['method'] == method_a].groupby('seed')[metric].max()
    final_b = hist_df[hist_df['method'] == method_b].groupby('seed')[metric].max()

    # Use only common seeds (paired comparison)
    common_seeds = sorted(set(final_a.index) & set(final_b.index))

    if len(common_seeds) < 3:
        raise ValueError(f"Need at least 3 common seeds, got {len(common_seeds)}")

    vals_a = final_a[common_seeds].values
    vals_b = final_b[common_seeds].values
    n = len(vals_a)

    # 1. Wilcoxon signed-rank test (non-parametric, paired)
    # Better than t-test when distribution is non-normal
    statistic, p_value = stats.wilcoxon(vals_a, vals_b, alternative='two-sided')

    # 2. Effect size: Cohen's d (standardized mean difference)
    diff = vals_a - vals_b
    mean_diff = np.mean(diff)
    pooled_std = np.sqrt((np.var(vals_a, ddof=1) + np.var(vals_b, ddof=1)) / 2)
    cohens_d = mean_diff / (pooled_std + 1e-12)

    # 3. Rank-biserial correlation (effect size for Wilcoxon)
    # r = Z / sqrt(n), ranges from -1 to 1
    z_score = stats.norm.ppf(1 - p_value / 2)  # Two-sided
    rank_biserial = z_score / np.sqrt(n)

    # 4. Bootstrap 95% confidence interval on difference
    bootstrap_diffs = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        bootstrap_diffs.append(np.mean(vals_a[idx] - vals_b[idx]))
    ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])

    # 5. Statistical power (if significant)
    if p_value < alpha:
        # Retrospective power analysis
        power = compute_power(vals_a, vals_b, alpha=alpha)
    else:
        power = None

    return {
        'method_a': method_a,
        'method_b': method_b,
        'n_seeds': n,
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'cohens_d': float(cohens_d),
        'rank_biserial': float(rank_biserial),
        'effect_interpretation': interpret_cohens_d(cohens_d),
        'mean_a': float(np.mean(vals_a)),
        'mean_b': float(np.mean(vals_b)),
        'mean_diff': float(mean_diff),
        'ci_95_lower': float(ci_low),
        'ci_95_upper': float(ci_high),
        'power': float(power) if power is not None else None,
        'test_used': 'Wilcoxon signed-rank',
    }


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size (Cohen, 1988)."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compute_power(vals_a: np.ndarray, vals_b: np.ndarray, alpha: float = 0.05) -> float:
    """
    Compute post-hoc statistical power using effect size and sample size.

    Power = P(reject H0 | H1 is true)
    """
    n = len(vals_a)
    effect_size = abs(np.mean(vals_a - vals_b)) / (np.std(vals_a - vals_b) + 1e-12)

    # Critical value for two-tailed test
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Non-centrality parameter
    delta = effect_size * np.sqrt(n)

    # Power = P(|Z| > z_crit | Z ~ N(delta, 1))
    power = 1 - stats.norm.cdf(z_crit - delta) + stats.norm.cdf(-z_crit - delta)

    return float(power)


def pairwise_comparison_table(
    hist_df: pd.DataFrame,
    methods: Optional[List[str]] = None,
    metric: str = "best_so_far",
    correction: str = "holm",  # "bonferroni", "holm", or None
) -> pd.DataFrame:
    """
    Pairwise comparison of all methods with multiple testing correction.

    Args:
        hist_df: History DataFrame
        methods: List of methods to compare (default: all)
        metric: Performance metric
        correction: Multiple testing correction method

    Returns:
        DataFrame with pairwise comparisons and adjusted p-values
    """
    if methods is None:
        methods = sorted(hist_df['method'].unique())

    results = []
    p_values = []

    # All pairwise comparisons
    for i, method_a in enumerate(methods):
        for method_b in methods[i+1:]:
            try:
                comp = statistical_comparison(hist_df, method_a, method_b, metric=metric)
                results.append(comp)
                p_values.append(comp['p_value'])
            except Exception as e:
                print(f"Warning: Could not compare {method_a} vs {method_b}: {e}")

    if not results:
        return pd.DataFrame()

    # Multiple testing correction
    if correction == "bonferroni":
        adjusted_p = [min(p * len(p_values), 1.0) for p in p_values]
    elif correction == "holm":
        adjusted_p = holm_bonferroni(p_values)
    else:
        adjusted_p = p_values

    # Add adjusted p-values
    for i, result in enumerate(results):
        result['p_value_adjusted'] = adjusted_p[i]
        result['significant_adjusted'] = adjusted_p[i] < 0.05

    return pd.DataFrame(results)


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni correction for multiple testing.

    More powerful than Bonferroni while controlling family-wise error rate.
    """
    n = len(p_values)
    # Sort p-values with original indices
    sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * n
    for rank, (idx, p) in enumerate(sorted_p):
        adjusted[idx] = min(p * (n - rank), 1.0)

    # Enforce monotonicity
    for i in range(1, n):
        adjusted[sorted_p[i][0]] = max(adjusted[sorted_p[i][0]], adjusted[sorted_p[i-1][0]])

    return adjusted


def convergence_comparison(
    hist_df: pd.DataFrame,
    methods: List[str],
    metric: str = "best_so_far",
    threshold: float = None,
) -> Dict:
    """
    Compare convergence speed of multiple methods.

    Returns iteration at which each method reaches threshold performance.
    """
    if threshold is None:
        # Use 90% of global maximum as threshold
        threshold = 0.9 * hist_df[metric].max()

    convergence = {}

    for method in methods:
        data = hist_df[hist_df['method'] == method]

        # For each seed, find first iteration reaching threshold
        iters_to_threshold = []
        for seed in data['seed'].unique():
            seed_data = data[data['seed'] == seed].sort_values('iter')
            reached = seed_data[seed_data[metric] >= threshold]

            if len(reached) > 0:
                iters_to_threshold.append(reached['iter'].min())
            else:
                iters_to_threshold.append(np.inf)

        convergence[method] = {
            'mean_iters': np.mean(iters_to_threshold),
            'median_iters': np.median(iters_to_threshold),
            'std_iters': np.std(iters_to_threshold),
            'success_rate': np.mean([x < np.inf for x in iters_to_threshold]),
            'iters_per_seed': iters_to_threshold,
        }

    return convergence


def plot_significance_matrix(
    comparison_df: pd.DataFrame,
    save_path: str = "significance_matrix.png"
):
    """
    Visualize pairwise significance as a matrix.

    Green = significantly better, Red = significantly worse, Gray = no difference
    """
    methods = sorted(set(comparison_df['method_a']) | set(comparison_df['method_b']))
    n = len(methods)

    # Create matrix
    matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))

    method_to_idx = {m: i for i, m in enumerate(methods)}

    for _, row in comparison_df.iterrows():
        i = method_to_idx[row['method_a']]
        j = method_to_idx[row['method_b']]

        if row['significant']:
            # Positive mean_diff means method_a better
            if row['mean_diff'] > 0:
                matrix[i, j] = 1  # A better than B
                matrix[j, i] = -1  # B worse than A
            else:
                matrix[i, j] = -1  # A worse than B
                matrix[j, i] = 1  # B better than A

        p_matrix[i, j] = row['p_value']
        p_matrix[j, i] = row['p_value']

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # Annotate with p-values
    for i in range(n):
        for j in range(n):
            if i != j:
                p_val = p_matrix[i, j]
                if p_val < 0.001:
                    text = "***"
                elif p_val < 0.01:
                    text = "**"
                elif p_val < 0.05:
                    text = "*"
                else:
                    text = "n.s."

                ax.text(j, i, text, ha="center", va="center",
                       color="black" if abs(matrix[i, j]) > 0.5 else "gray")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels(methods)
    ax.set_title("Pairwise Significance Matrix\n(Green=Better, Red=Worse, ***p<0.001)")

    plt.colorbar(im, ax=ax, label="Performance difference")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved significance matrix to: {save_path}")


def bootstrap_difference_distribution(
    hist_df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = "best_so_far",
    n_bootstrap: int = 10000,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate bootstrap distribution of performance difference.

    Useful for visualizing uncertainty and effect size.
    """
    final_a = hist_df[hist_df['method'] == method_a].groupby('seed')[metric].max()
    final_b = hist_df[hist_df['method'] == method_b].groupby('seed')[metric].max()

    common_seeds = sorted(set(final_a.index) & set(final_b.index))
    vals_a = final_a[common_seeds].values
    vals_b = final_b[common_seeds].values
    n = len(vals_a)

    # Bootstrap
    bootstrap_diffs = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        bootstrap_diffs.append(np.mean(vals_a[idx] - vals_b[idx]))

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Compute statistics
    stats_dict = {
        'mean': np.mean(bootstrap_diffs),
        'median': np.median(bootstrap_diffs),
        'std': np.std(bootstrap_diffs),
        'ci_95': tuple(np.percentile(bootstrap_diffs, [2.5, 97.5])),
        'p_value_bootstrap': np.mean(bootstrap_diffs <= 0) * 2,  # Two-sided
    }

    return bootstrap_diffs, stats_dict


if __name__ == "__main__":
    # Example usage
    print("Statistical Analysis Module")
    print("=" * 50)
    print("\nExample: Compare two methods")
    print("""
    from statistical_analysis import statistical_comparison, pairwise_comparison_table

    # Single comparison
    result = statistical_comparison(hist_df, "hybrid_best", "baseline_ei")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Effect size (Cohen's d): {result['cohens_d']:.3f} ({result['effect_interpretation']})")
    print(f"95% CI: [{result['ci_95_lower']:.4f}, {result['ci_95_upper']:.4f}]")

    # Multiple comparisons with correction
    table = pairwise_comparison_table(hist_df, correction='holm')
    print(table[['method_a', 'method_b', 'p_value_adjusted', 'cohens_d', 'significant_adjusted']])
    """)
