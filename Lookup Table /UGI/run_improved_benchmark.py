#!/usr/bin/env python3
"""
run_improved_benchmark.py
=========================

Example script showing how to run improved benchmarking with:
- Statistical significance testing
- Additional baselines (GP-UCB, Thompson Sampling)
- Proper evaluation metrics
- Publication-ready plots

This demonstrates the improvements outlined in METHODOLOGY_REVIEW.md
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Your existing code
from main_benchmark import load_lookup_csv, run_hybrid_lookup, run_baseline_ei_lookup, run_random_lookup

# New improved modules
from statistical_analysis import (
    statistical_comparison,
    pairwise_comparison_table,
    plot_significance_matrix,
    convergence_comparison,
)
from improved_baselines import run_gp_ucb_lookup, run_thompson_sampling_lookup


def run_comprehensive_experiment(
    csv_path: str = "ugi_merged_dataset.csv",
    n_init: int = 6,
    n_iter: int = 50,
    repeats: int = 10,  # Use 10+ for publication
    seed: int = 42,
    output_dir: str = "results",
):
    """
    Run comprehensive BO comparison with statistical rigor.

    Returns:
        - hist_df: Full history DataFrame
        - stats_df: Statistical comparison results
        - convergence_dict: Convergence analysis
    """
    print("="*70)
    print("IMPROVED BAYESIAN OPTIMIZATION BENCHMARK")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {csv_path}")
    print(f"  n_init: {n_init}, n_iter: {n_iter}, repeats: {repeats}")
    print(f"  Random seed: {seed}\n")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    lookup = load_lookup_csv(csv_path)
    print(f"  Loaded {lookup.n} samples with {lookup.d} features")
    print(f"  Objective range: [{float(lookup.y.min()):.4f}, {float(lookup.y.max()):.4f}]\n")

    # ===================================================================
    # 1. RUN ALL METHODS
    # ===================================================================

    print("Running methods...")
    all_dfs = []

    methods_to_run = [
        ("random", lambda s: run_random_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=s, init_method="sobol")),
        ("baseline_ei", lambda s: run_baseline_ei_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=s, init_method="sobol")),
        ("gp_ucb", lambda s: run_gp_ucb_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=s, beta=2.0, init_method="sobol")),
        ("thompson_sampling", lambda s: run_thompson_sampling_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=s, init_method="sobol")),
        ("hybrid_flat", lambda s: run_hybrid_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=s, readout_source="flat", init_method="sobol")),
        # Add your LLM-based priors here:
        # ("hybrid_best", lambda s: run_hybrid_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=s, readout_source="llm", prompt_profile="best")),
    ]

    for method_name, method_fn in methods_to_run:
        print(f"\n  Running {method_name}...")
        for r in range(repeats):
            current_seed = seed + r
            print(f"    Seed {current_seed}...", end="", flush=True)

            try:
                df = method_fn(current_seed)
                df['seed'] = current_seed
                df['method'] = method_name
                all_dfs.append(df)
                print(" ✓")
            except Exception as e:
                print(f" ✗ Failed: {e}")

    hist_df = pd.concat(all_dfs, ignore_index=True)
    hist_df.to_csv(f"{output_dir}/full_history.csv", index=False)
    print(f"\n✓ Saved full history to {output_dir}/full_history.csv")

    # ===================================================================
    # 2. STATISTICAL ANALYSIS
    # ===================================================================

    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    # Pairwise comparisons with multiple testing correction
    print("\nComputing pairwise comparisons...")
    stats_df = pairwise_comparison_table(
        hist_df,
        methods=None,  # Compare all
        metric="best_so_far",
        correction="holm"  # Holm-Bonferroni correction
    )
    stats_df.to_csv(f"{output_dir}/statistical_comparisons.csv", index=False)
    print(f"✓ Saved to {output_dir}/statistical_comparisons.csv\n")

    # Print key comparisons
    print("Key Findings:")
    print("-" * 70)

    # Compare your method to baselines
    your_method = "hybrid_flat"  # or "hybrid_best"
    baselines = ["baseline_ei", "gp_ucb", "thompson_sampling"]

    for baseline in baselines:
        comparison = stats_df[
            ((stats_df['method_a'] == your_method) & (stats_df['method_b'] == baseline)) |
            ((stats_df['method_a'] == baseline) & (stats_df['method_b'] == your_method))
        ]

        if len(comparison) > 0:
            row = comparison.iloc[0]
            # Ensure your_method is always method_a for consistent interpretation
            if row['method_a'] != your_method:
                mean_diff = -row['mean_diff']
                ci_low, ci_high = -row['ci_95_upper'], -row['ci_95_lower']
            else:
                mean_diff = row['mean_diff']
                ci_low, ci_high = row['ci_95_lower'], row['ci_95_upper']

            sig_symbol = "***" if row['p_value_adjusted'] < 0.001 else \
                         "**" if row['p_value_adjusted'] < 0.01 else \
                         "*" if row['p_value_adjusted'] < 0.05 else "n.s."

            print(f"\n{your_method} vs {baseline}:")
            print(f"  Mean difference: {mean_diff:+.5f} [{ci_low:.5f}, {ci_high:.5f}]")
            print(f"  p-value: {row['p_value_adjusted']:.4f} {sig_symbol}")
            print(f"  Effect size (Cohen's d): {row['cohens_d']:.3f} ({row['effect_interpretation']})")

            if row['significant_adjusted']:
                improvement_pct = (mean_diff / abs(row['mean_b'])) * 100
                print(f"  → {your_method} is {abs(improvement_pct):.1f}% better!")
            else:
                print(f"  → No significant difference")

    # Convergence analysis
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)

    threshold_90 = 0.9 * float(lookup.y.max())
    convergence = convergence_comparison(
        hist_df,
        methods=hist_df['method'].unique(),
        threshold=threshold_90
    )

    print(f"\nIterations to reach 90% optimum ({threshold_90:.4f}):")
    print("-" * 70)
    for method, data in sorted(convergence.items(), key=lambda x: x[1]['mean_iters']):
        print(f"{method:25s}: {data['mean_iters']:6.1f} ± {data['std_iters']:5.1f} iters "
              f"(success rate: {data['success_rate']:.1%})")

    # ===================================================================
    # 3. VISUALIZATION
    # ===================================================================

    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    # Plot 1: Convergence curves with bootstrap CI
    print("\n1. Convergence curves...")
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in hist_df['method'].unique():
        data = hist_df[(hist_df['method'] == method) & (hist_df['iter'] >= 0)]

        # Group by iteration
        agg = data.groupby('iter')['best_so_far'].agg(['mean', 'std', 'count']).reset_index()

        # Bootstrap confidence intervals
        from statistical_analysis import bootstrap_difference_distribution
        # Use 95% CI
        ci_95 = 1.96 * agg['std'] / np.sqrt(agg['count'])

        ax.plot(agg['iter'], agg['mean'], label=method, linewidth=2)
        ax.fill_between(agg['iter'], agg['mean'] - ci_95, agg['mean'] + ci_95, alpha=0.2)

    ax.axhline(threshold_90, color='red', linestyle='--', alpha=0.5, label='90% optimum')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Objective Found', fontsize=12)
    ax.set_title('Bayesian Optimization Comparison\n(Mean ± 95% CI)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence_curves.png", dpi=300)
    print(f"   ✓ Saved to {output_dir}/convergence_curves.png")

    # Plot 2: Final performance boxplot
    print("2. Final performance distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))

    final_perfs = []
    labels = []
    for method in sorted(hist_df['method'].unique()):
        data = hist_df[hist_df['method'] == method]
        finals = data.groupby('seed')['best_so_far'].max().values
        final_perfs.append(finals)
        labels.append(method)

    bp = ax.boxplot(final_perfs, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.6)

    ax.axhline(lookup.y.max(), color='red', linestyle='--', alpha=0.5, label='Global optimum')
    ax.set_ylabel('Final Best Objective', fontsize=12)
    ax.set_title('Final Performance Distribution Across Seeds', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_performance_boxplot.png", dpi=300)
    print(f"   ✓ Saved to {output_dir}/final_performance_boxplot.png")

    # Plot 3: Significance matrix
    print("3. Pairwise significance matrix...")
    plot_significance_matrix(stats_df, save_path=f"{output_dir}/significance_matrix.png")

    # ===================================================================
    # 4. SUMMARY REPORT
    # ===================================================================

    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)

    summary = hist_df.groupby('method').agg({
        'best_so_far': ['mean', 'std', 'max'],
        'seed': 'nunique'
    }).round(5)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={
        'best_so_far_mean': 'Mean Final',
        'best_so_far_std': 'Std Final',
        'best_so_far_max': 'Best Overall',
        'seed_nunique': 'Seeds'
    })

    # Add convergence info
    conv_means = {method: convergence[method]['mean_iters'] for method in convergence}
    summary['Iters to 90%'] = summary.index.map(lambda x: conv_means.get(x, np.nan))

    print(summary.to_string())

    # Save summary
    summary.to_csv(f"{output_dir}/summary_table.csv")
    print(f"\n✓ Saved summary to {output_dir}/summary_table.csv")

    # ===================================================================
    # 5. LATEX TABLE FOR PAPER
    # ===================================================================

    print("\n" + "="*70)
    print("LATEX TABLE (copy to paper)")
    print("="*70)

    latex_table = summary[['Mean Final', 'Std Final', 'Iters to 90%']].copy()
    latex_table['Mean ± Std'] = latex_table.apply(
        lambda row: f"{row['Mean Final']:.4f} ± {row['Std Final']:.4f}", axis=1
    )

    print("\n\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Method & Final Performance & Iterations to 90\\% \\\\")
    print("\\midrule")
    for method, row in latex_table.iterrows():
        iters = f"{row['Iters to 90%']:.1f}" if not np.isnan(row['Iters to 90%']) else "N/A"
        print(f"{method} & {row['Mean ± Std']} & {iters} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - full_history.csv: Raw data (all seeds, all iterations)")
    print(f"  - statistical_comparisons.csv: Pairwise significance tests")
    print(f"  - summary_table.csv: Performance summary")
    print(f"  - convergence_curves.png: Main results figure")
    print(f"  - final_performance_boxplot.png: Distribution analysis")
    print(f"  - significance_matrix.png: Pairwise comparisons")

    return hist_df, stats_df, convergence


if __name__ == "__main__":
    # Run experiment
    hist_df, stats_df, convergence = run_comprehensive_experiment(
        csv_path="ugi_merged_dataset.csv",
        n_init=6,
        n_iter=50,
        repeats=10,  # Use 10-20 for publication
        seed=42,
        output_dir="results_improved"
    )

    plt.show()

    print("\n" + "="*70)
    print("Next steps for paper:")
    print("="*70)
    print("1. Review statistical_comparisons.csv for p-values and effect sizes")
    print("2. Use convergence_curves.png as main figure")
    print("3. Include significance_matrix.png in appendix")
    print("4. Copy LaTeX table to paper")
    print("5. Report: 'Method X significantly outperforms baseline (p<0.001, Cohen's d=0.78)'")
