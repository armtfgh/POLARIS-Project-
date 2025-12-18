#!/usr/bin/env python3
"""
test_improved_cartographer.py
=============================

Demo script comparing the OLD vs IMPROVED knowledge extraction pipelines.

Key improvements in the new pipeline:
1. Direct JSON generation (eliminates prose→JSON translation loss)
2. No ground truth contamination in prompts
3. Enhanced data representation with binned analysis
4. Increased max_tokens (1500) to prevent truncation
5. Better handling of sparse data and uncertainty

Usage:
    python test_improved_cartographer.py
"""

from cartographer import (
    run_cartographer_analyst_pipeline,
    run_cartographer_analyst_pipeline_improved,
    plot_knowledge_curve,
)
import matplotlib.pyplot as plt
from os import getenv

# Get API key from environment
api_key = getenv("OPENAI_API_KEY", None)

def run_comparison_demo(budget: int = 100, analyst_every: int = 20):
    """
    Run both pipelines side-by-side and compare NDCG scores.
    """
    print("=" * 70)
    print("KNOWLEDGE EXTRACTION PIPELINE COMPARISON")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Budget: {budget} iterations")
    print(f"  Analyst checkpoints: every {analyst_every} iterations")
    print(f"  Initial samples: 6")
    print(f"  Seed: 42")
    print()

    # Run IMPROVED pipeline (direct JSON)
    print("Running IMPROVED pipeline (direct JSON generation)...")
    print("-" * 70)
    hist_improved, knowledge_improved = run_cartographer_analyst_pipeline_improved(
        csv_path="ugi_merged_dataset.csv",
        budget=budget,
        n_init=6,
        seed=42,
        analyst_every=analyst_every,
        analyst_model="gpt-4o",
        api_key=api_key,
        use_direct_json=True,  # NEW: Direct JSON approach
    )
    print("\nImproved pipeline results:")
    print(knowledge_improved[["iter", "n_obs", "ndcg"]])
    print(f"\nMean NDCG (improved): {knowledge_improved['ndcg'].mean():.4f}")
    print(f"Std NDCG (improved):  {knowledge_improved['ndcg'].std():.4f}")
    print()

    # Run LEGACY pipeline (two-step)
    print("\nRunning LEGACY pipeline (two-step prose→JSON)...")
    print("-" * 70)
    hist_legacy, knowledge_legacy = run_cartographer_analyst_pipeline_improved(
        csv_path="ugi_merged_dataset.csv",
        budget=budget,
        n_init=6,
        seed=42,
        analyst_every=analyst_every,
        analyst_model="gpt-4o",
        api_key=api_key,
        use_direct_json=False,  # OLD: Two-step approach
    )
    print("\nLegacy pipeline results:")
    print(knowledge_legacy[["iter", "n_obs", "ndcg"]])
    print(f"\nMean NDCG (legacy): {knowledge_legacy['ndcg'].mean():.4f}")
    print(f"Std NDCG (legacy):  {knowledge_legacy['ndcg'].std():.4f}")
    print()

    # Comparison
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    mean_diff = knowledge_improved['ndcg'].mean() - knowledge_legacy['ndcg'].mean()
    std_diff = knowledge_improved['ndcg'].std() - knowledge_legacy['ndcg'].std()
    print(f"Mean NDCG improvement:       {mean_diff:+.4f}")
    print(f"Std NDCG improvement:        {std_diff:+.4f}")
    print(f"Final NDCG (improved):       {knowledge_improved['ndcg'].iloc[-1]:.4f}")
    print(f"Final NDCG (legacy):         {knowledge_legacy['ndcg'].iloc[-1]:.4f}")
    print()

    if mean_diff > 0:
        print("✓ Improved pipeline shows BETTER knowledge extraction")
    else:
        print("⚠ Results are similar or worse (may need more iterations)")
    print()

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Improved pipeline
    plot_knowledge_curve(knowledge_improved, ax=ax1, title="IMPROVED: Direct JSON")
    ax1.axhline(knowledge_improved['ndcg'].mean(), color='green', linestyle='--', alpha=0.5, label='Mean')

    # Legacy pipeline
    plot_knowledge_curve(knowledge_legacy, ax=ax2, title="LEGACY: Two-step prose→JSON")
    ax2.axhline(knowledge_legacy['ndcg'].mean(), color='red', linestyle='--', alpha=0.5, label='Mean')

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig("cartographer_comparison.png", dpi=150)
    print("Saved comparison plot to: cartographer_comparison.png")

    return hist_improved, knowledge_improved, hist_legacy, knowledge_legacy


def run_quick_test():
    """
    Quick test with just a few checkpoints to verify the improved pipeline works.
    """
    print("=" * 70)
    print("QUICK TEST: Improved pipeline with 2 checkpoints")
    print("=" * 70)

    hist, knowledge = run_cartographer_analyst_pipeline_improved(
        csv_path="ugi_merged_dataset.csv",
        budget=40,
        n_init=6,
        seed=123,
        analyst_every=20,
        analyst_model="gpt-4o",
        api_key=api_key,
        use_direct_json=True,
    )

    print("\nResults:")
    print(knowledge[["iter", "n_obs", "ndcg"]])

    if len(knowledge) > 0 and "readout" in knowledge.columns:
        print("\nExample readout structure from last checkpoint:")
        last_readout = knowledge.iloc[-1]["readout"]
        print(f"  Effects: {list(last_readout.get('effects', {}).keys())}")
        print(f"  Interactions: {len(last_readout.get('interactions', []))}")
        print(f"  Bumps: {len(last_readout.get('bumps', []))}")

        if len(last_readout.get('effects', {})) > 0:
            print("\n  Example effect (x3):")
            x3_effect = last_readout.get('effects', {}).get('x3', {})
            print(f"    Type: {x3_effect.get('effect', 'N/A')}")
            print(f"    Scale: {x3_effect.get('scale', 0):.3f}")
            print(f"    Confidence: {x3_effect.get('confidence', 0):.3f}")

    print("\n✓ Quick test completed successfully!")
    return hist, knowledge


if __name__ == "__main__":
    import sys

    # Check if user wants quick test or full comparison
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("Running quick test...\n")
        hist, knowledge = run_quick_test()
    else:
        print("Running full comparison (this may take several minutes)...")
        print("For a quick test, run: python test_improved_cartographer.py --quick\n")
        results = run_comparison_demo(budget=100, analyst_every=20)

    plt.show()
