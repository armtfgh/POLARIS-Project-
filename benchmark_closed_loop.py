
#!/usr/bin/env python3
"""
benchmark_closed_loop.py

End-to-end, commented demo of closed-loop active learning with
LLM/heuristic priors using the residualized GP + EI acquisition.

Outputs:
- llm_prior_benchmark/convergence_demo.png
- llm_prior_benchmark/summary_demo.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the prior module (created earlier)
import sys
sys.path.append("/mnt/data")
import language_shaped_prior_llm as lsp

OUTDIR = Path("/mnt/data/llm_prior_benchmark")
OUTDIR.mkdir(parents=True, exist_ok=True)


# --------------------------
# 1) Define the domain & ground truth
# --------------------------

schema = lsp.Schema(
    continuous=[
        lsp.ContinuousVar("T", 25, 140, unit="C", role="temperature"),
        lsp.ContinuousVar("res_time", 0.2, 5.0, unit="min", role="time"),
        lsp.ContinuousVar("cat_loading", 0.1, 5.0, unit="mol%", role="catalyst"),
        lsp.ContinuousVar("base_equiv", 0.5, 3.0, unit="equiv", role="base_equiv"),
    ],
    categorical=[
        lsp.CategoricalVar("solvent", ["DMF","DMSO","MeCN","toluene"], role="solvent"),
    ],
)

def true_response(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Hidden simulator of the 'real world'. The optimizer never sees this function;
    it only sees y-values when it proposes x to evaluate.

    Design: peaked T, saturating res_time and cat_loading, peaked base_eq,
    solvent offsets, and a mild antagonism interaction (T x low catalyst).
    """
    T = df["T"].to_numpy(float)
    res = df["res_time"].to_numpy(float)
    cat = df["cat_loading"].to_numpy(float)
    base = df["base_equiv"].to_numpy(float)
    sol = df["solvent"].astype(str).values

    Tn = (T - 25) / (140 - 25)
    resn = (res - 0.2) / (5.0 - 0.2)
    catn = (cat - 0.1) / (5.0 - 0.1)
    basen = (base - 0.5) / (3.0 - 0.5)

    peak_T = np.exp(-0.5 * ((Tn - 0.70)/0.12)**2)
    inc_res = 1 - np.exp(-3.2 * np.clip(resn, 0, 1))
    inc_cat = 1 - np.exp(-1.5 * np.clip(catn, 0, 1))
    peak_base = np.exp(-0.5 * ((basen - 0.55)/0.18)**2)
    inter = (Tn * (0.5 - catn))

    sol_off = np.array([0.15 if s in {"DMF","DMSO"} else 0.05 if s=="MeCN" else -0.05 for s in sol])

    y = 40 + 35*peak_T + 15*inc_res + 8*inc_cat + 10*sol_off + 12*peak_base - 8*inter
    noise = rng.normal(0, 0.75, size=len(df))
    return y + noise


# --------------------------
# 2) Candidate generator (random sampling over the domain)
# --------------------------

def sample_designs(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "T": rng.uniform(25, 140, size=n),
        "res_time": rng.uniform(0.2, 5.0, size=n),
        "cat_loading": rng.uniform(0.1, 5.0, size=n),
        "base_equiv": rng.uniform(0.5, 3.0, size=n),
        "solvent": rng.choice(["DMF","DMSO","MeCN","toluene"], size=n)
    })


# --------------------------
# 3) Prior configurations
# --------------------------

def no_prior_readout(schema: lsp.Schema):
    return {"effects": {v.name: {"effect":"flat","scale":0.0,"confidence":0.0} for v in schema.continuous},
            "interactions": [],
            "category_similarity": {c.name: {a:{b:0.5 for b in c.choices if b!=a} for a in c.choices} for c in schema.categorical}}

heuristic_readout = lsp.HeuristicReadout().produce(schema, context_bullets=["substrate base-sensitive"])

llm_good = {
    "effects": {
        "T": {"effect": "nonmonotone-peak", "scale": 0.6, "confidence": 0.8, "range_hint": [0.55, 0.85]},
        "res_time": {"effect":"increase","scale":0.65,"confidence":0.7},
        "cat_loading": {"effect":"increase-saturating","scale":0.55,"confidence":0.7},
        "base_equiv": {"effect":"nonmonotone-peak","scale":0.55,"confidence":0.7,"range_hint":[0.45,0.75]},
    },
    "interactions":[{"pair":["T","cat_loading"],"type":"antagonism","confidence":0.4}],
    "category_similarity":{"solvent":{
        "DMF":{"DMSO":0.85,"MeCN":0.6,"toluene":0.1},
        "DMSO":{"DMF":0.85,"MeCN":0.6,"toluene":0.1},
        "MeCN":{"DMF":0.6,"DMSO":0.6,"toluene":0.2},
        "toluene":{"DMF":0.1,"DMSO":0.1,"MeCN":0.2}
    }}
}

llm_bad = {
    "effects": {
        "T": {"effect": "increase", "scale": 0.6, "confidence": 0.8},
        "res_time": {"effect":"nonmonotone-valley","scale":0.4,"confidence":0.5},
        "cat_loading": {"effect":"decrease","scale":0.5,"confidence":0.6},
        "base_equiv": {"effect":"increase","scale":0.5,"confidence":0.6},
    },
    "interactions":[{"pair":["T","cat_loading"],"type":"synergy","confidence":0.5}],
    "category_similarity":{"solvent":{
        "DMF":{"DMSO":0.2,"MeCN":0.2,"toluene":0.8},
        "DMSO":{"DMF":0.2,"MeCN":0.2,"toluene":0.8},
        "MeCN":{"DMF":0.2,"DMSO":0.2,"toluene":0.6},
        "toluene":{"DMF":0.8,"DMSO":0.8,"MeCN":0.6}
    }}
}

READOUTS = {
    "NoPrior": no_prior_readout(schema),
    "Heuristic": heuristic_readout,
    "LLM-Good": llm_good,
    "LLM-Bad": llm_bad,
}


# --------------------------
# 4) Residualized GP + EI acquisition (closed loop)
# --------------------------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import norm

def fit_predict_gp_fast(prior: lsp.PriorBuilder, X_df: pd.DataFrame, y: np.ndarray, Xcands: pd.DataFrame):
    """
    Train residualized GP on observed data and predict mean/std on candidate points.
    We fix GP hyperparams for speed/stability in-demo.
    """
    kernel = 1.0*RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None, random_state=0)

    model = lsp.ResidualizedRegressor(gp, prior)
    model.fit_df(X_df, y)

    # Build augmented features exactly like the residualized regressor does
    X_base, _ = prior.base_numeric_X(Xcands)
    X_phi = prior.phi_df(Xcands)
    X_aug = np.concatenate([X_base, X_phi], axis=1)
    X_aug = model._scaler.transform(X_aug)

    res_mean, res_std = model._fitted_estimator.predict(X_aug, return_std=True)
    mu = prior.m0_df(Xcands) + res_mean
    sigma = res_std
    return mu, sigma, model

def expected_improvement(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)

def closed_loop_once(readout_name: str, readout: dict, seed=0, n_init=6, iters=18, cand_pool=600):
    rng = np.random.default_rng(seed)
    prior = lsp.PriorBuilder(schema, readout)

    # Initial "ask" points (Design of Experiments)
    X_obs = sample_designs(n_init, rng)
    y_obs = true_response(X_obs, rng)

    best_so_far = [float(np.max(y_obs))]

    # Iterate: fit -> acquire -> evaluate -> update
    for t in range(iters):
        Xc = sample_designs(cand_pool, rng)  # random candidate set
        mu, sigma, model = fit_predict_gp_fast(prior, X_obs, y_obs, Xc)
        y_best = float(np.max(y_obs))
        ei = expected_improvement(mu, sigma, y_best, xi=0.01)
        x_next = Xc.iloc[[int(np.argmax(ei))]].copy()

        # "Tell" step: evaluate the true function at x_next
        y_next = true_response(x_next, rng)

        # Update dataset and logs
        X_obs = pd.concat([X_obs, x_next], ignore_index=True)
        y_obs = np.concatenate([y_obs, y_next])
        best_so_far.append(float(np.max(y_obs)))

    return {"readout": readout_name, "best_so_far": np.array(best_so_far)}


# --------------------------
# 5) Run reps and aggregate metrics
# --------------------------

def area_under_curve(curve):
    x = np.arange(len(curve))
    return np.trapz(curve, x)

def run_benchmark(n_reps=3, n_init=6, iters=18, seed=7):
    results = []
    for name, ro in READOUTS.items():
        for r in range(n_reps):
            results.append(closed_loop_once(name, ro, seed=seed + 17*r, n_init=n_init, iters=iters))
    return results

def summarize_and_plot(results, out_png, out_csv):
    # Plot convergence
    groups = {}
    for r in results: groups.setdefault(r["readout"], []).append(r["best_so_far"])

    xs = np.arange(results[0]["best_so_far"].shape[0])
    fig, ax = plt.subplots(figsize=(8,5))
    table_rows = []
    for name in sorted(groups.keys()):
        arr = np.vstack(groups[name])
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(xs, mean, label=name)
        ax.fill_between(xs, mean-std, mean+std, alpha=0.2)
        aucs = [area_under_curve(c) for c in arr]
        best10 = [c[min(10, len(c)-1)] for c in arr]
        table_rows.append({
            "readout": name,
            "AUC_mean": float(np.mean(aucs)),
            "AUC_std": float(np.std(aucs)),
            "Best@10_mean": float(np.mean(best10)),
            "Best@10_std": float(np.std(best10)),
        })
    ax.set_xlabel("Iterations (incl. init)")
    ax.set_ylabel("Best-so-far objective")
    ax.set_title("Closed-loop convergence (higher is better)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=180, bbox_inches="tight")

    df = pd.DataFrame(table_rows).sort_values("AUC_mean", ascending=False)
    df.to_csv(out_csv, index=False)
    return df

if __name__ == "__main__":
    results = run_benchmark(n_reps=3, n_init=6, iters=18, seed=7)
    df = summarize_and_plot(results, OUTDIR/"convergence_demo.png", OUTDIR/"summary_demo.csv")
    print(df.to_string(index=False))
    print(f"\nSaved plot: {OUTDIR/'convergence_demo.png'}")
    print(f"Saved CSV : {OUTDIR/'summary_demo.csv'}")
