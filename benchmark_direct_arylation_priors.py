
#!/usr/bin/env python
"""
benchmark_direct_arylation_priors.py

Run a closed-loop benchmark on the Direct Arylation oracle to compare prior strategies:
  - NoPrior
  - Heuristic (role + optional descriptor-based category similarity)
  - LLM-Good (requires OPENAI_API_KEY; else auto-falls back to Heuristic-Good template)
  - LLM-Bad  (synthetic mis-specified prior for stress-testing)

It expects:
  - oracle.pkl   : joblib pipeline that accepts DataFrame with columns
                   ["Base_SMILES", "Ligand_SMILES", "Solvent_SMILES", "Concentration", "Temp_C"]
  - experiment_index.csv : used to derive candidate categorical choices and numeric bounds
  - optional descriptors_dir with base/ligand/solvent descriptor CSVs to build category similarity

Outputs:
  - convergence.png
  - summary.csv
"""

import argparse, os, json, math, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import norm

try:
    import joblib
except Exception:
    joblib = None

# Import the prior module
import sys
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append("/mnt/data")  # fallback for this environment
import language_shaped_prior_llm as lsp

# -------------------- Oracle utilities --------------------

def load_oracle(oracle_path: str):
    if joblib is None:
        raise ImportError("joblib is required to load oracle.pkl. `pip install joblib`.")
    return joblib.load(oracle_path)

def oracle_predict_df(oracle, X_df: pd.DataFrame, noise_sd: float = 0.0) -> np.ndarray:
    mu = oracle.predict(X_df)
    if noise_sd > 0:
        mu = mu + np.random.normal(0, noise_sd, size=len(mu))
    return mu

# -------------------- Candidate space --------------------

def derive_domain_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    # Drop rows missing any key feature
    req = ["Base_SMILES","Ligand_SMILES","Solvent_SMILES","Concentration","Temp_C"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    df = df.dropna(subset=req)

    bases = sorted(df["Base_SMILES"].astype(str).unique().tolist())
    ligs  = sorted(df["Ligand_SMILES"].astype(str).unique().tolist())
    solv  = sorted(df["Solvent_SMILES"].astype(str).unique().tolist())

    c_lo, c_hi = float(df["Concentration"].min()), float(df["Concentration"].max())
    t_lo, t_hi = float(df["Temp_C"].min()), float(df["Temp_C"].max())

    schema = lsp.Schema(
        continuous=[
            lsp.ContinuousVar("Concentration", c_lo, c_hi, unit="M", role="concentration"),
            lsp.ContinuousVar("Temp_C", t_lo, t_hi, unit="C", role="temperature"),
        ],
        categorical=[
            lsp.CategoricalVar("Base_SMILES", bases, role="base"),
            lsp.CategoricalVar("Ligand_SMILES", ligs, role="ligand"),
            lsp.CategoricalVar("Solvent_SMILES", solv, role="solvent"),
        ],
    )
    return schema, df

def sample_candidates(n: int, rng: np.random.Generator, schema: lsp.Schema) -> pd.DataFrame:
    """Randomly sample mixed candidates from the domain defined by schema."""
    rows = []
    for _ in range(n):
        row = {}
        for v in schema.continuous:
            row[v.name] = rng.uniform(v.low, v.high)
        for c in schema.categorical:
            row[c.name] = rng.choice(c.choices)
        rows.append(row)
    return pd.DataFrame(rows)

# -------------------- Readouts (priors) --------------------

def build_no_prior(schema: lsp.Schema) -> Dict[str, Any]:
    return {
        "effects": {v.name: {"effect":"flat","scale":0.0,"confidence":0.0} for v in schema.continuous},
        "interactions": [],
        "category_similarity": {c.name: {a:{b:0.5 for b in c.choices if b!=a} for a in c.choices} for c in schema.categorical}
    }

def _ohe_safe():
    from sklearn.preprocessing import OneHotEncoder
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _cosine_sim(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    # normalize rows
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    X = A / n
    S = X @ X.T
    S = (S - S.min()) / (S.max() - S.min() + 1e-12)
    np.fill_diagonal(S, 0.0)
    return S

def _descriptor_similarity_table(items: List[str], table: pd.DataFrame, key_col: str) -> Dict[str, Dict[str, float]]:
    # Expect table with key column (SMILES) + numeric descriptor columns
    t = table.dropna(subset=[key_col]).copy()
    t = t[t[key_col].isin(items)].copy()
    num_cols = [c for c in t.columns if c != key_col and pd.api.types.is_numeric_dtype(t[c])]
    if not num_cols:
        return {a:{b:0.5 for b in items if b!=a} for a in items}
    X = t.set_index(key_col)[num_cols].astype(float).reindex(items).fillna(0.0).values
    S = _cosine_sim(X)
    sim = {}
    for i,a in enumerate(items):
        sim[a] = {}
        for j,b in enumerate(items):
            if a==b: continue
            sim[a][b] = float(S[i,j])
    return sim

def build_heuristic_readout(schema: lsp.Schema, descriptors_dir: Optional[str] = None) -> Dict[str, Any]:
    # Start from role-based heuristic
    readout = lsp.HeuristicReadout().produce(schema, context_bullets=[
        "Pd-catalyzed direct arylation (C–H activation)",
        "Base-sensitive steps influence rate/selectivity",
        "Temperature too high can cause decomposition; expect optimum",
        "Concentration often improves rate until mass-transfer/aggregation limits",
        "Polar aprotic solvents typically beneficial"
    ])
    # Adjust effects specifically
    # Temperature -> nonmonotone-peak, medium-high confidence
    readout["effects"]["Temp_C"] = {"effect":"nonmonotone-peak","scale":0.6,"confidence":0.7,"range_hint":[0.55,0.85]}
    # Concentration -> increase-saturating (or peak); choose peak with moderate confidence
    readout["effects"]["Concentration"] = {"effect":"nonmonotone-peak","scale":0.45,"confidence":0.6,"range_hint":[0.4,0.8]}
    # Optional continuous interaction (Temp x Conc): mild antagonism at extremes
    readout["interactions"] = [{"pair":["Temp_C","Concentration"],"type":"antagonism","confidence":0.3}]

    # Optionally build category similarity from descriptors
    if descriptors_dir and os.path.isdir(descriptors_dir):
        try:
            base_csv   = Path(descriptors_dir)/"base_descriptors.csv"
            ligand_csv = Path(descriptors_dir)/"ligand_descriptors.csv"
            solvent_csv= Path(descriptors_dir)/"solvent_descriptors.csv"
            if base_csv.exists():
                tb = pd.read_csv(base_csv)
                readout["category_similarity"]["Base_SMILES"]   = _descriptor_similarity_table(
                    schema.categorical[0].choices, tb, key_col="Base_SMILES" if "Base_SMILES" in tb.columns else "SMILES")
            if ligand_csv.exists():
                tl = pd.read_csv(ligand_csv)
                readout["category_similarity"]["Ligand_SMILES"] = _descriptor_similarity_table(
                    schema.categorical[1].choices, tl, key_col="Ligand_SMILES" if "Ligand_SMILES" in tl.columns else "SMILES")
            if solvent_csv.exists():
                ts = pd.read_csv(solvent_csv)
                readout["category_similarity"]["Solvent_SMILES"]= _descriptor_similarity_table(
                    schema.categorical[2].choices, ts, key_col="Solvent_SMILES" if "Solvent_SMILES" in ts.columns else "SMILES")
        except Exception as e:
            print("[warn] descriptor similarity failed; using neutral 0.5. Err:", e)

    return readout

def build_llm_good(schema: lsp.Schema) -> Dict[str, Any]:
    # Try LLM; if unavailable, generate a 'good' template prior
    try:
        llm_fn = lsp.make_openai_chat_llm_fn()
        return lsp.LLMReadout(llm_fn).produce(schema, context_bullets=[
            "Pd-catalyzed direct arylation (C–H activation)",
            "Sterically hindered biaryl phosphines often effective",
            "Strong, non-nucleophilic bases (tBuO-, Cs2CO3) common",
            "Polar aprotic solvents (DMAc, DMF, DMSO) beneficial"
        ])
    except Exception:
        # Template 'good' JSON
        return {
            "effects": {
                "Temp_C": {"effect":"nonmonotone-peak","scale":0.65,"confidence":0.8,"range_hint":[0.6,0.9]},
                "Concentration": {"effect":"increase-saturating","scale":0.5,"confidence":0.7}
            },
            "interactions":[{"pair":["Temp_C","Concentration"],"type":"antagonism","confidence":0.35}],
            "category_similarity": {c.name: {a:{b:0.6 for b in c.choices if b!=a} for a in c.choices} for c in schema.categorical}
        }

def build_llm_bad(schema: lsp.Schema) -> Dict[str, Any]:
    # Mis-specified prior for stress test
    return {
        "effects": {
            "Temp_C": {"effect":"increase","scale":0.6,"confidence":0.8},
            "Concentration": {"effect":"decrease","scale":0.5,"confidence":0.7}
        },
        "interactions":[{"pair":["Temp_C","Concentration"],"type":"synergy","confidence":0.5}],
        "category_similarity": {c.name: {a:{b:0.2 for b in c.choices if b!=a} for a in c.choices} for c in schema.categorical}
    }

# -------------------- BO loop (residualized GP + EI) --------------------

def fit_predict_gp(prior: lsp.PriorBuilder, X_df: pd.DataFrame, y: np.ndarray, Xcands: pd.DataFrame):
    kernel = 1.0*RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None, random_state=0)
    wrapper = lsp.ResidualizedRegressor(gp, prior)
    wrapper.fit_df(X_df, y)
    # Predict residuals on candidate augmented features
    X_base, _ = prior.base_numeric_X(Xcands)
    X_phi = prior.phi_df(Xcands)
    X_aug = np.concatenate([X_base, X_phi], axis=1)
    X_aug = wrapper._scaler.transform(X_aug)
    res_mean, res_std = wrapper._fitted_estimator.predict(X_aug, return_std=True)
    mu = prior.m0_df(Xcands) + res_mean
    sigma = res_std
    return mu, sigma, wrapper

def expected_improvement(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)

def closed_loop_once(oracle, schema: lsp.Schema, readout_name: str, readout: Dict[str, Any],
                     n_init=6, iters=20, cand_pool=600, seed=0, noise_sd=0.0):
    rng = np.random.default_rng(seed)
    prior = lsp.PriorBuilder(schema, readout)

    # Initial DoE
    X_obs = sample_candidates(n_init, rng, schema)
    y_obs = oracle_predict_df(oracle, X_obs, noise_sd=noise_sd)
    best_so_far = [float(np.max(y_obs))]

    for t in range(iters):
        Xc = sample_candidates(cand_pool, rng, schema)
        mu, sigma, _ = fit_predict_gp(prior, X_obs, y_obs, Xc)
        y_best = float(np.max(y_obs))
        ei = expected_improvement(mu, sigma, y_best, xi=0.01)
        x_next = Xc.iloc[[int(np.argmax(ei))]].copy()
        y_next = oracle_predict_df(oracle, x_next, noise_sd=noise_sd)

        X_obs = pd.concat([X_obs, x_next], ignore_index=True)
        y_obs = np.concatenate([y_obs, y_next])
        best_so_far.append(float(np.max(y_obs)))

    return {"readout": readout_name, "best_so_far": np.array(best_so_far)}

# -------------------- Run & summarize --------------------

def area_under_curve(curve):
    x = np.arange(len(curve))
    return np.trapz(curve, x)

def run_benchmark(oracle_path: str, data_csv: str, out_dir: str,
                  n_reps=3, n_init=6, iters=20, cand_pool=600, noise_sd=0.0, seed=0,
                  descriptors_dir: Optional[str] = None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    schema, df = derive_domain_from_csv(data_csv)
    oracle = load_oracle(oracle_path)

    READOUTS = {
        "NoPrior": build_no_prior(schema),
        "Heuristic": build_heuristic_readout(schema, descriptors_dir=descriptors_dir),
        "LLM-Good": build_llm_good(schema),
        "LLM-Bad": build_llm_bad(schema),
    }

    results = []
    for name, ro in READOUTS.items():
        for r in range(n_reps):
            results.append(
                closed_loop_once(oracle, schema, name, ro, n_init=n_init, iters=iters,
                                 cand_pool=cand_pool, seed=seed + 19*r + hash(name)%1000, noise_sd=noise_sd)
            )

    # Summaries
    groups = {}
    for r in results: groups.setdefault(r["readout"], []).append(r["best_so_far"])

    xs = np.arange(results[0]["best_so_far"].shape[0])
    fig, ax = plt.subplots(figsize=(8,5))
    rows = []
    for name in sorted(groups.keys()):
        arr = np.vstack(groups[name])
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(xs, mean, label=name)
        ax.fill_between(xs, mean-std, mean+std, alpha=0.2)
        aucs = [area_under_curve(c) for c in arr]
        best10 = [c[min(10, len(c)-1)] for c in arr]
        rows.append({
            "readout": name,
            "AUC_mean": float(np.mean(aucs)),
            "AUC_std": float(np.std(aucs)),
            "Best@10_mean": float(np.mean(best10)),
            "Best@10_std": float(np.std(best10)),
        })
    ax.set_xlabel("Iterations (incl. init)")
    ax.set_ylabel("Best-so-far yield")
    ax.set_title("Direct Arylation: closed-loop convergence (higher is better)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out/"convergence.png", dpi=180, bbox_inches="tight")

    summary = pd.DataFrame(rows).sort_values("AUC_mean", ascending=False)
    summary.to_csv(out/"summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"[ok] Saved plot: {out/'convergence.png'}")
    print(f"[ok] Saved table: {out/'summary.csv'}")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", type=str, required=True, help="Path to oracle.pkl")
    ap.add_argument("--data_csv", type=str, required=True, help="Path to experiment_index.csv")
    ap.add_argument("--out_dir", type=str, default="priors_benchmark_out")
    ap.add_argument("--n_reps", type=int, default=3)
    ap.add_argument("--n_init", type=int, default=6)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cand_pool", type=int, default=600)
    ap.add_argument("--noise_sd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--descriptors_dir", type=str, default=None)
    args = ap.parse_args()

    run_benchmark(args.oracle, args.data_csv, args.out_dir,
                  n_reps=args.n_reps, n_init=args.n_init, iters=args.iters, cand_pool=args.cand_pool,
                  noise_sd=args.noise_sd, seed=args.seed, descriptors_dir=args.descriptors_dir)

if __name__ == "__main__":
    main()
