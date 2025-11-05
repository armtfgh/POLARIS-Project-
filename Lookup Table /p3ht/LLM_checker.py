#%%
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

DOMAIN_CONTEXT = (
    "The formulation variables represent weight percentages (summing to 100%) for a composite made "
    "of regio-regular poly-3-hexylthiophene (P3HT) and carbon nanotube (CNT) additives:\n"
    "  • x1: P3HT content (%)\n"
    "  • x2: long single-wall CNTs (l-SWNTs)\n"
    "  • x3: short single-wall CNTs (s-SWNTs)\n"
    "  • x4: multi-walled CNTs (MWCNTs)\n"
    "  • x5: double-walled CNTs (DWCNTs)\n"
    "Objective is the composite conductivity (larger is better)."
)

try:
    import httpx
    from openai import OpenAI

    _OPENAI_CLIENT = OpenAI(http_client=httpx.Client(verify=False))
except Exception:
    _OPENAI_CLIENT = None


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.float32


@dataclass
class LookupTable:
    X_raw: torch.Tensor
    X: torch.Tensor
    y: torch.Tensor
    mins: torch.Tensor
    maxs: torch.Tensor
    feature_names: List[str]
    objective_name: str

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]


def load_lookup_csv(
    path: str,
    *,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
    impute_features: Optional[str] = None,
    aggregate_duplicates: Optional[str] = "mean",
) -> LookupTable:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least one feature column plus the objective.")

    feat_cols = list(df.columns[:-1])
    obj_col = df.columns[-1]

    for c in feat_cols + [obj_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df[obj_col] = df[obj_col].replace([np.inf, -np.inf], np.nan)

    mask_y = df[obj_col].notna()
    if impute_features == "median":
        med = df.loc[mask_y, feat_cols].median(numeric_only=True)
        df.loc[:, feat_cols] = df[feat_cols].fillna(med)
        mask_x = df[feat_cols].notna().all(axis=1)
    else:
        mask_x = df[feat_cols].notna().all(axis=1)

    df_clean = df.loc[mask_y & mask_x].reset_index(drop=True)
    if df_clean.shape[0] < 2:
        raise ValueError("Too few valid rows after cleaning.")

    dup_mask = df_clean.duplicated(subset=feat_cols, keep=False)
    if aggregate_duplicates and dup_mask.any():
        agg = aggregate_duplicates.lower()
        if agg not in {"mean", "median", "max"}:
            raise ValueError("aggregate_duplicates must be one of {'mean','median','max', None}")
        group = df_clean.groupby(feat_cols, as_index=False)[obj_col]
        if agg == "mean":
            df_clean = group.mean()
        elif agg == "median":
            df_clean = group.median()
        else:
            df_clean = group.max()
        print(f"[load_lookup_csv] Aggregated {int(dup_mask.sum())} duplicate rows using '{agg}'.")

    X_raw_np = df_clean[feat_cols].to_numpy(dtype=np.float64)
    y_np = df_clean[obj_col].to_numpy(dtype=np.float64)

    X_raw = torch.tensor(X_raw_np, dtype=dtype, device=device)
    y = torch.tensor(y_np, dtype=dtype, device=device).reshape(-1)

    mins = X_raw.min(dim=0).values
    maxs = X_raw.max(dim=0).values
    rng = (maxs - mins).clamp_min(1e-12)
    X = (X_raw - mins) / rng

    return LookupTable(
        X_raw=X_raw,
        X=X,
        y=y,
        mins=mins,
        maxs=maxs,
        feature_names=feat_cols,
        objective_name=str(obj_col),
    )


def select_initial_indices(n_total: int, n_init: int, seed: int) -> List[int]:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g)
    n_init = max(1, min(n_init, n_total))
    return perm[:n_init].tolist()


def remaining_indices(n_total: int, seen: set[int]) -> List[int]:
    return [i for i in range(n_total) if i not in seen]


def llm_predict_trend(
    summary: Dict[str, Any],
    *,
    objective_name: str,
) -> str:
    if _OPENAI_CLIENT is None:
        return random.choice(["increase", "decrease"])

    prompt = (
        "You are an analytical scientist helping a Bayesian optimizer develop P3HT/CNT composites.\n"
        f"{DOMAIN_CONTEXT}\n"
        "You receive a structured summary of the campaign so far (recent trials, incumbent recipe, posterior statistics, EI value) "
        "and the candidate the optimizer plans to run NEXT.\n"
        "predict whether the new measurement will IMPROVE (increase) or WORSEN (decrease) the best observed "
        f"{objective_name}. Respond strictly as JSON: "
        '{"prediction": "increase" or "decrease", "confidence": number between 0 and 1}.'
    )
    msg = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(summary)},
    ]
    try:
        resp = _OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=msg,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        guess = data.get("prediction", "").strip().lower()
        if guess not in {"increase", "decrease"}:
            raise ValueError("Invalid prediction key")
        return guess
    except Exception:
        print("exception is risen")
        return random.choice(["increase", "decrease"])


def run_ei_with_llm_checker(
    lookup: LookupTable,
    *,
    n_init: int = 5,
    n_iter: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    N = lookup.n
    initial_idxs = select_initial_indices(N, n_init, seed)
    seen: set[int] = set(initial_idxs)

    X_obs = lookup.X[initial_idxs]
    Y_obs = lookup.y[initial_idxs]
    observed_indices: List[int] = list(initial_idxs)

    history: List[Dict[str, Any]] = []
    best_so_far = -float("inf")
    for idx in initial_idxs:
        y_i = float(lookup.y[idx].item())
        improved = y_i > best_so_far
        best_so_far = max(best_so_far, y_i)
        history.append(
            {
                "x": {name: float(val) for name, val in zip(lookup.feature_names, lookup.X_raw[idx].tolist())},
                "y": y_i,
                "best_so_far": best_so_far,
                "improved": improved,
            }
        )

    records: List[Dict[str, Any]] = []

    for t in range(n_iter):
        gp = SingleTaskGP(X_obs, Y_obs.unsqueeze(-1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        remaining = remaining_indices(N, seen)
        if not remaining:
            break
        X_pool = lookup.X[remaining]

        with torch.no_grad():
            posterior = gp.posterior(X_obs.unsqueeze(1))
            best_f = float(posterior.mean.max().item())
            ei = ExpectedImprovement(model=gp, best_f=best_f, maximize=True)
            ei_vals = ei(X_pool.unsqueeze(1)).reshape(-1)
            next_local = int(torch.argmax(ei_vals))
            next_idx = remaining[next_local]

            cand_post = gp.posterior(lookup.X[next_idx].unsqueeze(0))
            cand_mean = float(cand_post.mean.item())
            cand_std = float(torch.sqrt(cand_post.variance).item())
            ei_value = float(ei_vals[next_local].item())

        candidate_raw = {
            name: float(val)
            for name, val in zip(lookup.feature_names, lookup.X_raw[next_idx].tolist())
        }
        candidate_norm = {
            name: float(val)
            for name, val in zip(lookup.feature_names, lookup.X[next_idx].tolist())
        }

        best_local_idx = int(torch.argmax(Y_obs).item())
        best_global_idx = observed_indices[best_local_idx]
        best_features = {
            name: float(val)
            for name, val in zip(lookup.feature_names, lookup.X_raw[best_global_idx].tolist())
        }

        recent_history = []
        prev_best = None
        for h in history[-5:]:
            entry = {
                "x": h["x"],
                "y": h["y"],
                "best_after": h["best_so_far"],
                "improved": h.get("improved", False),
            }
            if prev_best is not None:
                entry["delta_best"] = h["best_so_far"] - prev_best
            prev_best = h["best_so_far"]
            recent_history.append(entry)

        summary = {
            "objective": lookup.objective_name,
            "candidate": {
                "features_raw": candidate_raw,
                "features_normalized": candidate_norm,
                "posterior_mean": cand_mean,
                "posterior_std": cand_std,
                "expected_improvement": ei_value,
                "delta_vs_best": cand_mean - best_so_far,
            },
            "incumbent": {"value": best_so_far, "features_raw": best_features},
            "history_tail": recent_history,
            "global_stats": {
                "evaluations_completed": len(history),
                "iterations_remaining": n_iter - t,
            },
        }

        llm_guess = llm_predict_trend(summary, objective_name=lookup.objective_name)

        candidate_raw = {
            name: float(val)
            for name, val in zip(lookup.feature_names, lookup.X_raw[next_idx].tolist())
        }

        y_next = float(lookup.y[next_idx].item())
        improved = y_next > best_so_far

        best_so_far = max(best_so_far, y_next)
        seen.add(next_idx)
        X_obs = torch.cat([X_obs, lookup.X[next_idx].unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, torch.tensor([y_next], dtype=DTYPE, device=DEVICE)], dim=0)
        observed_indices.append(next_idx)

        history.append(
            {
                "x": candidate_raw,
                "y": y_next,
                "best_so_far": best_so_far,
                "improved": improved,
            }
        )

        records.append(
            {
                "iter": t,
                "idx": next_idx,
                "llm_prediction": llm_guess,
                "actual_improved": improved,
                "correct": (llm_guess == ("increase" if improved else "decrease")),
                "y": y_next,
                "best_so_far": best_so_far,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        print("LLM directional accuracy: n/a (no iterations)")
        return df

    overall_accuracy = df["correct"].mean()
    improvement_mask = df["actual_improved"]
    if improvement_mask.any():
        improvement_accuracy = df.loc[improvement_mask, "correct"].mean()
        improvement_rate = improvement_mask.mean()
    else:
        improvement_accuracy = float("nan")
        improvement_rate = 0.0

    print(
        "LLM directional accuracy: {:.2%} overall | {:.2%} of suggestions improved ({} / {}) | "
        "Accuracy on improvement cases: {}".format(
            overall_accuracy,
            improvement_rate,
            improvement_mask.sum(),
            len(df),
            "{:.2%}".format(improvement_accuracy) if improvement_mask.any() else "n/a",
        )
    )
    return df


def plot_trend(df: pd.DataFrame, *, title: Optional[str] = None) -> None:
    """Visualize best objective and cumulative LLM accuracy across iterations."""
    if df.empty:
        print("No optimization history to plot.")
        return

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    ax1.plot(df["iter"], df["best_so_far"], marker="o", label="Best so far")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best objective")
    ax1.grid(True, alpha=0.3)

    accuracy_curve = df["correct"].cumsum() / (df.index + 1)
    ax2 = ax1.twinx()
    ax2.plot(df["iter"], accuracy_curve, color="tab:orange", linestyle="--", label="LLM accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.0)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    if title:
        ax1.set_title(title)
    plt.tight_layout()
    plt.show()
#%%

if __name__ == "__main__":
    lookup = load_lookup_csv("P3HT_dataset.csv", impute_features="median")
    results = run_ei_with_llm_checker(lookup, n_init=3, n_iter=25, seed=45)
    print(results[["iter", "llm_prediction", "actual_improved", "correct"]])
    plot_trend(results, title="EI + LLM Trend")

# %%
