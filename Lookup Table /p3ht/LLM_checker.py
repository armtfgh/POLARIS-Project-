from __future__ import annotations

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
    history: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    *,
    feature_names: List[str],
    objective_name: str,
) -> str:
    if _OPENAI_CLIENT is None:
        return random.choice(["increase", "decrease"])

    prompt = (
        "You are an analytical scientist helping a Bayesian optimizer. "
        "Given the historical experiments (in original units) and the next candidate "
        "the optimizer plans to evaluate, predict whether the new measurement will "
        "IMPROVE (increase) or WORSEN (decrease) the best observed objective so far. "
        "Respond strictly as JSON: {\"prediction\": \"increase\" or \"decrease\", \"confidence\": 0.0-1.0}."
    )
    payload = {
        "objective": objective_name,
        "features": feature_names,
        "history": history,
        "next_candidate": candidate,
    }

    msg = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload)},
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

    history: List[Dict[str, Any]] = []
    best_so_far = float(Y_obs.max().item())

    for idx in initial_idxs:
        history.append(
            {
                "x": {name: float(val) for name, val in zip(lookup.feature_names, lookup.X_raw[idx].tolist())},
                "y": float(lookup.y[idx].item()),
                "best_so_far": best_so_far,
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

        candidate_raw = {
            name: float(val)
            for name, val in zip(lookup.feature_names, lookup.X_raw[next_idx].tolist())
        }
        llm_guess = llm_predict_trend(
            history,
            {"features": candidate_raw, "predicted_best": best_so_far},
            feature_names=lookup.feature_names,
            objective_name=lookup.objective_name,
        )

        y_next = float(lookup.y[next_idx].item())
        improved = y_next > best_so_far

        best_so_far = max(best_so_far, y_next)
        seen.add(next_idx)
        X_obs = torch.cat([X_obs, lookup.X[next_idx].unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, torch.tensor([y_next], dtype=DTYPE, device=DEVICE)], dim=0)

        history.append(
            {
                "x": candidate_raw,
                "y": y_next,
                "best_so_far": best_so_far,
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
    accuracy = df["correct"].mean() if not df.empty else float("nan")
    print(f"LLM directional accuracy: {accuracy:.2%} over {len(df)} iterations.")
    return df


if __name__ == "__main__":
    lookup = load_lookup_csv("P3HT_dataset.csv", impute_features="median")
    results = run_ei_with_llm_checker(lookup, n_init=5, n_iter=25, seed=42)
    print(results[["iter", "llm_prediction", "actual_improved", "correct"]])
