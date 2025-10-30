from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch import Tensor
import numpy as np
import json
import httpx
try:
    from openai import OpenAI
    _LLM_CLIENT = OpenAI(http_client=httpx.Client(verify=False))
except Exception:
    _LLM_CLIENT = None



SYS_PROMPT_SCOUT = """
You are an experimental designer running a short scouting phase BEFORE BO.
Goal: choose the next experiment to rapidly learn the response surface (not pure exploitation).
Choose exactly one goal from: GOAL_A (safe extreme), GOAL_B (vary key dim), GOAL_C (midpoint test),
GOAL_D (two-dim interaction probe), GOAL_E (exploit current best).
Propose x as a length-d vector in [0,1], matching the provided d.
Return STRICT JSON: {"goal":"...", "x":[...], "rationale":"..."} with no extra keys.
"""

# ============================== State ==============================

@dataclass
class ExperimentState:
    """
    Container for the scouting + BO initialization phase.

    Conventions:
    - In lookup-table mode, candidate_pool (N,d) and candidate_y (N,) must be provided.
      candidate_pool is expected to be in the SAME (normalized) space as X_obs.
    - If using continuous bounds mode, provide bounds=[(low,high), ...] in the SAME space
      as the LLM proposals (by default normalized to [0,1]).
    """
    X_obs: Tensor                              # shape (t,d) or empty tensor (0,d)
    y_obs: Tensor                              # shape (t,)
    log: List[Dict[str, Any]] = field(default_factory=list)

    # One of the following two modes:
    bounds: Optional[List[Tuple[float, float]]] = None       # continuous mode
    candidate_pool: Optional[Tensor] = None                  # lookup mode: (N,d)
    candidate_y: Optional[Tensor] = None                     # lookup mode: (N,)
    remaining_idx: Optional[List[int]] = None                # lookup mode: unseen indices

    def device(self) -> torch.device:
        if self.X_obs is not None and self.X_obs.numel() > 0:
            return self.X_obs.device
        if self.candidate_pool is not None:
            return self.candidate_pool.device
        return torch.device("cpu")

    def dtype(self) -> torch.dtype:
        if self.X_obs is not None and self.X_obs.numel() > 0:
            return self.X_obs.dtype
        if self.candidate_pool is not None:
            return self.candidate_pool.dtype
        return torch.float32

    def dim(self) -> int:
        if self.X_obs is not None and self.X_obs.numel() > 0:
            return self.X_obs.shape[-1]
        if self.candidate_pool is not None:
            return self.candidate_pool.shape[-1]
        if self.bounds is not None:
            return len(self.bounds)
        raise ValueError("Cannot infer dimensionality d (provide X_obs with shape (0,d), candidate_pool, or bounds).")


# ============================ Helpers ==============================

def clamp_to_bounds(x: Tensor, bounds: Optional[List[Tuple[float, float]]]) -> Tensor:
    """
    Clamp x (shape (..., d)) to given bounds in the SAME space as x.
    If bounds is None, return x unchanged.
    """
    if bounds is None:
        return x
    lo = torch.tensor([b[0] for b in bounds], device=x.device, dtype=x.dtype)
    hi = torch.tensor([b[1] for b in bounds], device=x.device, dtype=x.dtype)
    return torch.max(lo, torch.min(hi, x))


def pick_closest_unseen_candidate(x_prop: Tensor, state: ExperimentState) -> Tuple[int, Tensor]:
    """
    Snap the proposed point x_prop (1,d) to the nearest AVAILABLE candidate in state.remaining_idx
    (Euclidean distance in feature space). Returns (idx, x_exec=(1,d)).
    Assumes candidate_pool uses the same coordinate system as x_prop (typically normalized [0,1]).
    """
    assert state.candidate_pool is not None and state.remaining_idx is not None, \
        "pick_closest_unseen_candidate requires lookup-table mode (candidate_pool & remaining_idx)."

    # Gather available candidates
    avail_idx = torch.tensor(state.remaining_idx, device=state.candidate_pool.device, dtype=torch.long)
    cand = state.candidate_pool.index_select(0, avail_idx)  # (M,d)
    # Distances to proposed point
    dists = torch.linalg.norm(cand - x_prop.to(cand), dim=1)  # (M,)
    j = int(torch.argmin(dists).item())
    idx = int(avail_idx[j].item())
    x_exec = cand[j].unsqueeze(0)  # (1,d)
    return idx, x_exec


def mark_candidate_used(idx: int, state: ExperimentState) -> None:
    """Remove an index from remaining_idx (no-op if already removed)."""
    if state.remaining_idx is None:
        return
    try:
        state.remaining_idx.remove(idx)
    except ValueError:
        pass


def run_real_experiment(x_next: Tensor) -> float:
    """
    Placeholder for continuous-mode execution. Replace with actual lab dispatch / simulator.
    """
    raise NotImplementedError("run_real_experiment is not implemented for continuous mode.")


# ===================== LLM (deterministic placeholder) =====================

_GOAL_SEQUENCE = ["GOAL_A", "GOAL_B", "GOAL_C", "GOAL_D", "GOAL_E"]

def _propose_by_goal(goal: str, d: int, device: torch.device, dtype: torch.dtype,
                     state: ExperimentState) -> Tensor:
    """
    Deterministic proposal generator in normalized space [0,1] (or same space as bounds/candidate_pool).
    Keeps behavior stable across seeds.
    """
    x = torch.full((1, d), 0.5, device=device, dtype=dtype)

    if goal == "GOAL_A":
        # Explore a "safe extreme" along the presumed key axis x1
        x[..., 0] = 0.9

    elif goal == "GOAL_B":
        # Vary x1 strongly in the opposite direction; others centered
        x[..., 0] = 0.1

    elif goal == "GOAL_C":
        # Probe the center to test monotonic vs peak behavior (already 0.5)
        pass

    elif goal == "GOAL_D":
        # Probe interaction between x1 and x2 (if d>1)
        x[..., 0] = 0.8
        if d > 1:
            x[..., 1] = 0.2

    elif goal == "GOAL_E":
        # Exploit current best: reuse the best observed x (if available), else center
        if state.X_obs is not None and state.X_obs.numel() > 0:
            i_best = int(torch.argmax(state.y_obs).item())
            x = state.X_obs[i_best:i_best+1].to(device=device, dtype=dtype)

    return x

def llm_choose_next_scout_point(state: ExperimentState) -> Dict[str, Any]:
    """
    LLM-backed version. If the API is unavailable, falls back to the deterministic policy.
    Keeps the original signature and return format.
    """
    d = state.dim()
    t = int(state.X_obs.shape[0]) if (state.X_obs is not None and state.X_obs.ndim == 2) else 0
    dev, dt = state.device(), state.dtype()

    # If LLM client not available, use the existing deterministic policy
    if _LLM_CLIENT is None:
        return _deterministic_scout_choice(state)  # reuse your current implementation

    # Minimal, stable context (no huge payloads)
    ctx = {
        "d": d,
        "n_obs": t,
        "goals": ["GOAL_A","GOAL_B","GOAL_C","GOAL_D","GOAL_E"],
        "incumbent": (
            {"x": state.X_obs[int(torch.argmax(state.y_obs))].tolist(),
             "y": float(torch.max(state.y_obs).item())}
            if t > 0 else None
        ),
    }

    messages = [
        {"role": "system", "content": SYS_PROMPT_SCOUT},
        {"role": "user",   "content": json.dumps(ctx)}
    ]

    try:
        resp = _LLM_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=messages,
        )
        obj = json.loads(resp.choices[0].message.content)
        goal = str(obj.get("goal", "GOAL_C"))
        x_prop = torch.tensor(obj.get("x", [0.5]*d), device=dev, dtype=dt).view(1, d)
    except Exception:
        # Robust fallback
        return _deterministic_scout_choice(state)

    # Clamp to bounds if any
    x_prop = clamp_to_bounds(x_prop, state.bounds)

    # Snap in lookup-table mode
    if (state.candidate_pool is not None) and (state.remaining_idx is not None) and len(state.remaining_idx) > 0:
        _, x_exec = pick_closest_unseen_candidate(x_prop, state)
    else:
        x_exec = x_prop

    return {
        "goal": goal,
        "x": x_exec.squeeze(0).detach().cpu().tolist(),
        "rationale": str(obj.get("rationale", "LLM-scouted choice for fast structure learning."))
    }

# helper used above (your existing deterministic logic)
def _deterministic_scout_choice(state: ExperimentState) -> Dict[str, Any]:
    _GOAL_SEQUENCE = ["GOAL_A", "GOAL_B", "GOAL_C", "GOAL_D", "GOAL_E"]
    t = int(state.X_obs.shape[0]) if (state.X_obs is not None and state.X_obs.ndim == 2) else 0
    goal = _GOAL_SEQUENCE[min(t, len(_GOAL_SEQUENCE)-1)]
    x_prop = _propose_by_goal(goal, d=state.dim(), device=state.device(), dtype=state.dtype(), state=state)
    x_prop = clamp_to_bounds(x_prop, state.bounds)
    if (state.candidate_pool is not None) and (state.remaining_idx is not None) and len(state.remaining_idx) > 0:
        _, x_exec = pick_closest_unseen_candidate(x_prop, state)
    else:
        x_exec = x_prop
    rationale_map = {
        "GOAL_A": "Explore a safe extreme on x1 to gauge slope.",
        "GOAL_B": "Vary x1 strongly while others centered.",
        "GOAL_C": "Sample midpoint for monotone vs peak.",
        "GOAL_D": "Probe x1–x2 interaction.",
        "GOAL_E": "Exploit current best."
    }
    return {"goal": goal, "x": x_exec.squeeze(0).cpu().tolist(), "rationale": rationale_map[goal]}

# ============================ Main loop ============================

def run_llm_scout_init(state: ExperimentState, n_scout: int = 5) -> ExperimentState:
    """
    Runs the LLM-Scouted Initialization loop:
      - Deterministic policy (GOAL_A..E) proposes points in normalized space
      - Clamps to bounds if provided
      - In lookup-table mode, snaps to nearest available candidate and uses ground-truth y
      - Appends (x,y) to state and logs the rationale
    Returns the UPDATED state.
    """
    # Ensure X_obs and y_obs exist with correct shapes, even if empty
    d = state.dim()
    dev, dt = state.device(), state.dtype()
    if state.X_obs is None or state.X_obs.numel() == 0:
        state.X_obs = torch.empty((0, d), device=dev, dtype=dt)
    if state.y_obs is None or state.y_obs.numel() == 0:
        state.y_obs = torch.empty((0,), device=dev, dtype=dt)
    if state.log is None:
        state.log = []

    for k in range(n_scout):
        # Ask the "LLM" (deterministic policy) for the next point
        choice = llm_choose_next_scout_point(state)

        x_exec = torch.tensor(choice["x"], device=dev, dtype=dt).view(1, d)

        # Evaluate
        if (state.candidate_pool is not None) and (state.remaining_idx is not None) and (state.candidate_y is not None):
            # Re-identify the nearest AVAILABLE index to log / mark used (cheap & robust)
            if len(state.remaining_idx) == 0:
                raise RuntimeError("No remaining candidates in lookup-table mode during scouting.")
            idx, x_exec_snapped = pick_closest_unseen_candidate(x_exec, state)
            x_exec = x_exec_snapped  # ensure executed x is exactly the candidate
            y_next = float(state.candidate_y[idx].item())
            mark_candidate_used(idx, state)
        else:
            # Continuous mode: user must implement the real experiment
            y_next = float(run_real_experiment(x_exec))

        # Append observation
        state.X_obs = torch.cat([state.X_obs, x_exec], dim=0)
        state.y_obs = torch.cat([state.y_obs, torch.tensor([y_next], device=dev, dtype=dt)], dim=0)

        # Log
        state.log.append({
            "iter_id": k,
            "goal": choice["goal"],
            "x": x_exec.squeeze(0).detach().cpu().tolist(),
            "y": float(y_next),
            "rationale": choice.get("rationale", "")
        })

    return state
