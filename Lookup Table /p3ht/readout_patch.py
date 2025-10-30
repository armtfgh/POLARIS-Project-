"""
Patch-based, stateful LLM readout for Hybrid BO
------------------------------------------------
This module lets the LLM *edit* (patch) the prior readout instead of replacing it
from scratch every iteration. You keep a small memory (ledger) of recent readouts
and outcomes; the LLM receives (prev_readout, ledger_tail, posterior context,
budget/limits) and returns a small set of ops (set_effect, add/retune/remove_bump,
set/clear_interaction, set_m0_weight) plus a rationale.

You can override the default system prompt by passing `sys_prompt=...` from your
external `bo_readout_prompts.py`.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import copy
import json

# ------------------------- Defaults -------------------------
DEFAULT_PATCH_LIMITS: Dict[str, Any] = {
    "step_scale_max": 0.20,      # max absolute change to effect.scale per op
    "step_conf_max":  0.20,      # max absolute change to effect.confidence per op
    "range_expand_max": 0.20,    # max increase of range width per op
    "sigma_bounds":   [0.05, 0.40],
    "amp_bounds":     [0.0, 0.50],
}

DEFAULT_BASE_BUDGET = 2   # ops per iteration normally
DEFAULT_STUCK_BUDGET = 5  # ops when stagnating / misaligned

# A compact, strongly-instructional default prompt (override recommended)
SYS_PROMPT_READOUT_PATCH_V1 = (
    "You improve a 2D prior for Bayesian Optimization over x1,x2 in [0,1].\n"
    "Each turn you get: the previous readout, a short memory of outcomes, and a\n"
    "posterior summary (incumbent, top-EI cells, high-variance cells, density).\n\n"
    "Output STRICT JSON with keys: ops (list), optional m0_weight (0..1), optional\n"
    "trust_region ([[l1,h1],[l2,h2]]), and rationale (short text).\n\n"
    "Allowed ops (each as an object with an 'op' field):\n"
    "- set_effect: {op:'set_effect', var:'x1'|'x2', effect:'increase'|'decrease'|\n"
    "   'nonmonotone-peak'|'nonmonotone-valley'|'flat', scale:0..1, confidence:0..1,\n"
    "   range_hint:[lo,hi]}\n"
    "- set_interaction: {op:'set_interaction', type:'synergy'|'antagonism', confidence:0..1}\n"
    "- clear_interaction: {op:'clear_interaction'}\n"
    "- add_bump:    {op:'add_bump',    mu:[x1,x2], sigma:0.05..0.40, amp:0..0.50}\n"
    "- retune_bump: {op:'retune_bump', index:int, mu:[x1,x2], sigma:0.05..0.40, amp:0..0.50}\n"
    "- remove_bump: {op:'remove_bump', index:int}\n"
    "- set_m0_weight: {op:'set_m0_weight', target:0..1}\n\n"
    "Guidelines: prefer small edits; respect the change budget and step limits.\n"
    "When improvement stalled or prior misaligned, consider widening ranges,\n"
    "reducing confidence, or adding one bump near promising EI/variance cells.\n"
)

# ------------------------- Ledger (memory) -------------------------

def init_ledger() -> Dict[str, Any]:
    return {"entries": []}


def log_entry(
    ledger: Dict[str, Any],
    *,
    t: int,
    readout: Dict[str, Any],
    alpha_ls: float,
    rho: float,
    best_so_far: float,
    improvement: float,
    chosen_x: List[float],
    y_next: float,
    ei_summary: Optional[Dict[str, Any]] = None,
    patch: Optional[Dict[str, Any]] = None,
    rationale: Optional[str] = None,
    patch_magnitude: Optional[float] = None,
) -> None:
    ledger["entries"].append({
        "t": t,
        "readout": copy.deepcopy(readout),
        "alpha_ls": float(alpha_ls),
        "rho": float(rho),
        "best_so_far": float(best_so_far),
        "improvement": float(improvement),
        "chosen_x": list(map(float, chosen_x)),
        "y_next": float(y_next),
        "ei_summary": ei_summary or {},
        "patch": patch or {},
        "rationale": rationale or "",
        "patch_magnitude": float(patch_magnitude) if patch_magnitude is not None else None,
    })


def ledger_tail(ledger: Dict[str, Any], K: int = 3) -> List[Dict[str, Any]]:
    return ledger["entries"][-K:]

# --------------------- Patch application -------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _ensure_effect(ro: Dict[str, Any], var: str) -> None:
    ro.setdefault("effects", {})
    ro["effects"].setdefault(var, {"effect": "flat", "scale": 0.0, "confidence": 0.0})


def _norm_range_hint(rh: Optional[List[float]]) -> Optional[List[float]]:
    if not isinstance(rh, (list, tuple)) or len(rh) != 2:
        return None
    lo, hi = float(rh[0]), float(rh[1])
    lo, hi = _clamp(lo), _clamp(hi)
    if hi < lo:
        lo, hi = hi, lo
    return [lo, hi]


def _bump_bounds(lims: Dict[str, Any]):
    sb = lims.get("sigma_bounds", [0.05, 0.40])
    ab = lims.get("amp_bounds", [0.0, 0.50])
    return (float(sb[0]), float(sb[1])), (float(ab[0]), float(ab[1]))


def compute_patch_magnitude(prev: Dict[str, Any], nxt: Dict[str, Any]) -> float:
    mag = 0.0
    # effects
    for var in ["x1", "x2"]:
        pe = prev.get("effects", {}).get(var, {})
        ne = nxt.get("effects", {}).get(var, {})
        mag += abs(float(ne.get("scale", 0.0)) - float(pe.get("scale", 0.0)))
        mag += abs(float(ne.get("confidence", 0.0)) - float(pe.get("confidence", 0.0)))
        prh = pe.get("range_hint"); nrh = ne.get("range_hint")
        if isinstance(prh, (list, tuple)) and len(prh) == 2 and isinstance(nrh, (list, tuple)) and len(nrh) == 2:
            pw = prh[1] - prh[0]; nw = nrh[1] - nrh[0]
            mag += abs(float(nw - pw))
        if pe.get("effect") != ne.get("effect"):
            mag += 0.5
    # bumps count contributes
    pb = prev.get("bumps", []) or []; nb = nxt.get("bumps", []) or []
    mag += abs(len(nb) - len(pb)) * 0.5
    return mag


def apply_readout_patch(
    prev_readout: Dict[str, Any],
    patch: Dict[str, Any],
    *,
    limits: Dict[str, Any] = DEFAULT_PATCH_LIMITS,
    max_ops: int = DEFAULT_BASE_BUDGET,
    allow_flip: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (next_readout, meta) where meta includes applied ops and suggested m0_weight/trust_region.
    Guardrails: clamp values; limit step sizes; enforce op budget.
    """
    ro = copy.deepcopy(prev_readout)
    ro.setdefault("effects", {})
    ro.setdefault("interactions", [])
    ro.setdefault("bumps", [])

    ops = list(patch.get("ops", []) or [])
    applied: List[Dict[str, Any]] = []

    step_scale = float(limits.get("step_scale_max", 0.20))
    step_conf  = float(limits.get("step_conf_max", 0.20))
    range_expand = float(limits.get("range_expand_max", 0.20))
    (sig_lo, sig_hi), (amp_lo, amp_hi) = _bump_bounds(limits)

    # Helper: change with step cap
    def _bounded_delta(old: float, target: float, step: float) -> float:
        delta = max(-step, min(step, float(target) - float(old)))
        return _clamp(float(old) + delta)

    # Interaction helpers
    def _set_interaction(it_type: str, conf: float):
        ro["interactions"] = [{"pair": ["x1", "x2"], "type": it_type, "confidence": _clamp(conf)}]

    def _clear_interaction():
        ro["interactions"] = []

    # Ensure bumps are a list of dicts with indices
    bumps = ro.get("bumps", [])

    # Apply up to max_ops
    for op in ops[:max_ops]:
        kind = op.get("op")
        if kind == "set_effect":
            var = op.get("var")
            if var not in ("x1", "x2"):
                continue
            _ensure_effect(ro, var)
            eff_tgt = op.get("effect", "flat")
            if eff_tgt not in ("increase", "decrease", "nonmonotone-peak", "nonmonotone-valley", "flat"):
                eff_tgt = "flat"
            eff_cur = ro["effects"][var].get("effect", "flat")
            if eff_tgt != eff_cur and not allow_flip and eff_cur != "flat":
                # skip flips unless allowed; still allow move from flat -> something
                eff_tgt = eff_cur
            ro["effects"][var]["effect"] = eff_tgt
            # scale/conf bounded steps
            sc_old = float(ro["effects"][var].get("scale", 0.0))
            cf_old = float(ro["effects"][var].get("confidence", 0.0))
            sc_new = _bounded_delta(sc_old, _clamp(op.get("scale", sc_old)), step_scale)
            cf_new = _bounded_delta(cf_old, _clamp(op.get("confidence", cf_old)), step_conf)
            ro["effects"][var]["scale"] = sc_new
            ro["effects"][var]["confidence"] = cf_new
            # range hint widen with cap
            rh_tgt = _norm_range_hint(op.get("range_hint"))
            if rh_tgt is not None:
                rh_old = _norm_range_hint(ro["effects"][var].get("range_hint")) or [0.0, 1.0]
                old_w = rh_old[1] - rh_old[0]
                tgt_w = rh_tgt[1] - rh_tgt[0]
                # limit widening amount
                max_w = min(1.0, old_w + range_expand)
                new_w = min(tgt_w, max_w)
                # center toward target center but keep inside [0,1]
                c_tgt = 0.5 * (rh_tgt[0] + rh_tgt[1])
                lo = _clamp(c_tgt - 0.5 * new_w); hi = _clamp(c_tgt + 0.5 * new_w)
                if hi < lo:
                    lo, hi = hi, lo
                ro["effects"][var]["range_hint"] = [lo, hi]
            applied.append(op)

        elif kind == "set_interaction":
            _set_interaction(op.get("type", "synergy"), float(op.get("confidence", 0.0)))
            applied.append(op)

        elif kind == "clear_interaction":
            _clear_interaction(); applied.append(op)

        elif kind == "add_bump":
            mu = op.get("mu", [0.5, 0.5])
            sig = _clamp(op.get("sigma", 0.15), sig_lo, sig_hi)
            amp = _clamp(op.get("amp", 0.1), amp_lo, amp_hi)
            bumps.append({"mu": [ _clamp(mu[0]), _clamp(mu[1]) ], "sigma": sig, "amp": amp})
            ro["bumps"] = bumps; applied.append(op)

        elif kind == "retune_bump":
            idx = int(op.get("index", -1))
            if 0 <= idx < len(bumps):
                b = bumps[idx]
                if "mu" in op:
                    mu = op["mu"]; b["mu"] = [ _clamp(mu[0]), _clamp(mu[1]) ]
                if "sigma" in op:
                    b["sigma"] = _clamp(op["sigma"], sig_lo, sig_hi)
                if "amp" in op:
                    b["amp"] = _clamp(op["amp"], amp_lo, amp_hi)
                applied.append(op)

        elif kind == "remove_bump":
            idx = int(op.get("index", -1))
            if 0 <= idx < len(bumps):
                bumps.pop(idx); ro["bumps"] = bumps; applied.append(op)

        elif kind == "set_m0_weight":
            # handled as advisory in meta
            applied.append(op)

    meta = {
        "applied_ops": applied,
        "m0_weight": patch.get("m0_weight", None),
        "trust_region": patch.get("trust_region", None),
        "rationale": patch.get("rationale", ""),
    }

    return ro, meta

# --------------------- LLM patch generator ----------------------

def llm_generate_patch(
    *,
    openai_client,
    prev_readout: Dict[str, Any],
    ledger_tail_entries: List[Dict[str, Any]],
    context: Dict[str, Any],
    budget: Dict[str, Any],  # {max_ops:int, allow_flip:bool}
    limits: Dict[str, Any] = DEFAULT_PATCH_LIMITS,
    sys_prompt: str = SYS_PROMPT_READOUT_PATCH_V1,
    few_shots: Optional[List[Dict[str, Any]]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    payload = {
        "prev_readout": prev_readout,
        "memory": ledger_tail_entries,
        "posterior_context": context,
        "budget": {"max_ops": int(budget.get("max_ops", DEFAULT_BASE_BUDGET)),
                    "allow_flip": bool(budget.get("allow_flip", False))},
        "limits": limits,
        "rules": {
            "effects": ["increase", "decrease", "nonmonotone-peak", "nonmonotone-valley", "flat"],
            "vars": ["x1", "x2"],
            "keep_json_strict": True,
        },
    }

    messages = [{"role": "system", "content": sys_prompt}]
    if few_shots:
        for ex in few_shots:
            messages.append({"role": "user", "content": json.dumps(ex.get("user", {}))})
            messages.append({"role": "assistant", "content": json.dumps(ex.get("assistant", {}))})
    messages.append({"role": "user", "content": json.dumps(payload)})

    resp = openai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=messages,
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {"ops": [], "rationale": "fallback-empty"}
    # guarantee structure
    data.setdefault("ops", [])
    return data
