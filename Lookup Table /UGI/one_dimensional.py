#!/usr/bin/env python3
"""1-D demo that shows how a text-shaped prior bends a GP mean function."""
#%%
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Optional, Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from prior_gp import Prior, alignment_on_obs
from readout_schema import readout_to_prior

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

PRESET_HINTS = {
    "flat": "No prior knowledge. Assume the function is flat across the domain.",
    "left_peak": (
        "There should be a pronounced peak near x=0.25 with high confidence, "
        "and values should fall off away from that window."
    ),
    "right_peak": (
        "Expect a strong increase that tops out around x=0.75 and then drops, "
        "forming a sharp bump near that region."
    ),
    "monotone": (
        "The response should gently increase from left to right without any bump, "
        "with only weak confidence."
    ),
}

ALLOWED_EFFECTS = {
    "flat",
    "increase",
    "increasing",
    "decrease",
    "decreasing",
    "nonmonotone-peak",
    "peak",
    "nonmonotone-valley",
    "valley",
}

COLORS = {
    "baseline": "#1f77b4",  # blue
    "shaped": "#ff7f0e",  # orange
    "obs": "#d62728",  # red
    "prior": "#2ca02c",  # green
}


def true_function(x: torch.Tensor) -> torch.Tensor:
    """Ground-truth 1-D function (kept secret from the prior)."""
    wave = 0.55 * torch.sin(2.0 * math.pi * x) + 0.25 * torch.cos(5.0 * math.pi * x)
    bump = 0.6 * torch.exp(-30.0 * (x - 0.3) ** 2)
    tail = 0.35 * torch.exp(-50.0 * (x - 0.8) ** 2)
    return wave + bump + tail


def sample_training_points(n: int, noise_sd: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick evenly spaced 1-D points with slight jitter and evaluate the function."""
    torch.manual_seed(seed)
    base = torch.linspace(0.05, 0.95, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    jitter = 0.03 * torch.randn_like(base)
    train_x = torch.clamp(base + jitter, 0.0, 1.0)
    clean_y = true_function(train_x)
    if noise_sd <= 0.0:
        noisy_y = clean_y.clone()
    else:
        noisy_y = clean_y + noise_sd * torch.randn_like(clean_y)
    return train_x, noisy_y


def parse_text_hint(text: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Convert a loose natural-language hint into a readout dictionary."""
    lower = text.lower()
    effect = "flat"
    if any(k in lower for k in {"peak", "bump", "hill"}):
        effect = "nonmonotone-peak"
    elif any(k in lower for k in {"valley", "dip", "trough"}):
        effect = "nonmonotone-valley"
    elif any(k in lower for k in {"increase", "rising", "uphill"}):
        effect = "increase"
    elif any(k in lower for k in {"decrease", "falling", "downhill"}):
        effect = "decrease"

    scale = 0.6
    confidence = 0.65
    if "strong" in lower or "pronounced" in lower:
        scale, confidence = 0.9, 0.9
    elif "weak" in lower or "gentle" in lower or "soft" in lower:
        scale, confidence = 0.35, 0.4

    width = 0.25
    if "narrow" in lower or "sharp" in lower:
        width = 0.15
    elif "broad" in lower or "wide" in lower:
        width = 0.4

    center = 0.5
    numbers = [float(match) for match in re.findall(r"\d*\.?\d+", lower)]
    for value in numbers:
        if 0.0 <= value <= 1.0:
            center = value
            break

    lo = max(center - width / 2.0, 0.0)
    hi = min(center + width / 2.0, 1.0)

    effects = {
        "x1": {
            "effect": effect,
            "scale": scale,
            "confidence": confidence,
            "range_hint": [lo, hi],
        }
    }

    bumps = []
    if effect in {"nonmonotone-peak", "nonmonotone-valley"}:
        ampsign = 1.0 if effect == "nonmonotone-peak" else -1.0
        bumps.append({"mu": [center], "sigma": [max(width / 3.0, 0.05)], "amp": 0.2 * ampsign})

    return {"effects": effects, "interactions": [], "bumps": bumps}


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip()


def _extract_json_object(text: str) -> str:
    t = _strip_code_fences(text)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM output does not contain a JSON object.")
    return t[start : end + 1]


def sanitize_readout_1d(readout: Dict[str, Any]) -> Dict[str, Any]:
    """Make a best-effort 1-D readout usable by Prior."""
    ro: Dict[str, Any] = dict(readout or {})
    effects = dict(ro.get("effects") or {})
    spec = dict(effects.get("x1") or {})

    eff = str(spec.get("effect", "flat")).strip().lower()
    if eff not in ALLOWED_EFFECTS:
        eff = "flat"
    scale = float(spec.get("scale", 0.0) or 0.0)
    conf = float(spec.get("confidence", 0.0) or 0.0)
    scale = float(np.clip(scale, 0.0, 1.0))
    conf = float(np.clip(conf, 0.0, 1.0))

    rh = spec.get("range_hint", [0.0, 1.0])
    if isinstance(rh, (list, tuple)) and len(rh) == 2:
        lo, hi = float(rh[0]), float(rh[1])
    else:
        lo, hi = 0.0, 1.0
    lo, hi = float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))
    if hi < lo:
        lo, hi = hi, lo

    effects["x1"] = {"effect": eff, "scale": scale, "confidence": conf, "range_hint": [lo, hi]}
    ro["effects"] = effects
    ro["interactions"] = ro.get("interactions") or []
    ro["bumps"] = ro.get("bumps") or []
    return ro


def llm_generate_readout(
    *,
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 600,
    api_key: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Call an LLM to produce a JSON readout for a 1-D prior."""
    try:
        import httpx  # type: ignore
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LLM mode requires the `openai` and `httpx` packages to be installed."
        ) from exc

    client = OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))

    system = (
        "You design prior means for Bayesian optimization.\n"
        "We have ONE normalized variable x1 in [0,1].\n"
        "Return STRICT JSON (no markdown, no prose) with this schema:\n"
        "{\n"
        '  "effects": {\n'
        '    "x1": {"effect": "<flat|increase|decrease|nonmonotone-peak|nonmonotone-valley>",\n'
        '           "scale": <0..1>, "confidence": <0..1>, "range_hint": [<0..1>, <0..1>]}\n'
        "  },\n"
        '  "interactions": [],\n'
        '  "bumps": [{"mu":[<0..1>], "sigma":[<0..1>], "amp": <float>}] \n'
        "}\n"
        "Use bumps only if helpful; otherwise return an empty list.\n"
    )

    user = (
        "Design a prior readout consistent with this user hint:\n"
        f"{prompt.strip()}\n"
        "Remember: output only JSON."
    )

    # Support both legacy and newer SDK shapes by trying a couple call styles.
    raw_text: Optional[str] = None
    try:  # chat.completions
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        raw_text = resp.choices[0].message.content
    except Exception:
        # responses API fallback
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        raw_text = getattr(resp, "output_text", None)

    if not raw_text:
        raise RuntimeError("LLM call returned empty text.")

    json_text = _extract_json_object(raw_text)
    try:
        ro = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM JSON: {exc}\nRaw:\n{raw_text}") from exc
    return sanitize_readout_1d(ro), raw_text


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_f: float, *, xi: float = 0.0) -> np.ndarray:
    """Analytic EI for Normal(mu, sigma^2)."""
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)
    imp = mu - float(best_f) - float(xi)
    z = imp / sigma_safe
    phi = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z**2)
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    ei = imp * Phi + sigma_safe * phi
    ei[sigma <= 1e-12] = np.maximum(imp[sigma <= 1e-12], 0.0)
    return np.maximum(ei, 0.0)


def set_manuscript_style() -> None:
    """Matplotlib rcParams tuned for manuscript figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 350,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
        }
    )

def fit_plain_gp(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    """Standard GP trained with a flat prior."""
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def fit_text_prior_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    prior: Prior,
    *,
    prior_strength: float = 1.0,
    prior_scale_mode: str = "learned",  # "learned" | "fixed"
    rho_floor: float = 0.05,
    alpha_max: float = 2.0,
    allow_prior_inversion: bool = False,
) -> Tuple[SingleTaskGP, float, Dict[str, float]]:
    """Fit a residual GP and return the model plus scaled prior weight."""
    if prior_strength <= 0.0:
        # User requested no prior influence; fall back to plain GP on raw targets.
        model = fit_plain_gp(train_x, train_y)
        return model, 0.0, {"alpha_raw": 0.0, "alpha_used": 0.0, "rho": 0.0, "rho_weight": 0.0, "m0_scale": 0.0}

    X = train_x
    Y = train_y
    m0 = prior.m0_torch(X).reshape(-1)
    yv = Y.reshape(-1)
    m0c = m0 - m0.mean()
    yc = yv - yv.mean()

    rho = float(alignment_on_obs(X, Y, prior))
    rho_weight = max(abs(rho), float(rho_floor))

    denom = torch.dot(m0c, m0c).item()
    alpha_raw = (torch.dot(m0c, yc).item() / (denom + 1e-12)) if denom > 0 else 0.0

    mode = str(prior_scale_mode or "learned").lower()
    if mode == "fixed":
        alpha_used = 1.0
        m0_scale = float(prior_strength)
    else:
        alpha_used = float(alpha_raw)
        if not allow_prior_inversion:
            # Keep the sign consistent with the prior's own sign; do not invert it from scarce data.
            alpha_used = abs(alpha_used)
        if alpha_max is not None and alpha_max > 0:
            alpha_used = float(np.clip(alpha_used, -float(alpha_max), float(alpha_max)))

        # Apply prior_strength consistently: it affects BOTH the training residualization and the add-back at prediction.
        m0_scale = float(prior_strength) * float(alpha_used) * float(rho_weight)

    Y_resid = Y - m0_scale * m0.unsqueeze(-1)

    gp_resid = SingleTaskGP(X, Y_resid)
    mll = ExactMarginalLogLikelihood(gp_resid.likelihood, gp_resid)
    fit_gpytorch_mll(mll)
    gp_resid.eval()
    debug = {
        "alpha_raw": float(alpha_raw),
        "alpha_used": float(alpha_used),
        "rho": float(rho),
        "rho_weight": float(rho_weight),
        "m0_scale": float(m0_scale),
        "prior_scale_mode": 1.0 if mode == "fixed" else 0.0,
    }
    return gp_resid, m0_scale, debug


def posterior_stats(
    model: torch.nn.Module,
    grid: torch.Tensor,
    *,
    prior: Optional[Prior] = None,
    prior_scale: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return posterior mean and standard deviation over a grid as numpy arrays."""
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4):
        posterior = model.posterior(grid)
        mean = posterior.mean.squeeze(-1)
        if prior is not None:
            mean = mean + prior_scale * prior.m0_torch(grid).reshape(mean.shape)
        std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)
    return mean.cpu().numpy(), std.cpu().numpy()


def plot_gp(
    ax: plt.Axes,
    x_grid: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    true_curve: np.ndarray,
    title: str,
    prior_curve: Optional[np.ndarray] = None,
    y_limits: Optional[Tuple[float, float]] = None,
) -> None:
    """Common plotting helper."""
    ax.plot(x_grid, true_curve, color="k", lw=1.2, label="True function")
    ax.plot(x_grid, mean, color="#4C72B0", lw=2.0, label="Posterior mean")
    ax.fill_between(x_grid, mean - 2 * std, mean + 2 * std, color="#4C72B0", alpha=0.2, label="±2σ")
    ax.scatter(train_x, train_y, color="#C44E52", edgecolors="k", zorder=5, label="Observations")
    if prior_curve is not None:
        ax.plot(x_grid, prior_curve, color="#55A868", lw=1.5, ls="--", label="Prior mean $m_0(x)$")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0.0, 1.0)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)


def _compute_for_demo(
    *,
    n_train: int,
    noise: float,
    seed: int,
    hint: Optional[str],
    preset: str,
    use_llm: bool,
    llm_prompt: Optional[str],
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_api_key: Optional[str],
    prior_strength: float,
    prior_scale_mode: str,
    alpha_max: float,
    allow_prior_inversion: bool,
    grid_n: int,
    acq_xi: float,
    verbose: bool,
) -> Dict[str, object]:
    if hint is None:
        if preset not in PRESET_HINTS:
            raise ValueError(f"Unknown preset '{preset}'. Valid options: {sorted(PRESET_HINTS)}")
        text_hint = PRESET_HINTS[preset]
    else:
        text_hint = hint

    train_x, train_y = sample_training_points(n_train, noise, seed)
    train_x = train_x.to(DEVICE)
    train_y = train_y.to(DEVICE)

    model_flat = fit_plain_gp(train_x, train_y)

    llm_raw: Optional[str] = None
    readout_source = "heuristic"
    prompt_used = text_hint
    if use_llm:
        readout_source = "llm"
        prompt_text = (llm_prompt or text_hint).strip()
        prompt_used = prompt_text
        readout, llm_raw = llm_generate_readout(
            prompt=prompt_text,
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            api_key=llm_api_key,
        )
    else:
        readout = sanitize_readout_1d(parse_text_hint(text_hint))

    prior = readout_to_prior(readout, feature_names=["x1"])
    gp_resid, prior_scale, prior_debug = fit_text_prior_gp(
        train_x,
        train_y,
        prior,
        prior_strength=prior_strength,
        prior_scale_mode=prior_scale_mode,
        alpha_max=alpha_max,
        allow_prior_inversion=allow_prior_inversion,
    )

    grid = torch.linspace(0.0, 1.0, int(grid_n), device=DEVICE, dtype=DTYPE).unsqueeze(-1)
    x_np = grid.squeeze(-1).cpu().numpy()
    true_np = true_function(grid).squeeze(-1).cpu().numpy()

    mean_flat, std_flat = posterior_stats(model_flat, grid)
    mean_resid, std_resid = posterior_stats(gp_resid, grid)
    mean_text, std_text = posterior_stats(gp_resid, grid, prior=prior, prior_scale=prior_scale)

    m0_grid = prior.m0_torch(grid).squeeze(-1).cpu().numpy()
    m0_scaled = prior_scale * m0_grid

    train_x_np = train_x.squeeze(-1).cpu().numpy()
    train_y_np = train_y.squeeze(-1).cpu().numpy()
    m0_train = prior.m0_torch(train_x).reshape(-1).detach().cpu().numpy()
    y_resid_train = train_y_np - prior_scale * m0_train

    best_f = float(np.max(train_y_np)) if train_y_np.size else float("-inf")
    ei_flat = expected_improvement(mean_flat, std_flat, best_f, xi=acq_xi)
    ei_text = expected_improvement(mean_text, std_text, best_f, xi=acq_xi)
    next_x_flat = float(x_np[int(np.argmax(ei_flat))])
    next_x_text = float(x_np[int(np.argmax(ei_text))])

    rh_lo, rh_hi = 0.0, 1.0
    try:
        rh = ((readout.get("effects") or {}).get("x1") or {}).get("range_hint")
        if isinstance(rh, (list, tuple)) and len(rh) == 2:
            rh_lo, rh_hi = float(rh[0]), float(rh[1])
            rh_lo, rh_hi = float(np.clip(rh_lo, 0.0, 1.0)), float(np.clip(rh_hi, 0.0, 1.0))
            if rh_hi < rh_lo:
                rh_lo, rh_hi = rh_hi, rh_lo
    except Exception:
        pass

    if verbose:
        print("=== Hint ===")
        print(text_hint)
        print(f"\n=== Readout ({readout_source}) ===")
        print(json.dumps(readout, indent=2))
        if llm_raw is not None:
            print("\n=== LLM Raw Output ===")
            print(llm_raw)

    readout_json = json.dumps(readout, indent=2)

    return {
        "grid": x_np,
        "true_curve": true_np,
        "train_x": train_x_np,
        "train_y": train_y_np,
        "train_y_resid": y_resid_train,
        "mean_flat": mean_flat,
        "std_flat": std_flat,
        "mean_resid": mean_resid,
        "std_resid": std_resid,
        "mean_text": mean_text,
        "std_text": std_text,
        "m0_grid": m0_grid,
        "m0_scaled": m0_scaled,
        "ei_flat": ei_flat,
        "ei_text": ei_text,
        "next_x_flat": next_x_flat,
        "next_x_text": next_x_text,
        "best_f": best_f,
        "hint_text": text_hint,
        "prompt_used": prompt_used,
        "readout": readout,
        "readout_json": readout_json,
        "readout_source": readout_source,
        "llm_raw": llm_raw,
        "prior_scale": prior_scale,
        "prior_strength": prior_strength,
        "prior_scale_mode": prior_scale_mode,
        "prior_debug": prior_debug,
        "range_hint": (rh_lo, rh_hi),
    }


def run_demo_(
    *,
    n_train: int = 8,
    noise: float = 0.03,
    seed: int = 0,
    hint: Optional[str] = None,
    preset: str = "left_peak",
    use_llm: bool = False,
    llm_prompt: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 600,
    llm_api_key: Optional[str] = None,
    prior_strength: float = 1.0,
    prior_scale_mode: str = "learned",
    alpha_max: float = 2.0,
    allow_prior_inversion: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the 1-D prior demo and return a payload that is notebook-friendly.

    Example (Jupyter):
        from one_dimensional import run_demo
        result = run_demo(preset="right_peak", show=False)
        display(result["fig"])
    """
    payload = _compute_for_demo(
        n_train=n_train,
        noise=noise,
        seed=seed,
        hint=hint,
        preset=preset,
        use_llm=use_llm,
        llm_prompt=llm_prompt,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_api_key=llm_api_key,
        prior_strength=prior_strength,
        prior_scale_mode=prior_scale_mode,
        alpha_max=alpha_max,
        allow_prior_inversion=allow_prior_inversion,
        grid_n=400,
        acq_xi=0.0,
        verbose=verbose,
    )

    x_np = payload["grid"]  # type: ignore[assignment]
    true_np = payload["true_curve"]  # type: ignore[assignment]
    train_x_np = payload["train_x"]  # type: ignore[assignment]
    train_y_np = payload["train_y"]  # type: ignore[assignment]
    mean_flat = payload["mean_flat"]  # type: ignore[assignment]
    std_flat = payload["std_flat"]  # type: ignore[assignment]
    mean_text = payload["mean_text"]  # type: ignore[assignment]
    std_text = payload["std_text"]  # type: ignore[assignment]
    prior_curve = payload["m0_grid"]  # type: ignore[assignment]
    y_limits = (float(true_np.min()) - 0.2, float(true_np.max()) + 0.2)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    plot_gp(
        axes[0],
        x_np,
        mean_flat,
        std_flat,
        train_x_np,
        train_y_np,
        true_np,
        "Baseline GP (flat prior)",
        y_limits=y_limits,
    )
    plot_gp(
        axes[1],
        x_np,
        mean_text,
        std_text,
        train_x_np,
        train_y_np,
        true_np,
        "Text-shaped prior",
        prior_curve=prior_curve,
        y_limits=y_limits,
    )
    fig.suptitle("Effect of language-shaped priors on a 1-D GP", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=220)
        if verbose:
            print(f"\nSaved figure to {save_path}")

    if show:
        plt.show()

    return {
        "fig": fig,
        "axes": axes,
        "grid": x_np,
        "true_curve": true_np,
        "train_x": train_x_np,
        "train_y": train_y_np,
        "mean_flat": mean_flat,
        "std_flat": std_flat,
        "mean_text": mean_text,
        "std_text": std_text,
        "prior_curve": prior_curve,
        "readout": payload["readout"],
        "readout_source": payload["readout_source"],
        "llm_raw": payload["llm_raw"],
        "prior_scale": payload["prior_scale"],
        "prior_strength": payload["prior_strength"],
        "prior_scale_mode": payload["prior_scale_mode"],
        "prior_debug": payload["prior_debug"],
        "hint_text": payload["hint_text"],
    }


# out = run_demo_(
#     n_train=3,
#     noise=0.0,
#     use_llm=True,
#     llm_prompt="the trend of x is increasing with high confidence",
#     llm_model="gpt-4o-mini",
#     prior_strength=10,
#     show=False,
# )
#%%

# Backwards-compatible alias (older notebooks may import run_demo)
run_demo = run_demo_


def run_demo_multipanel_(
    *,
    n_train: int = 8,
    noise: float = 0.03,
    seed: int = 0,
    hint: Optional[str] = None,
    preset: str = "left_peak",
    use_llm: bool = False,
    llm_prompt: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 600,
    llm_api_key: Optional[str] = None,
    prior_strength: float = 1.0,
    prior_scale_mode: str = "learned",
    alpha_max: float = 2.0,
    allow_prior_inversion: bool = False,
    acq_xi: float = 0.0,
    grid_n: int = 600,
    figsize: Tuple[float, float] = (12.5, 7.0),
    save_path: Optional[str] = None,
    show: bool = True,
    verbose: bool = False,
    manuscript_style: bool = True,
    print_prompt_and_readout: bool = True,
) -> Dict[str, object]:
    """Manuscript-ready 2x2 panel figure showing prior→posterior→acquisition.

    Layout:
      - Top-left: prior mean (former panel C)
      - Top-right: blank white space for your text
      - Bottom-left: posterior comparison (former panel E)
      - Bottom-right: acquisition impact (former panel F)
    """
    if manuscript_style:
        set_manuscript_style()

    payload = _compute_for_demo(
        n_train=n_train,
        noise=noise,
        seed=seed,
        hint=hint,
        preset=preset,
        use_llm=use_llm,
        llm_prompt=llm_prompt,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_api_key=llm_api_key,
        prior_strength=prior_strength,
        prior_scale_mode=prior_scale_mode,
        alpha_max=alpha_max,
        allow_prior_inversion=allow_prior_inversion,
        grid_n=grid_n,
        acq_xi=acq_xi,
        verbose=verbose,
    )

    if print_prompt_and_readout:
        print("----- PROMPT (copy) -----")
        print(str(payload.get("prompt_used", "")))
        print("\n----- READOUT JSON (copy) -----")
        print(str(payload.get("readout_json", "")))
        llm_raw = payload.get("llm_raw")
        if llm_raw:
            print("\n----- LLM RAW OUTPUT (debug) -----")
            print(str(llm_raw))

    x = payload["grid"]  # type: ignore[assignment]
    y_true = payload["true_curve"]  # type: ignore[assignment]
    x_obs = payload["train_x"]  # type: ignore[assignment]
    y_obs = payload["train_y"]  # type: ignore[assignment]

    mean_flat = payload["mean_flat"]  # type: ignore[assignment]
    std_flat = payload["std_flat"]  # type: ignore[assignment]
    mean_text = payload["mean_text"]  # type: ignore[assignment]
    std_text = payload["std_text"]  # type: ignore[assignment]
    m0 = payload["m0_grid"]  # type: ignore[assignment]
    m0_scaled = payload["m0_scaled"]  # type: ignore[assignment]
    ei_flat = payload["ei_flat"]  # type: ignore[assignment]
    ei_text = payload["ei_text"]  # type: ignore[assignment]
    next_x_flat = payload["next_x_flat"]  # type: ignore[assignment]
    next_x_text = payload["next_x_text"]  # type: ignore[assignment]
    rh_lo, rh_hi = payload["range_hint"]  # type: ignore[misc]

    y_limits = (float(np.min(y_true)) - 0.2, float(np.max(y_true)) + 0.2)

    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax_text, ax_prior, ax_post, ax_acq = axs.flatten().tolist()

    # Top-left: Blank space for manuscript text
    ax_text.axis("off")

    # Top-right: Prior mean
    ax_prior.axvspan(rh_lo, rh_hi, color=COLORS["prior"], alpha=0.12, label="Hint window")
    ax_prior.plot(x, m0, color=COLORS["prior"], ls="--", lw=1.6, label="Raw $m_0(x)$")
    ax_prior.plot(x, m0_scaled, color=COLORS["prior"], lw=2.2, label="Scaled contribution")
    ax_prior.set_title("Prior Mean")
    ax_prior.set(xlabel="x", ylabel="prior", xlim=(0.0, 1.0))
    ax_prior.grid(True, alpha=0.2)
    ax_prior.legend(loc="best")

    # Bottom-left: Posterior comparison (former panel E)
    ax_post.plot(x, y_true, color="k", lw=1.2, alpha=0.7, label="True function")
    ax_post.scatter(x_obs, y_obs, color=COLORS["obs"], edgecolors="k", zorder=5, label="Observations")
    ax_post.plot(x, mean_flat, color=COLORS["baseline"], lw=2.0, label="Baseline mean")
    ax_post.fill_between(
        x,
        mean_flat - 2 * std_flat,
        mean_flat + 2 * std_flat,
        color=COLORS["baseline"],
        alpha=0.15,
    )
    ax_post.plot(x, mean_text, color=COLORS["shaped"], lw=2.0, label="Language-shaped mean")
    ax_post.fill_between(
        x,
        mean_text - 2 * std_text,
        mean_text + 2 * std_text,
        color=COLORS["shaped"],
        alpha=0.15,
    )
    ax_post.axvspan(rh_lo, rh_hi, color=COLORS["prior"], alpha=0.08)
    ax_post.set_title("Posterior Comparison")
    ax_post.set(xlabel="x", ylabel="y", xlim=(0.0, 1.0), ylim=y_limits)
    ax_post.grid(True, alpha=0.2)
    ax_post.legend(loc="best")

    # Bottom-right: Acquisition impact (former panel F)
    ax_acq.plot(x, ei_flat, color=COLORS["baseline"], lw=2.0, label="EI (baseline)")
    ax_acq.plot(x, ei_text, color=COLORS["shaped"], lw=2.0, label="EI (shaped)")
    ax_acq.axvline(next_x_flat, color=COLORS["baseline"], lw=1.2, ls="--", alpha=0.8, label="Next (baseline)")
    ax_acq.axvline(next_x_text, color=COLORS["shaped"], lw=1.2, ls="--", alpha=0.8, label="Next (shaped)")
    ax_acq.axvspan(rh_lo, rh_hi, color=COLORS["prior"], alpha=0.08)
    ax_acq.set_title("Optimization Impact (EI)")
    ax_acq.set(xlabel="x", ylabel="EI", xlim=(0.0, 1.0))
    ax_acq.grid(True, alpha=0.2)
    ax_acq.legend(loc="best")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return {"fig": fig, "axes": axs, "text_ax": ax_text, **payload}


out = run_demo_multipanel_(
    n_train=3,
    preset="right_peak",
    prior_strength=0.4,
    show=False,
)

# add your manuscript text into the blank panel
out["text_ax"].text(0, 1, "Your text here...", va="top", ha="left", transform=out["text_ax"].transAxes)
out["fig"]

#%%