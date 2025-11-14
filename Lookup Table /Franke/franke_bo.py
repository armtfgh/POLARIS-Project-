#%%
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore
    print("[LLM WARNING] httpx not installed; LLM summaries disabled.")

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    print("[LLM WARNING] openai package not installed; LLM summaries disabled.")
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("ignore")


def franke_surface(X: torch.Tensor) -> torch.Tensor:
    """Vectorized Franke surface evaluated row-wise for points in [0, 1]^2."""
    x = X[..., 0]
    y = X[..., 1]
    term1 = 0.75 * torch.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    term2 = 0.75 * torch.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    term3 = 0.5 * torch.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    term4 = -0.2 * torch.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def approximate_franke_max(resolution: int = 200) -> float:
    """Approximate the global optimum of the Franke surface via grid evaluation."""
    grid = torch.linspace(0.0, 1.0, resolution, dtype=torch.double)
    mesh_x, mesh_y = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack((mesh_x.reshape(-1), mesh_y.reshape(-1)), dim=-1)
    with torch.no_grad():
        return franke_surface(points).max().item()


def compute_search_space_coverage(X: torch.Tensor) -> float:
    """Approximate coverage as normalized area of the bounding box inside [0,1]^2."""
    if X.shape[0] < 2:
        return 0.0
    mins = X.min(dim=0).values
    maxs = X.max(dim=0).values
    return float(torch.clamp(maxs - mins, min=0.0).prod().item())


def geometric_mean_lengthscale(lengthscale: torch.Tensor) -> float:
    flat = lengthscale.detach().view(-1)
    return float(torch.exp(torch.log(flat).mean()).item())


def extract_gp_lengthscale(gp_model: SingleTaskGP) -> float:
    """Robustly extract a representative length scale from the GP kernel."""
    kernel = getattr(gp_model, "covar_module", None)
    if kernel is None:
        return 1.0
    base_kernel = getattr(kernel, "base_kernel", kernel)
    lengthscale = getattr(base_kernel, "lengthscale", None)
    if lengthscale is None:
        return 1.0
    return geometric_mean_lengthscale(lengthscale)


def categorize_level(value: float, *, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value < high:
        return "medium"
    return "high"


def describe_uncertainty_strategy(level: str) -> str:
    lookup = {
        "low": "focused on exploitation near incumbents",
        "medium": "balanced exploration around promising regions",
        "high": "prioritized exploration of uncertain areas",
    }
    return lookup.get(level, "followed mixed strategies")


def describe_regret_behavior(level: str) -> str:
    lookup = {
        "low": "the optimization typically converged",
        "medium": "progress slowed but continued improving",
        "high": "search kept exploring broadly",
    }
    return lookup.get(level, "behavior was mixed")


def interpret_length_scale(length_scale: float) -> str:
    if length_scale >= 0.3:
        return "a smooth landscape"
    if length_scale >= 0.12:
        return "moderately smooth terrain"
    return "a rough or highly varying surface"


def ensure_log_file(path: Optional[Union[str, Path]]) -> Optional[Path]:
    if path is None:
        return None
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    return log_path


def init_openai_client(
    api_key: Optional[str],
    endpoint: Optional[str],
    timeout: float,
):
    if not api_key:
        return None, None
    if httpx is None:
        raise ImportError("httpx is required for OpenAI summaries. Install httpx.")
    if OpenAI is None:
        raise ImportError("openai package is required for LLM summaries. Install openai.")
    http_client = httpx.Client(verify=False, timeout=timeout)
    client_kwargs = {
        "api_key": api_key,
        "http_client": http_client,
    }
    if endpoint:
        client_kwargs["base_url"] = endpoint
    return OpenAI(**client_kwargs), http_client


def append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def evaluate_acquisition_landscape(
    acq_function: ExpectedImprovement,
    bounds: torch.Tensor,
    *,
    n_samples: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, Union[List[float], float]]:
    raw = (
        draw_sobol_samples(bounds=bounds, n=n_samples, q=1)
        .squeeze(1)
        .to(device=device, dtype=dtype)
    )
    candidates = raw.unsqueeze(1)
    with torch.no_grad():
        values = acq_function(candidates).squeeze(-1).squeeze(-1)
    topk = min(3, values.numel())
    top_values = torch.topk(values, k=topk).values.tolist()
    rank_gap = float(top_values[0] - top_values[1]) if len(top_values) > 1 else 0.0
    return {"top_values": top_values, "rank_gap": rank_gap}


def build_iteration_metrics(
    *,
    iteration_idx: int,
    X_with_new: torch.Tensor,
    Y_with_new: torch.Tensor,
    gp_model: SingleTaskGP,
    marginal_ll: float,
    posterior_mean: float,
    posterior_std: float,
    ei_value: float,
    alt_acq_stats: Dict[str, Union[List[float], float]],
    observed_value: float,
    previous_best: float,
    global_best: float,
    exploration_weight: float,
) -> Dict[str, dict]:
    dataset_size = int(X_with_new.shape[0])
    best_value = float(Y_with_new.max().item())
    coverage = compute_search_space_coverage(X_with_new.detach().cpu())
    lengthscale = extract_gp_lengthscale(gp_model)
    noise_level = float(gp_model.likelihood.noise.mean().item())
    pred_uncertainty = float(posterior_std)
    uncertainty_level = categorize_level(pred_uncertainty, low=0.05, high=0.15)
    regret = max(global_best - best_value, 0.0)
    regret_level = categorize_level(regret, low=0.02, high=0.08)

    state = {
        "iteration": iteration_idx,
        "best_objective": best_value,
        "n_obs": dataset_size,
        "coverage": coverage,
    }
    gp_info = {
        "length_scale": lengthscale,
        "noise": noise_level,
        "marginal_log_likelihood": float(marginal_ll),
        "pred_uncertainty": pred_uncertainty,
        "uncertainty_level": uncertainty_level,
        "length_scale_insight": interpret_length_scale(lengthscale),
    }
    acquisition = {
        "ei_value": float(ei_value),
        "exploration_weight": float(exploration_weight),
        "rank_gap": float(alt_acq_stats.get("rank_gap", 0.0)),
        "top_ei_values": alt_acq_stats.get("top_values", []),
    }
    decision = {
        "observed": float(observed_value),
        "prediction_error": float(observed_value - posterior_mean),
        "simple_regret": float(regret),
        "regret_level": regret_level,
        "improvement": float(best_value - previous_best),
        "is_new_best": bool(best_value > previous_best + 1e-9),
    }
    return {"state": state, "gp": gp_info, "acquisition": acquisition, "decision": decision}


def format_info_log(iteration_metrics: Dict[str, dict], summary: str) -> str:
    state = iteration_metrics["state"]
    gp_info = iteration_metrics["gp"]
    af_info = iteration_metrics["acquisition"]
    decision = iteration_metrics["decision"]
    return (
        f"Iteration {state['iteration']}:\n"
        f"State: [best={state['best_objective']:.4f}, iter={state['iteration']}, "
        f"n_obs={state['n_obs']}, coverage={state['coverage']:.3f}]\n"
        f"GP: [length_scale={gp_info['length_scale']:.4f}, noise={gp_info['noise']:.4f}, "
        f"marg_likelihood={gp_info['marginal_log_likelihood']:.3f}, "
        f"pred_uncertainty={gp_info['pred_uncertainty']:.4f}]\n"
        f"AF: [EI_value={af_info['ei_value']:.4f}, exploration_weight={af_info['exploration_weight']:.3f}, "
        f"rank_gap={af_info['rank_gap']:.4f}]\n"
        f"Outcome: [observed={decision['observed']:.4f}, "
        f"pred_error={decision['prediction_error']:.4f}, regret={decision['simple_regret']:.4f}]\n"
        f"Summary: {summary}\n\n"
    )


def build_llm_prompt(
    iteration_metrics: Dict[str, dict],
    *,
    context_description: str,
    history: List[Dict[str, dict]],
) -> str:
    state = iteration_metrics["state"]
    gp_info = iteration_metrics["gp"]
    decision = iteration_metrics["decision"]
    uncertainty_level = gp_info["uncertainty_level"]
    regret_level = decision["regret_level"]
    previous_summary_text = (
        history[-1].get("summary")
        if history
        else "No previous iterations completed yet."
    )

    def find_past(section_key: str, level_key: str, level_value: str, default: str) -> str:
        history_default = (
            history[-1].get("summary", default) if history else default
        )
        for past in reversed(history):
            section = past.get(section_key, {})
            if section.get(level_key) == level_value:
                return past.get("summary", history_default)
        return history_default

    past_uncertainty = find_past(
        "gp", "uncertainty_level", uncertainty_level, describe_uncertainty_strategy(uncertainty_level)
    )
    past_regret = find_past(
        "decision", "regret_level", regret_level, describe_regret_behavior(regret_level)
    )

    pretty_state = json.dumps(state)
    pretty_gp = json.dumps(
        {k: v for k, v in gp_info.items() if k not in {"uncertainty_level", "length_scale_insight"}}
    )
    pretty_af = json.dumps(iteration_metrics["acquisition"])
    pretty_decision = json.dumps(decision)

    prompt = (
        "You are creating training data to teach an AI about Bayesian optimization decision-making.\n\n"
        f"Current Iteration: {state['iteration']}\n"
        f"Context: {context_description}\n"
        f"Previous iteration summary: {previous_summary_text}\n\n"
        "Iteration Data:\n"
        f"State: {pretty_state}\n"
        f"GP: {pretty_gp}\n"
        f"AF: {pretty_af}\n"
        f"Outcome: {pretty_decision}\n\n"
        "Compare this iteration to similar past situations:\n"
        f"- When uncertainty was {uncertainty_level}, previous strategies were {past_uncertainty}\n"
        f"- When regret was {regret_level}, the optimization {past_regret}\n"
        f"- GP length scale of {gp_info['length_scale']:.4f} suggests {gp_info['length_scale_insight']}\n\n"
        "Generate a summary that explains:\n"
        "1. The strategic choice made (and why it differs from or aligns with past patterns)\n"
        "2. The expected outcome based on GP predictions\n"
        "3. The actual outcome and what this reveals about the objective function\n\n"
        "Include explicit reasoning about trade-offs between exploration and exploitation.\n\n"
        "Summary:"
    )
    return prompt


def request_llm_summary(
    prompt: str,
    *,
    client,
    model: str,
    temperature: float = 0.2,
) -> str:
    if client is None:
        raise RuntimeError("OpenAI client is not available.")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Bayesian optimization analyst who writes concise yet insightful iteration reports.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def fallback_summary(iteration_metrics: Dict[str, dict]) -> str:
    state = iteration_metrics["state"]
    gp_info = iteration_metrics["gp"]
    decision = iteration_metrics["decision"]
    acquisition = iteration_metrics["acquisition"]
    mode = "exploration" if acquisition["exploration_weight"] > 0.5 else "exploitation"
    new_best = "set a new best value" if decision["is_new_best"] else "did not beat the incumbent"
    return (
        f"Iteration {state['iteration']} emphasized {mode}, guided by EI={acquisition['ei_value']:.3f} "
        f"and posterior uncertainty {gp_info['pred_uncertainty']:.3f}. "
        f"The GP expected {decision['observed'] - decision['prediction_error']:.3f} but observed "
        f"{decision['observed']:.3f}, so the prediction error was {decision['prediction_error']:.3f}. "
        f"The search {new_best} with regret {decision['simple_regret']:.3f}, "
        f"consistent with a length scale indicating {gp_info['length_scale_insight']}."
    )


def run_franke_bo(
    num_init: int = 5,
    num_iterations: int = 20,
    *,
    dtype: torch.dtype = torch.double,
    device: Optional[Union[torch.device, str]] = None,
    seed: Optional[int] = None,
    info_log_path: Optional[Union[str, Path]] = "logs/info_logs.txt",
    llm_log_path: Optional[Union[str, Path]] = "logs/llm_logs.txt",
    llm_endpoint: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_timeout: float = 30.0,
    llm_model: Optional[str] = None,
    true_max: Optional[float] = None,
    af_rank_samples: int = 256,
    llm_per_iteration: bool = True,
    dataset_path: Optional[Union[str, Path]] = None,
    context_description: Optional[str] = None,
) -> dict:
    """Run Bayesian optimization with SingleTaskGP + EI on the Franke surface."""
    if isinstance(device, str):
        device = torch.device(device)
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
    resolved_llm_endpoint = llm_endpoint or os.environ.get("LLM_ENDPOINT")
    resolved_llm_api_key = (
        llm_api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    resolved_llm_model = llm_model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    openai_http_client = None
    openai_client = None
    if llm_per_iteration:
        openai_client, openai_http_client = init_openai_client(
            resolved_llm_api_key, resolved_llm_endpoint, llm_timeout
        )

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dtype, device=resolved_device)
    info_log_file = ensure_log_file(info_log_path)
    llm_log_file = ensure_log_file(llm_log_path)
    dataset_file = None
    if dataset_path is not None:
        dataset_file = Path(dataset_path)
        dataset_file.parent.mkdir(parents=True, exist_ok=True)
    context_description = context_description or "SingleTaskGP + Expected Improvement on the Franke surface."
    global_best = true_max if true_max is not None else approximate_franke_max()
    history: List[Dict[str, dict]] = []

    X_train = (
        draw_sobol_samples(bounds=bounds, n=num_init, q=1)
        .squeeze(1)
        .to(device=resolved_device, dtype=dtype)
    )
    with torch.no_grad():
        Y_train = franke_surface(X_train).unsqueeze(-1)
    Y_best = [Y_train.max().item()]

    try:
        for iteration_idx in range(1, num_iterations + 1):
            gp = SingleTaskGP(X_train, Y_train)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            with torch.no_grad():
                marginal_ll = mll(gp(X_train), Y_train.squeeze(-1)).item()

            best_before = float(Y_train.max().item())
            ei = ExpectedImprovement(model=gp, best_f=Y_train.max(), maximize=True)
            candidate_raw, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=1,
                num_restarts=8,
                raw_samples=128,
            )
            if candidate_raw.dim() == 2:
                candidate_eval = candidate_raw.unsqueeze(0)
            elif candidate_raw.dim() == 3:
                candidate_eval = candidate_raw
            else:
                candidate_eval = candidate_raw.view(1, 1, -1)
            candidate_points = candidate_eval.view(-1, candidate_eval.size(-1))

            with torch.no_grad():
                posterior = gp.posterior(candidate_eval)
                posterior_mean = float(posterior.mean.mean().item())
                posterior_std = float(posterior.variance.clamp_min(0.0).sqrt().mean().item())
                ei_value = float(ei(candidate_eval).mean().item())

            with torch.no_grad():
                new_y = franke_surface(candidate_points).unsqueeze(-1)
            observed_value = float(new_y.squeeze(-1)[-1].item())
            next_X = torch.cat([X_train, candidate_points], dim=0)
            next_Y = torch.cat([Y_train, new_y], dim=0)

            alt_stats = evaluate_acquisition_landscape(
                ei,
                bounds,
                n_samples=af_rank_samples,
                dtype=dtype,
                device=resolved_device,
            )
            exploitation_signal = abs(posterior_mean - best_before)
            exploration_weight = float(
                posterior_std / (posterior_std + exploitation_signal + 1e-8)
            )

            iteration_metrics = build_iteration_metrics(
                iteration_idx=iteration_idx,
                X_with_new=next_X,
                Y_with_new=next_Y,
                gp_model=gp,
                marginal_ll=marginal_ll,
                posterior_mean=posterior_mean,
                posterior_std=posterior_std,
                ei_value=ei_value,
                alt_acq_stats=alt_stats,
                observed_value=observed_value,
                previous_best=best_before,
                global_best=global_best,
                exploration_weight=exploration_weight,
            )

            prompt = build_llm_prompt(
                iteration_metrics,
                context_description=context_description,
                history=history,
            )
            if openai_client:
                try:
                    summary = request_llm_summary(
                        prompt,
                        client=openai_client,
                        model=resolved_llm_model,
                    )
                except Exception as exc:
                    print(f"[LLM WARNING] Request failed (iteration {iteration_idx}): {exc}")
                    summary = fallback_summary(iteration_metrics)
            else:
                summary = fallback_summary(iteration_metrics)

            if info_log_file:
                append_text(info_log_file, format_info_log(iteration_metrics, summary))
            if llm_log_file:
                llm_block = (
                    f"Iteration {iteration_metrics['state']['iteration']}:\n"
                    f"Prompt:\n{prompt}\n\nSummary:\n{summary}\n\n"
                )
                append_text(llm_log_file, llm_block)

            history.append(
                {
                    "state": iteration_metrics["state"],
                    "gp": iteration_metrics["gp"],
                    "acquisition": iteration_metrics["acquisition"],
                    "decision": iteration_metrics["decision"],
                    "summary": summary,
                }
            )

            X_train = next_X
            Y_train = next_Y
            best_now = iteration_metrics["state"]["best_objective"]
            Y_best.append(best_now)
    finally:
        if openai_http_client is not None:
            openai_http_client.close()

    if dataset_file is not None:
        dataset_file.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    return {
        "points": X_train,
        "values": Y_train,
        "best_trace": Y_best,
        "info_log_path": str(info_log_file) if info_log_file else None,
        "llm_log_path": str(llm_log_file) if llm_log_file else None,
        "iteration_history": history,
        "dataset_path": str(dataset_file) if dataset_file else None,
        "context_description": context_description,
    }


def build_repeat_summary_prompt(
    iteration_history: List[Dict[str, dict]],
    *,
    context_description: str,
) -> str:
    if not iteration_history:
        raise ValueError("Iteration history is empty; cannot build summary prompt.")

    condensed = []
    for entry in iteration_history:
        state = entry["state"]
        gp_info = entry["gp"]
        acq = entry["acquisition"]
        decision = entry["decision"]
        condensed.append(
            {
                "iteration": state["iteration"],
                "best_objective": state["best_objective"],
                "coverage": state["coverage"],
                "length_scale": gp_info["length_scale"],
                "noise": gp_info["noise"],
                "pred_uncertainty": gp_info["pred_uncertainty"],
                "ei_value": acq["ei_value"],
                "exploration_weight": acq["exploration_weight"],
                "observed": decision["observed"],
                "prediction_error": decision["prediction_error"],
                "regret": decision["simple_regret"],
                "improvement": decision["improvement"],
                "is_new_best": decision["is_new_best"],
                "previous_summary": entry.get("summary"),
            }
        )
    condensed_json = json.dumps(condensed, indent=2)

    instructions = (
        "You are analyzing a full Bayesian optimization run on the Franke surface.\n"
        f"Context: {context_description}\n\n"
        "Below is structured data for each iteration (chronological order):\n"
        f"{condensed_json}\n\n"
        "Produce an explainable narrative with the following format for each iteration:\n"
        "Iteration k:\n"
        "- What happened: Summarize the BO action, acquisition signal (EI/exploration weight), and observed outcome "
        "using the provided metrics. Mention whether a new best was found.\n"
        "- Why it happened: Reason about the decision by referencing earlier iterations and the memory field "
        "('previous_summary') when present. Explain the trade-off between exploration and exploitation, connect the decision "
        "to uncertainty/regret/length-scale trends, and describe how past performance informed this step.\n"
        "Use explicit continuity cues such as 'Previously', 'Compared to iteration j', or 'Because earlier iterations showed ...'. "
        "Highlight cause-and-effect relationships (e.g., high uncertainty → exploration, high regret → continued search).\n\n"
        "After the per-iteration analysis, add two sections:\n"
        "Overall Trajectory: Describe how the BO strategy evolved over time and why key shifts occurred, referencing multiple iterations.\n"
        "Lessons & Next Steps: Give actionable takeaways for future BO runs based on the observed behavior and reasoning.\n"
        "Summary:"
    )
    return instructions


def summarize_iteration_history(
    iteration_history: List[Dict[str, dict]],
    *,
    context_description: str,
    client,
    model: str,
    temperature: float = 0.2,
    summary_path: Optional[Union[str, Path]] = None,
) -> str:
    prompt = build_repeat_summary_prompt(
        iteration_history,
        context_description=context_description,
    )
    summary = request_llm_summary(
        prompt,
        client=client,
        model=model,
        temperature=temperature,
    )
    if summary_path is not None:
        summary_file = Path(summary_path)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(summary, encoding="utf-8")
    return summary


def run_franke_bo_repeats(
    num_repeats: int = 50,
    *,
    num_iterations: int = 20,
    num_init: int = 5,
    base_seed: int = 0,
    seeds: Optional[Sequence[int]] = None,
    log_dir: Union[str, Path] = "logs",
    dataset_dir: Optional[Union[str, Path]] = "datasets",
    summarize_after: bool = False,
    summary_dir: Union[str, Path] = "summaries",
    summary_model: Optional[str] = None,
    summary_temperature: float = 0.2,
    summary_endpoint: Optional[str] = None,
    summary_api_key: Optional[str] = None,
    summary_timeout: float = 60.0,
    **bo_kwargs,
) -> List[dict]:
    """Run multiple BO repeats, saving per-repeat raw/LLM logs."""
    if seeds is not None and len(seeds) < num_repeats:
        raise ValueError("Provided seeds list shorter than num_repeats.")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir_path = Path(dataset_dir) if dataset_dir is not None else None
    if dataset_dir_path is not None:
        dataset_dir_path.mkdir(parents=True, exist_ok=True)
    summary_dir_path = Path(summary_dir)
    if summarize_after:
        summary_dir_path.mkdir(parents=True, exist_ok=True)

    bo_kwargs = dict(bo_kwargs)
    if summarize_after and "llm_per_iteration" not in bo_kwargs:
        bo_kwargs["llm_per_iteration"] = False
    resolved_true_max = bo_kwargs.pop("true_max", None)
    if resolved_true_max is None:
        resolved_true_max = approximate_franke_max()

    resolved_summary_endpoint = summary_endpoint or os.environ.get("LLM_SUMMARY_ENDPOINT") or os.environ.get("LLM_ENDPOINT")
    resolved_summary_api_key = (
        summary_api_key
        or os.environ.get("LLM_SUMMARY_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    resolved_summary_model = summary_model or os.environ.get("LLM_SUMMARY_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    summary_client = None
    summary_http_client = None
    if summarize_after:
        summary_client, summary_http_client = init_openai_client(
            resolved_summary_api_key,
            resolved_summary_endpoint,
            summary_timeout,
        )
        if summary_client is None:
            raise ValueError("LLM summarization requested but no API key provided.")

    aggregated_results: List[dict] = []
    try:
        for idx in range(num_repeats):
            repeat_id = idx + 1
            repeat_seed = seeds[idx] if seeds is not None else base_seed + idx
            info_path = log_dir / f"raw_logs_repeat{repeat_id}.txt"
            llm_path = log_dir / f"llm_logs_repeat{repeat_id}.txt"
            dataset_path = (
                dataset_dir_path / f"dataset_repeat{repeat_id}.json"
                if dataset_dir_path is not None
                else None
            )
            result = run_franke_bo(
                num_init=num_init,
                num_iterations=num_iterations,
                seed=repeat_seed,
                info_log_path=info_path,
                llm_log_path=llm_path,
                true_max=resolved_true_max,
                dataset_path=dataset_path,
                **bo_kwargs,
            )
            summary_path = None
            repeat_summary = None
            if summarize_after:
                summary_path = summary_dir_path / f"llm_summary_repeat{repeat_id}.txt"
                repeat_summary = summarize_iteration_history(
                    result["iteration_history"],
                    context_description=result["context_description"],
                    client=summary_client,
                    model=resolved_summary_model,
                    temperature=summary_temperature,
                    summary_path=summary_path,
                )

            aggregated_results.append(
                {
                    "repeat": repeat_id,
                    "seed": repeat_seed,
                    "best_trace": result["best_trace"],
                    "points": result["points"],
                    "values": result["values"],
                    "info_log_path": str(info_path),
                    "llm_log_path": str(llm_path),
                    "dataset_path": result["dataset_path"],
                    "summary_path": str(summary_path) if summary_path else None,
                    "summary": repeat_summary,
                }
            )
            print(
                f"[REPEAT {repeat_id}/{num_repeats}] seed={repeat_seed} "
                f"best={result['best_trace'][-1]:.4f} "
                f"raw_log={info_path.name} llm_log={llm_path.name} "
                f"dataset={Path(result['dataset_path']).name if result['dataset_path'] else 'none'} "
                f"summary={'yes' if repeat_summary else 'no'}"
            )
    finally:
        if summary_http_client is not None:
            summary_http_client.close()

    return aggregated_results


# if __name__ == "__main__":
#     repeats = run_franke_bo_repeats(
#         num_repeats=1,
#         num_iterations=5,
#         num_init=4,
#         base_seed=123,
#         log_dir="logs",
#         dataset_dir="datasets",
#         summarize_after=False,
#         llm_endpoint=None,
#         llm_model="gpt-4o-mini",
#         llm_per_iteration=False,
#     )
#     for repeat_run in repeats:
#         trace = repeat_run["best_trace"]
#         print(
#             f"[SUMMARY repeat={repeat_run['repeat']}] seed={repeat_run['seed']} "
#             f"best_final={trace[-1]:.4f} "
#             f"info_log={repeat_run['info_log_path']} "
#             f"dataset={repeat_run.get('dataset_path')}"
#         )
# #%%


runs = run_franke_bo_repeats(
    num_repeats=1,
    num_iterations=20,
    num_init=3,
    base_seed=0,
    log_dir="logs",
    dataset_dir="datasets",
    summarize_after=True,      # single LLM call per repeat
    summary_dir="summaries",
    llm_per_iteration=False,   # no per-iteration LLM usage
)
#%%
