"""
llm_agent.py
=============

This module provides an agent harness that lets a large language model (LLM)
run a Bayesian optimization (BO) loop on the P3HT/CNT lookup table.

Key ideas:
----------
1. The lookup table is treated as the ground-truth oracle. The LLM can inspect
   feature ranges but must call `evaluate_candidate` to reveal objective values.
2. GP + EI machinery is reused from the rest of the codebase (SingleTaskGP,
   ExpectedImprovement) so behaviour matches existing scripts.
3. The agent exposes a small tool API (list candidates, fit GP, score EI, etc.)
   that the LLM can call through OpenAI-style function/tool calling.
4. A controller loop (`LLMControlledBOAgent`) streams the current BO state to
   the model, executes tool calls, and stops when the LLM emits a final report.

To run this agent you need an LLM client that supports tool/function calling.
By default we attempt to create an OpenAI client if `OPENAI_API_KEY` is set,
but you can inject any custom client that implements the same interface.
"""
#%%
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

try:
    import httpx
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]
    httpx = None  # type: ignore[assignment]

from LLM_checker import LookupTable, load_lookup_csv


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


# ---------------------------------------------------------------------------
# Dataclasses and tool registry
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    dataset_path: str = "P3HT_dataset.csv"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_chat_rounds: int = 32
    default_budget: int = 15
    max_tool_failures: int = 2


@dataclass
class ToolFunction:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]

    def as_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolFunction] = {}

    def register(self, tool: ToolFunction) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        self._tools[tool.name] = tool

    def schemas(self) -> List[Dict[str, Any]]:
        return [tool.as_openai_tool() for tool in self._tools.values()]

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._tools:
            raise ValueError(f"Unknown tool '{name}'.")
        handler = self._tools[name].handler
        try:
            return {"status": "ok", "data": handler(arguments)}
        except Exception as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }


# ---------------------------------------------------------------------------
# Bayesian optimization environment
# ---------------------------------------------------------------------------

def _tensor_to_float_list(t: Tensor) -> List[float]:
    return [float(x) for x in t.detach().cpu().view(-1)]


def _tensor_to_float(t: Tensor) -> float:
    return float(t.detach().cpu().item())


class BayesianOptimizationEnvironment:
    """Stateful environment that exposes BO primitives as callable tools."""

    def __init__(self, lookup: LookupTable) -> None:
        self.lookup = lookup
        # Work on CPU tensors to simplify tool execution.
        self.lookup.X = self.lookup.X.detach().cpu().to(DTYPE)
        self.lookup.X_raw = self.lookup.X_raw.detach().cpu().to(DTYPE)
        self.lookup.y = self.lookup.y.detach().cpu().to(DTYPE)
        self._feature_index = {name: i for i, name in enumerate(self.lookup.feature_names)}
        self._budget: Optional[int] = None
        self.reset()

    # --- State helpers -------------------------------------------------
    @property
    def observed_indices(self) -> List[int]:
        return list(self._observed_order)

    def reset(self) -> None:
        self._observed_order: List[int] = []
        self._observed_set: set[int] = set()
        self._X_obs: Optional[Tensor] = None
        self._Y_obs: Optional[Tensor] = None
        self._gp: Optional[SingleTaskGP] = None
        self._last_gp_time: Optional[float] = None
        self._budget_used: int = 0

    def set_budget(self, budget: int) -> None:
        self._budget = max(1, int(budget))

    # --- Lookup utilities ----------------------------------------------
    def _serialize_point(self, idx: int) -> Dict[str, Any]:
        raw_vals = self.lookup.X_raw[idx]
        norm_vals = self.lookup.X[idx]
        feature_names = self.lookup.feature_names
        return {
            "idx": idx,
            "features_raw": {
                name: float(val) for name, val in zip(feature_names, raw_vals.tolist())
            },
            "features_normalized": {
                name: float(val) for name, val in zip(feature_names, norm_vals.tolist())
            },
        }

    def describe_lookup(self) -> Dict[str, Any]:
        mins = {
            name: float(val)
            for name, val in zip(self.lookup.feature_names, self.lookup.mins.tolist())
        }
        maxs = {
            name: float(val)
            for name, val in zip(self.lookup.feature_names, self.lookup.maxs.tolist())
        }
        return {
            "num_candidates": self.lookup.n,
            "num_features": self.lookup.d,
            "feature_names": list(self.lookup.feature_names),
            "feature_min": mins,
            "feature_max": maxs,
            "objective_name": self.lookup.objective_name,
        }

    def list_unobserved(
        self,
        *,
        k: Optional[int] = None,
        shuffle: bool = False,
        filters: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        remaining = [
            idx for idx in range(self.lookup.n) if idx not in self._observed_set
        ]
        if filters:
            remaining = self._apply_filters(remaining, filters)
        if shuffle:
            rng = torch.Generator(device="cpu")
            seed = int(time.time_ns() % (2**32 - 1))
            rng.manual_seed(seed)
            perm = torch.randperm(len(remaining), generator=rng).tolist()
            remaining = [remaining[i] for i in perm]
        if k is not None:
            remaining = remaining[: max(0, int(k))]
        return {
            "status": "ok",
            "candidates": [self._serialize_point(idx) for idx in remaining],
        }

    def _apply_filters(
        self,
        indices: List[int],
        filters: List[Dict[str, float]],
    ) -> List[int]:
        filtered: List[int] = []
        for idx in indices:
            keep = True
            for spec in filters:
                name = spec.get("feature")
                feat_idx = self._feature_index.get(name)
                if feat_idx is None:
                    keep = False
                    break
                raw = float(self.lookup.X_raw[idx, feat_idx].item())
                lo = spec.get("min", -math.inf)
                hi = spec.get("max", math.inf)
                if not (lo <= raw <= hi):
                    keep = False
                    break
            if keep:
                filtered.append(idx)
        return filtered

    # --- Observations --------------------------------------------------
    def evaluate_candidate(self, idx: int) -> Dict[str, Any]:
        if idx in self._observed_set:
            raise ValueError(f"Candidate {idx} already evaluated.")
        if not (0 <= idx < self.lookup.n):
            raise ValueError("Index out of range.")
        if self._budget is not None and self._budget_used >= self._budget:
            raise RuntimeError("Evaluation budget exhausted.")
        y = float(self.lookup.y[idx].item())
        self._observed_order.append(idx)
        self._observed_set.add(idx)
        x_row = self.lookup.X[idx].unsqueeze(0)
        y_row = self.lookup.y[idx].reshape(1, 1)
        self._X_obs = x_row if self._X_obs is None else torch.cat([self._X_obs, x_row], dim=0)
        self._Y_obs = y_row if self._Y_obs is None else torch.cat([self._Y_obs, y_row], dim=0)
        self._budget_used += 1
        self._gp = None  # invalidate fitted model
        payload = self._serialize_point(idx)
        payload["objective"] = {self.lookup.objective_name: y}
        payload["budget_used"] = self._budget_used
        payload["budget_remaining"] = (
            None if self._budget is None else max(self._budget - self._budget_used, 0)
        )
        return payload

    # --- GP + EI -------------------------------------------------------
    def fit_gp(self) -> Dict[str, Any]:
        if self._X_obs is None or self._Y_obs is None or self._X_obs.shape[0] < 2:
            raise RuntimeError("Need at least two observations to fit a GP.")
        X = self._X_obs.to(DEVICE, DTYPE)
        Y = self._Y_obs.to(DEVICE, DTYPE)
        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(DEVICE, DTYPE)
        fit_gpytorch_mll(mll)
        gp = gp.to("cpu")
        self._gp = gp
        self._last_gp_time = time.time()
        lengthscale = None
        try:
            ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            lengthscale = [float(x) for x in ls.reshape(-1)]
        except AttributeError:
            pass
        noise = float(gp.likelihood.noise_covar.noise.detach().cpu().item())
        best_y = float(self._Y_obs.max().item())
        return {
            "status": "ok",
            "n_train": int(self._X_obs.shape[0]),
            "best_observed": best_y,
            "lengthscale": lengthscale,
            "noise": noise,
            "timestamp": self._last_gp_time,
        }

    def _ensure_gp(self) -> SingleTaskGP:
        if self._gp is None:
            info = self.fit_gp()
            if info.get("status") != "ok":
                raise RuntimeError("GP fit failed.")
        assert self._gp is not None
        return self._gp

    def expected_improvement(
        self,
        candidate_indices: Optional[Sequence[int]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        if not self._observed_order:
            raise RuntimeError("Collect observations before calling EI.")
        gp = self._ensure_gp()
        gp_device = gp.to(DEVICE, DTYPE).eval()
        best_y = float(self._Y_obs.max().item()) if self._Y_obs is not None else float("-inf")
        if candidate_indices is None:
            candidate_indices = [
                idx for idx in range(self.lookup.n) if idx not in self._observed_set
            ]
        pool = list(candidate_indices)
        if not pool:
            gp_device.to("cpu")
            self._gp = gp_device
            return {"status": "ok", "candidates": []}
        X_pool = self.lookup.X[pool].to(DEVICE, DTYPE)
        with torch.no_grad():
            acq = ExpectedImprovement(model=gp_device, best_f=best_y, maximize=True)
            scores = acq(X_pool.unsqueeze(1)).reshape(-1).detach().cpu()
        order = torch.argsort(scores, descending=True)
        results: List[Dict[str, Any]] = []
        for rank in order[: max(1, int(top_k))]:
            idx = pool[int(rank)]
            point = self._serialize_point(idx)
            posterior = gp_device.posterior(
                self.lookup.X[idx].unsqueeze(0).to(DEVICE, DTYPE)
            )
            mean = _tensor_to_float(posterior.mean.cpu())
            variance = torch.clamp(posterior.variance.cpu(), min=0.0)
            std = float(torch.sqrt(variance).item())
            point.update(
                {
                    "posterior_mean": mean,
                    "posterior_std": std,
                    "expected_improvement": float(scores[int(rank)].item()),
                }
            )
            results.append(point)
        gp_device.to("cpu")
        self._gp = gp_device
        return {"status": "ok", "candidates": results}

    # --- State summaries -----------------------------------------------
    def summarize_state(self) -> Dict[str, Any]:
        records: List[Dict[str, Any]] = []
        best_idx = None
        best_val = -math.inf
        for position, idx in enumerate(self._observed_order):
            y = float(self.lookup.y[idx].item())
            records.append(
                {
                    "position": position,
                    "candidate": self._serialize_point(idx),
                    "objective": {self.lookup.objective_name: y},
                }
            )
            if y > best_val:
                best_val = y
                best_idx = idx
        summary: Dict[str, Any] = {
            "num_observations": len(self._observed_order),
            "history": records,
            "best_observed": (
                None
                if best_idx is None
                else {
                    "candidate": self._serialize_point(best_idx),
                    "objective": {self.lookup.objective_name: best_val},
                }
            ),
            "budget_used": self._budget_used,
            "budget_remaining": (
                None if self._budget is None else max(self._budget - self._budget_used, 0)
            ),
        }
        return summary


# ---------------------------------------------------------------------------
# LLM client wrappers
# ---------------------------------------------------------------------------

class OpenAIChatWrapper:
    """Thin wrapper so we can swap in a mock client for testing."""

    def __init__(self, config: AgentConfig) -> None:
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI client not available. Provide a custom llm_client or install openai."
            )
        http_client = httpx.Client(verify=False) if httpx is not None else None
        self._client = OpenAI(http_client=http_client)
        self._config = config

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._config.model_name,
            temperature=self._config.temperature,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message
        tool_calls: List[Dict[str, Any]] = []
        if msg.tool_calls:
            for call in msg.tool_calls:
                call_type = getattr(call, "type", None)
                if not isinstance(call_type, str) or not call_type:
                    call_type = "function"
                fn = getattr(call, "function", None)
                if fn is None:
                    continue
                arguments = getattr(fn, "arguments", "{}")
                if not isinstance(arguments, str):
                    try:
                        arguments = json.dumps(arguments)
                    except Exception:
                        arguments = "{}"
                tool_calls.append(
                    {
                        "id": call.id,
                        "type": call_type,
                        "function": {
                            "name": fn.name,
                            "arguments": arguments or "{}",
                        },
                    }
                )
        return {
            "role": msg.role,
            "content": msg.content,
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason,
        }


# ---------------------------------------------------------------------------
# Agent controller
# ---------------------------------------------------------------------------

class LLMControlledBOAgent:
    """Controller that relays between the LLM and BO environment via tools."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        lookup: Optional[LookupTable] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.lookup = lookup or load_lookup_csv(self.config.dataset_path, impute_features="median")
        self.env = BayesianOptimizationEnvironment(self.lookup)
        self.registry = ToolRegistry()
        self._register_tools()
        self.llm_client = llm_client or self._build_default_client()

    def _build_default_client(self) -> OpenAIChatWrapper:
        return OpenAIChatWrapper(self.config)

    # --- Tool registration ---------------------------------------------
    def _register_tools(self) -> None:
        self.registry.register(
            ToolFunction(
                name="describe_lookup",
                description="Inspect feature ranges and metadata for the lookup oracle.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=lambda _: self.env.describe_lookup(),
            )
        )

        self.registry.register(
            ToolFunction(
                name="list_unobserved_candidates",
                description=(
                    "List unseen candidates from the lookup table. "
                    "Optional filters operate on RAW feature values."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "k": {"type": "integer", "minimum": 1, "description": "Max candidates to return"},
                        "shuffle": {"type": "boolean", "default": False},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "feature": {"type": "string"},
                                    "min": {"type": "number"},
                                    "max": {"type": "number"},
                                },
                                "required": ["feature"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "additionalProperties": False,
                },
                handler=self._handle_list_unobserved,
            )
        )

        self.registry.register(
            ToolFunction(
                name="evaluate_candidate",
                description="Query the oracle for an unseen candidate (counts against budget).",
                parameters={
                    "type": "object",
                    "properties": {
                        "idx": {"type": "integer", "minimum": 0},
                    },
                    "required": ["idx"],
                    "additionalProperties": False,
                },
                handler=self._handle_evaluate,
            )
        )

        self.registry.register(
            ToolFunction(
                name="fit_gp_model",
                description="Fit or refit a SingleTaskGP on current observations.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=lambda _: self.env.fit_gp(),
            )
        )

        self.registry.register(
            ToolFunction(
                name="score_expected_improvement",
                description=(
                    "Compute Expected Improvement for specified candidates using the "
                    "latest GP model. Implicitly fits the GP if needed."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "candidate_indices": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                        "top_k": {"type": "integer", "minimum": 1, "default": 10},
                    },
                    "additionalProperties": False,
                },
                handler=self._handle_ei,
            )
        )

        self.registry.register(
            ToolFunction(
                name="summarize_progress",
                description="Return all observations so far, including the incumbent.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=lambda _: self.env.summarize_state(),
            )
        )

        self.registry.register(
            ToolFunction(
                name="set_budget",
                description="Set or update the remaining evaluation budget.",
                parameters={
                    "type": "object",
                    "properties": {"budget": {"type": "integer", "minimum": 1}},
                    "required": ["budget"],
                    "additionalProperties": False,
                },
                handler=self._handle_set_budget,
            )
        )

        self.registry.register(
            ToolFunction(
                name="report_final_result",
                description=(
                    "Use this when you are done. Provide the best candidate and a short justification."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "incumbent_idx": {"type": "integer"},
                        "summary": {"type": "string"},
                        "observations_used": {"type": "integer"},
                    },
                    "required": ["incumbent_idx", "summary"],
                    "additionalProperties": False,
                },
                handler=lambda payload: {"status": "ok", "final_report": payload},
            )
        )

    # --- Tool handlers -------------------------------------------------
    def _handle_list_unobserved(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.env.list_unobserved(
            k=payload.get("k"),
            shuffle=bool(payload.get("shuffle", False)),
            filters=payload.get("filters"),
        )

    def _handle_evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.env.evaluate_candidate(int(payload["idx"]))

    def _handle_ei(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        indices = payload.get("candidate_indices")
        if indices is not None:
            indices = [int(i) for i in indices]
        top_k = int(payload.get("top_k", 10))
        return self.env.expected_improvement(indices, top_k)

    def _handle_set_budget(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        budget = int(payload["budget"])
        self.env.set_budget(budget)
        return {"status": "ok", "budget": budget}

    # --- Controller loop -----------------------------------------------
    def system_prompt(self, budget: Optional[int]) -> str:
        lookup_info = self.env.describe_lookup()
        feature_lines = "\n".join(
            f"- {name}: [{lookup_info['feature_min'][name]:.2f}, {lookup_info['feature_max'][name]:.2f}]"
            for name in lookup_info["feature_names"]
        )
        budget_line = (
            "You have unlimited evaluations."
            if budget is None
            else f"You must not exceed {budget} oracle evaluations."
        )
        return (
            "You are an autonomous Bayesian optimization scientist. "
            "Use the provided tools to select formulation candidates, fit Gaussian processes, "
            "evaluate Expected Improvement, and query the lookup oracle. "
            "The oracle corresponds to the P3HT/CNT conductivity dataset.\n\n"
            f"Budget guidance: {budget_line}\n"
            "Feature ranges:\n"
            f"{feature_lines}\n\n"
            "Each call to `evaluate_candidate` consumes budget and reveals the objective value. "
            "Always update your GP after new data if you plan to use EI. "
            "When you are satisfied with the campaign, call `report_final_result` summarizing the incumbent."
        )

    def run_session(
        self,
        user_task: str,
        *,
        budget: Optional[int] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if budget is None:
            budget = self.config.default_budget
        self.env.reset()
        self.env.set_budget(budget)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt(budget)}
        ]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_task})
        tools = self.registry.schemas()
        final_report: Optional[Dict[str, Any]] = None
        tool_failures = 0
        for round_idx in range(self.config.max_chat_rounds):
            reply = self.llm_client.complete(messages, tools)
            assistant_msg = {
                "role": "assistant",
                "content": reply.get("content"),
            }
            if reply.get("tool_calls"):
                assistant_msg["tool_calls"] = reply["tool_calls"]
            messages.append(assistant_msg)

            tool_calls = reply.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    call_type = call.get("type")
                    if not isinstance(call_type, str) or not call_type:
                        call_type = "function"
                    fn_payload = call.get("function") or {}
                    name = fn_payload.get("name")
                    if call_type != "function" or not name:
                        continue
                    raw_args = fn_payload.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError as exc:
                        args = {}
                        tool_response = {
                            "status": "error",
                            "error": f"JSONDecodeError: {exc}",
                        }
                    else:
                        tool_response = self.registry.call(name, args)
                        if name == "report_final_result" and tool_response.get("status") == "ok":
                            final_report = tool_response["data"]["final_report"]  # type: ignore[index]
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id"),
                            "name": name,
                            "content": json.dumps(tool_response),
                        }
                    )
                    if tool_response.get("status") == "error":
                        tool_failures += 1
                        if tool_failures >= self.config.max_tool_failures:
                            messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        "Tool execution errors exceeded the configured limit. "
                                        "Double-check your arguments before retrying."
                                    ),
                                }
                            )
                continue

            if reply.get("finish_reason") in {"stop", "length"}:
                break

        return {
            "messages": messages,
            "final_report": final_report,
            "observations": self.env.summarize_state(),
        }


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
#%%
def run_default_agent(
    instruction: str,
    *,
    budget: Optional[int] = None,
    config: Optional[AgentConfig] = None,
) -> Dict[str, Any]:
    """One-liner helper that instantiates and runs the default agent."""
    agent = LLMControlledBOAgent(config=config)
    return agent.run_session(instruction, budget=budget)


__all__ = [
    "AgentConfig",
    "LLMControlledBOAgent",
    "BayesianOptimizationEnvironment",
    "run_default_agent",
]

# %%

result = run_default_agent(
    instruction=(
        "Design a Bayesian optimization loop on the P3HT lookup oracle. "
        "Respect a budget of 15 evaluations and hand back the best recipe."
    ),
    budget=15,
)

# %%
