"""
CostTracker — callback handler для подсчёта токенов и стоимости.
Разбивает затраты по узлам графа (classify, retrieve, grade, generate и т.д.).
"""
import time
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# ---------------------------------------------------------------
# Цены GigaChat-2 (Lite) — ₽ за 1 токен
# 65 ₽ за 1 млн токенов = 0.000065 ₽ за токен
# ---------------------------------------------------------------
PRICE_INPUT_PER_TOKEN = 0.000065
PRICE_OUTPUT_PER_TOKEN = 0.000065


class CostTracker(BaseCallbackHandler):
    """
    Считает токены и стоимость по каждому узлу графа.
    LangGraph передаёт имя узла через run_name в on_chain_start.
    """

    def __init__(self):
        super().__init__()
        # run_id → имя текущего узла
        self._run_to_node: dict[str, str] = {}
        # имя узла → {"input": int, "output": int, "calls": int, "time": float}
        self.node_stats: dict[str, dict] = defaultdict(
            lambda: {"input": 0, "output": 0, "calls": 0, "time": 0.0}
        )
        # общие счётчики
        self.total_input = 0
        self.total_output = 0
        self.total_calls = 0
        self.total_time = 0.0
        self._llm_start_times: dict[str, float] = {}

    # -----------------------------------------------------------
    # LangGraph: on_chain_start при входе в узел
    # -----------------------------------------------------------
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        name = kwargs.get("name") or (serialized or {}).get("name", "unknown")
        if run_id:
            self._run_to_node[run_id] = name

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._run_to_node.pop(run_id, None)

    # -----------------------------------------------------------
    # LLM: on_llm_start / on_llm_end
    # -----------------------------------------------------------
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        parent_run_id = str(kwargs.get("parent_run_id", ""))
        node = self._run_to_node.get(parent_run_id, "unknown")
        self._llm_start_times[run_id] = time.time()
        # запоминаем узел для этого LLM-вызова
        self._run_to_node[f"llm:{run_id}"] = node

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        node = self._run_to_node.pop(f"llm:{run_id}", "unknown")
        start = self._llm_start_times.pop(run_id, None)
        elapsed = time.time() - start if start else 0.0

        # Достаём usage_metadata из ответа
        input_tokens = 0
        output_tokens = 0
        try:
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            if message is not None:
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    input_tokens = usage.get("input_tokens", 0) or 0
                    output_tokens = usage.get("output_tokens", 0) or 0
        except (IndexError, AttributeError):
            pass

        # Fallback: llm_output.token_usage
        if input_tokens == 0 and output_tokens == 0:
            llm_out = response.llm_output or {}
            usage = llm_out.get("token_usage", {}) or {}
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0

        # Аккумулируем
        stats = self.node_stats[node]
        stats["input"] += input_tokens
        stats["output"] += output_tokens
        stats["calls"] += 1
        stats["time"] += elapsed

        self.total_input += input_tokens
        self.total_output += output_tokens
        self.total_calls += 1
        self.total_time += elapsed

    # -----------------------------------------------------------
    # Стоимость
    # -----------------------------------------------------------
    @property
    def total_cost(self) -> float:
        return (
            self.total_input * PRICE_INPUT_PER_TOKEN
            + self.total_output * PRICE_OUTPUT_PER_TOKEN
        )

    def node_cost(self, node: str) -> float:
        s = self.node_stats.get(node, {})
        return (
            s.get("input", 0) * PRICE_INPUT_PER_TOKEN
            + s.get("output", 0) * PRICE_OUTPUT_PER_TOKEN
        )

    def to_dict(self) -> dict:
        """Для сохранения в session_state / JSON."""
        return {
            "nodes": {
                name: {
                    **stats,
                    "cost": self.node_cost(name),
                }
                for name, stats in self.node_stats.items()
            },
            "total": {
                "input": self.total_input,
                "output": self.total_output,
                "calls": self.total_calls,
                "time": self.total_time,
                "cost": self.total_cost,
            },
        }