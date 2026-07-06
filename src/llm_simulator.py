"""
Simulated LLM client.

IMPORTANT: This project makes zero external API calls. Every "LLM response"
is produced by deterministic, rule-based logic in briefing_generator.py and
quality_reviewer.py. This module exists to give that deterministic logic a
realistic *shape* -- a prompt goes in, a response comes out, and every call
is metered for tokens and cost -- exactly like a real LLM integration would
be, so the surrounding pipeline (logging, cost tracking, prompt templates)
is a faithful stand-in for a production integration and a real API client
could be swapped in behind this same interface later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List

from src import config


def estimate_tokens(text: str) -> int:
    """Rough words->tokens estimate. Good enough for cost-log demonstration
    purposes; not a substitute for a real tokenizer."""
    word_count = len(text.split())
    return max(1, math.ceil(word_count * config.TOKENS_PER_WORD))


@dataclass
class LLMCallRecord:
    timestamp: str
    task: str
    reference_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    def as_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "task": self.task,
            "reference_id": self.reference_id,
            "model": config.SIMULATED_MODEL_NAME,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


@dataclass
class SimulatedLLMClient:
    """Wraps a deterministic 'response generator' function with prompt
    construction bookkeeping and token/cost logging, mimicking the call
    signature of a real LLM client (`.call(prompt) -> response`)."""

    call_log: List[LLMCallRecord] = field(default_factory=list)

    def call(self, task: str, reference_id: str, prompt: str,
              response_fn: Callable[[], str]) -> str:
        """
        Args:
            task: short label for what kind of call this is
                  (e.g. "account_briefing", "quality_review").
            reference_id: the account_id / output_id this call is about,
                          for traceability in the log.
            prompt: the fully-rendered prompt text (from prompts.py) that
                    would be sent to a real model. Logged for token counting
                    and for the prompt audit trail.
            response_fn: zero-arg callable that deterministically produces
                         the "response" text. Kept as a callback so this
                         module never needs to know the domain logic that
                         generates briefings vs. quality verdicts.
        Returns:
            The response text produced by response_fn().
        """
        response_text = response_fn()

        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(response_text)
        total_tokens = prompt_tokens + completion_tokens
        cost = (
            (prompt_tokens / 1000) * config.SIMULATED_INPUT_RATE_PER_1K_TOKENS
            + (completion_tokens / 1000) * config.SIMULATED_OUTPUT_RATE_PER_1K_TOKENS
        )

        record = LLMCallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            task=task,
            reference_id=reference_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
        )
        self.call_log.append(record)

        return response_text

    def total_cost(self) -> float:
        return sum(r.estimated_cost_usd for r in self.call_log)

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.call_log)

    def summary(self) -> dict:
        return {
            "model": config.SIMULATED_MODEL_NAME,
            "total_calls": len(self.call_log),
            "total_prompt_tokens": sum(r.prompt_tokens for r in self.call_log),
            "total_completion_tokens": sum(r.completion_tokens for r in self.call_log),
            "total_tokens": self.total_tokens(),
            "total_estimated_cost_usd": round(self.total_cost(), 6),
        }
