"""Completer wrappers around :class:`toy_modal.SamplingClient`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, Sequence

from toy_modal import types
from toy_modal.renderers import Message, Renderer, get_text_content


@dataclass(frozen=True)
class TokensWithLogprobs:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str


class TokenCompleter(Protocol):
    async def __call__(
        self,
        prompt: types.ModelInput,
        *,
        stop: Sequence[str | int] | None = None,
    ) -> TokensWithLogprobs: ...


class MessageCompleter(Protocol):
    async def __call__(self, messages: Sequence[Message | dict[str, object]]) -> dict[str, object]: ...


class TinkerTokenCompleter:
    """Async token completer backed by a ``SamplingClient``."""

    def __init__(
        self,
        sampling_client,
        *,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int | None = None,
    ) -> None:
        self.sampling_client = sampling_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed

    async def __call__(
        self,
        prompt: types.ModelInput,
        *,
        stop: Sequence[str | int] | None = None,
    ) -> TokensWithLogprobs:
        string_stops = [item for item in stop or [] if isinstance(item, str)]
        params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            stop=string_stops or None,
        )
        response = await self.sampling_client.sample_async(
            prompt,
            1,
            params,
        )
        sequence = response.sequences[0]
        prompt_length = prompt.length()
        completion_tokens = sequence.tokens[prompt_length:]
        logprobs = sequence.logprobs or [0.0] * len(completion_tokens)
        return TokensWithLogprobs(
            tokens=completion_tokens,
            logprobs=[float(item) for item in logprobs],
            stop_reason=sequence.stop_reason,
        )


class TinkerMessageCompleter:
    """Async message completer backed by a renderer and ``SamplingClient``."""

    def __init__(
        self,
        sampling_client,
        renderer: Renderer,
        *,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int | None = None,
    ) -> None:
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.token_completer = TinkerTokenCompleter(
            sampling_client,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )

    async def __call__(self, messages: Sequence[Message | dict[str, object]]) -> dict[str, object]:
        prompt = await asyncio.to_thread(self.renderer.build_generation_prompt, messages)
        result = await self.token_completer(prompt, stop=self.renderer.get_stop_sequences())
        message, termination = await asyncio.to_thread(self.renderer.parse_response, result.tokens)
        message["termination"] = termination.value
        return message


ToyModalTokenCompleter = TinkerTokenCompleter
ToyModalMessageCompleter = TinkerMessageCompleter


async def judge_reward(
    message_completer: MessageCompleter,
    question: str,
    answer: str,
    *,
    max_score: float = 5.0,
) -> float:
    """Run the common LLM-as-judge pattern and parse a normalized score."""

    import re

    response = await message_completer(
        [
            {
                "role": "user",
                "content": (
                    "Rate this answer from 1 to 5.\n"
                    f"Question: {question}\nAnswer: {answer}\nScore:"
                ),
            }
        ]
    )
    match = re.search(r"[1-5]", get_text_content(response))
    return float(match.group()) / max_score if match else 0.0
