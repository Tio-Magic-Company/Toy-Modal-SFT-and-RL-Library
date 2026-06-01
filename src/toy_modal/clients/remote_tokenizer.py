"""Small tokenizer proxy for deployed transports."""

from __future__ import annotations

from typing import Any


class RemoteTokenizer:
    """Tokenizer-like encode/decode facade backed by transport routes."""

    def __init__(
        self,
        *,
        transport,
        base_model: str | None = None,
        model_path: str | None = None,
    ) -> None:
        self._transport = transport
        self.base_model = base_model
        self.model_path = model_path

    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        response = self._transport.submit(
            "tokenizer.encode",
            {
                "base_model": self.base_model,
                "model_path": self.model_path,
                "text": text,
            },
            result_type=dict,
        ).result()
        return [int(token) for token in response["tokens"]]

    def decode(self, tokens: list[int], *args: Any, **kwargs: Any) -> str:
        response = self._transport.submit(
            "tokenizer.decode",
            {
                "base_model": self.base_model,
                "model_path": self.model_path,
                "tokens": tokens,
            },
            result_type=dict,
        ).result()
        return str(response["text"])
