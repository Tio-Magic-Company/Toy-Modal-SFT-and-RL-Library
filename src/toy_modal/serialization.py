"""Small helpers for serializing Pydantic payloads and decoding results."""

from __future__ import annotations

from typing import Any, get_args, get_origin

from pydantic import BaseModel


def to_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [to_payload(item) for item in value]
    if isinstance(value, tuple):
        return [to_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: to_payload(item) for key, item in value.items()}
    return value


def decode_result(value: Any, result_type: type[Any] | Any) -> Any:
    if result_type is Any or result_type is None:
        return value

    origin = get_origin(result_type)
    if origin is list:
        (inner_type,) = get_args(result_type) or (Any,)
        return [decode_result(item, inner_type) for item in value]

    if isinstance(result_type, type) and issubclass(result_type, BaseModel):
        if isinstance(value, result_type):
            return value
        return result_type.model_validate(value)

    return value
