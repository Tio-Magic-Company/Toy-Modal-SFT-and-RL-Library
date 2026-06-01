"""Clean-room chat renderers for tutorial-style workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, Sequence

from toy_modal import types


@dataclass(frozen=True)
class TextPart:
    type: str = "text"
    text: str = ""


@dataclass(frozen=True)
class ImagePart:
    type: str = "image"
    image: str = ""


MessageContent = str | Sequence[TextPart | ImagePart | dict[str, object]]


@dataclass(frozen=True)
class Message:
    role: str
    content: MessageContent
    train: bool | None = None


class TrainOnWhat(str, Enum):
    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    LAST_ASSISTANT_TURN = "last_assistant_turn"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"
    ALL_MESSAGES = "all_messages"
    ALL_TOKENS = "all_tokens"
    CUSTOMIZED = "customized"


class ParseTermination(str, Enum):
    STOP_SEQUENCE = "stop_sequence"
    EOS = "eos"
    MALFORMED = "malformed"

    @property
    def is_clean(self) -> bool:
        return self in {ParseTermination.STOP_SEQUENCE, ParseTermination.EOS}

    @property
    def is_stop_sequence(self) -> bool:
        return self is ParseTermination.STOP_SEQUENCE


class TokenWeights(list[float]):
    """Small tensor-like list used by tutorials without requiring torch."""

    def tolist(self) -> list[float]:
        return list(self)

    def __gt__(self, value: float) -> "TokenWeights":
        return TokenWeights([1.0 if item > value else 0.0 for item in self])

    def sum(self) -> "ScalarValue":  # type: ignore[override]
        return ScalarValue(float(sum(float(item) for item in self)))


class ScalarValue(float):
    def item(self) -> float:
        return float(self)


class Renderer(Protocol):
    def build_generation_prompt(self, messages: Sequence[Message | dict[str, object]]) -> types.ModelInput: ...

    def get_stop_sequences(self) -> list[str | int]: ...

    def parse_response(self, tokens: Sequence[int]) -> tuple[dict[str, object], ParseTermination]: ...

    def build_supervised_example(
        self,
        messages: Sequence[Message | dict[str, object]],
        *,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[types.ModelInput, TokenWeights]: ...


class RoleColonRenderer:
    """Tokenizer-agnostic renderer using ``Role: content`` lines.

    The class intentionally avoids claiming vendor-specific chat-template
    equivalence. The registry maps common tutorial renderer names to this
    clean-room implementation so tutorial workflows can run against
    ``toy_modal`` locally and on user-owned Modal infrastructure.
    """

    def __init__(
        self,
        tokenizer,
        *,
        role_prefixes: dict[str, str] | None = None,
        assistant_role: str = "assistant",
        stop_sequences: Sequence[str | int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.role_prefixes = role_prefixes or {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
        }
        self.assistant_role = assistant_role
        self._stop_sequences = list(stop_sequences or ["\nUser:", "\nSystem:", "\nTool:"])

    def build_generation_prompt(self, messages: Sequence[Message | dict[str, object]]) -> types.ModelInput:
        normalized = [_normalize_message(item) for item in messages]
        text = self._render_messages(normalized, add_generation_prompt=True)
        return types.ModelInput.from_ints(self.tokenizer.encode(text))

    def get_stop_sequences(self) -> list[str | int]:
        return list(self._stop_sequences)

    def parse_response(self, tokens: Sequence[int]) -> tuple[dict[str, object], ParseTermination]:
        text = self.tokenizer.decode(list(tokens))
        termination = ParseTermination.EOS
        for stop in self._stop_sequences:
            if isinstance(stop, int):
                if stop in tokens:
                    text = self.tokenizer.decode(list(tokens)[: list(tokens).index(stop)])
                    termination = ParseTermination.STOP_SEQUENCE
                    break
            else:
                index = text.find(stop)
                if index >= 0:
                    text = text[:index]
                    termination = ParseTermination.STOP_SEQUENCE
                    break
        return {"role": self.assistant_role, "content": text.strip()}, termination

    def build_supervised_example(
        self,
        messages: Sequence[Message | dict[str, object]],
        *,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[types.ModelInput, TokenWeights]:
        normalized = [_normalize_message(item) for item in messages]
        if not normalized:
            raise ValueError("messages must not be empty")

        train_indexes = self._train_indexes(normalized, train_on_what)
        if not train_indexes and train_on_what is not TrainOnWhat.ALL_TOKENS:
            raise ValueError("messages do not contain trainable content")

        all_tokens: list[int] = []
        weights = TokenWeights()
        for index, message in enumerate(normalized):
            segment = self._render_message(message, is_last=index == len(normalized) - 1)
            tokens = self.tokenizer.encode(segment)
            all_tokens.extend(tokens)
            weight = 1.0 if train_on_what is TrainOnWhat.ALL_TOKENS or index in train_indexes else 0.0
            weights.extend([weight] * len(tokens))
        return types.ModelInput.from_ints(all_tokens), weights

    def conversation_to_datum(
        self,
        messages: Sequence[Message | dict[str, object]],
        *,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> types.Datum:
        model_input, weights = self.build_supervised_example(messages, train_on_what=train_on_what)
        target_tokens = [
            token for token, weight in zip(model_input.to_ints(), weights) if weight > 0
        ]
        return types.Datum(
            model_input=model_input,
            loss_fn_inputs={"target_tokens": target_tokens, "weights": weights.tolist()},
        )

    def _render_messages(self, messages: Sequence[Message], *, add_generation_prompt: bool) -> str:
        parts = [self._render_message(message, is_last=False).rstrip("\n") for message in messages]
        if add_generation_prompt:
            parts.append(f"{self.role_prefixes.get(self.assistant_role, 'Assistant')}:")
        return "\n".join(parts)

    def _render_message(self, message: Message, *, is_last: bool) -> str:
        prefix = self.role_prefixes.get(message.role, message.role.title())
        suffix = "" if is_last else "\n"
        return f"{prefix}: {get_text_content(message)}{suffix}"

    def _train_indexes(self, messages: Sequence[Message], train_on_what: TrainOnWhat) -> set[int]:
        if train_on_what is TrainOnWhat.ALL_MESSAGES:
            return set(range(len(messages)))
        if train_on_what is TrainOnWhat.CUSTOMIZED:
            return {index for index, message in enumerate(messages) if message.train}
        assistant_indexes = [
            index for index, message in enumerate(messages) if message.role == self.assistant_role
        ]
        if train_on_what is TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            return set(assistant_indexes)
        if train_on_what in {TrainOnWhat.LAST_ASSISTANT_MESSAGE, TrainOnWhat.LAST_ASSISTANT_TURN}:
            return {assistant_indexes[-1]} if assistant_indexes else set()
        return set()


class ChatTemplateRenderer:
    """Renderer that prefers a tokenizer's native chat template.

    Hugging Face chat tokenizers expose ``apply_chat_template`` for model-family
    specific prompt formatting. Toy Modal uses that path when present so Qwen,
    Llama, DeepSeek, and similar tutorial examples follow the base model's own
    tokenizer contract. The role-colon renderer remains the deterministic
    offline fallback for custom simple tokenizers.
    """

    def __init__(
        self,
        tokenizer,
        *,
        name: str = "chat_template",
        image_processor: object | None = None,
        fallback: RoleColonRenderer | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.name = name
        self.image_processor = image_processor
        self.fallback = fallback or RoleColonRenderer(tokenizer)

    def build_generation_prompt(self, messages: Sequence[Message | dict[str, object]]) -> types.ModelInput:
        normalized = [_normalize_message(item) for item in messages]
        tokens = self._apply_template(normalized, add_generation_prompt=True)
        if tokens is None:
            return self.fallback.build_generation_prompt(normalized)
        return types.ModelInput.from_ints(tokens)

    def get_stop_sequences(self) -> list[str | int]:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return [int(eos_token_id)]
        return self.fallback.get_stop_sequences()

    def parse_response(self, tokens: Sequence[int]) -> tuple[dict[str, object], ParseTermination]:
        text = self.tokenizer.decode(list(tokens))
        for stop in self.fallback.get_stop_sequences():
            if isinstance(stop, str) and stop in text:
                return self.fallback.parse_response(tokens)
        return {"role": "assistant", "content": text.strip()}, ParseTermination.EOS

    def build_supervised_example(
        self,
        messages: Sequence[Message | dict[str, object]],
        *,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[types.ModelInput, TokenWeights]:
        normalized = [_normalize_message(item) for item in messages]
        full_tokens = self._apply_template(normalized, add_generation_prompt=False)
        if full_tokens is None:
            return self.fallback.build_supervised_example(normalized, train_on_what=train_on_what)

        train_indexes = self.fallback._train_indexes(normalized, train_on_what)
        if not train_indexes and train_on_what is not TrainOnWhat.ALL_TOKENS:
            raise ValueError("messages do not contain trainable content")

        weights = TokenWeights([1.0 if train_on_what is TrainOnWhat.ALL_TOKENS else 0.0] * len(full_tokens))
        if train_on_what is not TrainOnWhat.ALL_TOKENS:
            for index in train_indexes:
                start = len(self._apply_template(normalized[:index], add_generation_prompt=False) or [])
                end = len(self._apply_template(normalized[: index + 1], add_generation_prompt=False) or [])
                for position in range(start, min(end, len(weights))):
                    weights[position] = 1.0
        return types.ModelInput.from_ints(full_tokens), weights

    def conversation_to_datum(
        self,
        messages: Sequence[Message | dict[str, object]],
        *,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> types.Datum:
        model_input, weights = self.build_supervised_example(messages, train_on_what=train_on_what)
        target_tokens = [
            token for token, weight in zip(model_input.to_ints(), weights) if weight > 0
        ]
        return types.Datum(
            model_input=model_input,
            loss_fn_inputs={"target_tokens": target_tokens, "weights": weights.tolist()},
        )

    def _apply_template(self, messages: Sequence[Message], *, add_generation_prompt: bool) -> list[int] | None:
        template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(template):
            return None
        try:
            rendered = template(
                [_hf_message(message) for message in messages],
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return None
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return [int(token) for token in rendered]


class VLMChatTemplateRenderer(ChatTemplateRenderer):
    """Image-text chat renderer scaffold.

    The renderer preserves image parts for tokenizers/processors that understand
    multimodal chat messages. Token-only local fallbacks stringify image parts as
    placeholders, so this class should be treated as prompt-construction parity
    until a VLM Modal validation run is completed.
    """


RendererFactory = Callable[[object, object | None], Renderer]


_REGISTRY: dict[str, RendererFactory] = {}


def register_renderer(name: str, factory: RendererFactory) -> None:
    if not name:
        raise ValueError("renderer name must not be empty")
    _REGISTRY[name] = factory


def unregister_renderer(name: str) -> None:
    _REGISTRY.pop(name, None)


def get_registered_renderer_names() -> list[str]:
    return sorted(_REGISTRY)


def get_renderer(name: str, tokenizer, image_processor: object | None = None) -> Renderer:
    try:
        return _REGISTRY[name](tokenizer, image_processor)
    except KeyError as exc:
        raise ValueError(f"unknown renderer: {name}") from exc


def get_text_content(message: Message | dict[str, object]) -> str:
    normalized = _normalize_message(message)
    content = normalized.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append(part.text)
        elif isinstance(part, ImagePart):
            parts.append(f"[image:{part.image}]")
        elif isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif part.get("type") == "image":
                parts.append(f"[image:{part.get('image', part.get('url', ''))}]")
    return "".join(parts)


def conversation_to_datum(
    tokenizer,
    messages: Sequence[Message | dict[str, object]],
    *,
    renderer: Renderer | None = None,
    renderer_name: str = "role_colon",
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
) -> types.Datum:
    active_renderer = renderer or get_renderer(renderer_name, tokenizer)
    if hasattr(active_renderer, "conversation_to_datum"):
        return active_renderer.conversation_to_datum(messages, train_on_what=train_on_what)  # type: ignore[attr-defined]
    model_input, weights = active_renderer.build_supervised_example(
        messages, train_on_what=train_on_what
    )
    target_tokens = [token for token, weight in zip(model_input.to_ints(), weights) if weight > 0]
    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights.tolist()},
    )


def _normalize_message(message: Message | dict[str, object]) -> Message:
    if isinstance(message, Message):
        return message
    if hasattr(message, "role") and hasattr(message, "content"):
        return Message(
            role=str(getattr(message, "role")),
            content=getattr(message, "content"),
            train=getattr(message, "train", None),
        )
    return Message(
        role=str(message["role"]),
        content=message.get("content", ""),
        train=message.get("train") if isinstance(message.get("train"), bool) else None,
    )


def _role_colon_factory(tokenizer, image_processor: object | None = None) -> Renderer:
    return RoleColonRenderer(tokenizer)


def _chat_template_factory(name: str) -> RendererFactory:
    def factory(tokenizer, image_processor: object | None = None) -> Renderer:
        return ChatTemplateRenderer(tokenizer, name=name, image_processor=image_processor)

    return factory


def _vlm_chat_template_factory(tokenizer, image_processor: object | None = None) -> Renderer:
    return VLMChatTemplateRenderer(tokenizer, name="qwen3_vl_instruct", image_processor=image_processor)


register_renderer("role_colon", _role_colon_factory)

for _name in (
    "qwen3",
    "qwen3_disable_thinking",
    "qwen3_instruct",
    "qwen3_5",
    "llama3",
    "deepseekv3",
    "deepseekv3_thinking",
    "nemotron3",
    "kimi_k2",
):
    register_renderer(_name, _chat_template_factory(_name))

register_renderer("qwen3_vl_instruct", _vlm_chat_template_factory)


def _hf_message(message: Message) -> dict[str, object]:
    content = message.content
    if isinstance(content, str):
        return {"role": message.role, "content": content}
    parts: list[dict[str, object]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            parts.append({"type": "image", "image": part.image})
        elif isinstance(part, dict):
            parts.append(dict(part))
    return {"role": message.role, "content": parts}
