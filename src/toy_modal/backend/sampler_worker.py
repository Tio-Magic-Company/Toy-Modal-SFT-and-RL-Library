"""Sampler engines used by Modal workers and local tiny-backend tests."""

from __future__ import annotations

import base64
from contextlib import nullcontext
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from toy_modal import types
from toy_modal.backend.losses import token_logprobs_from_logits
from toy_modal.errors import BackendUnavailableError, BadRequestError, CheckpointNotFoundError
from toy_modal.paths import parse_toy_path


class TinySamplerEngine:
    """Deterministic sampler for local smoke tests and route-shape checks."""

    @classmethod
    def load(
        cls,
        *,
        base_model: str | None,
        model_path: str | None,
        model_root: str,
        run_root: str,
    ) -> "TinySamplerEngine":
        return cls(base_model=base_model, model_path=model_path, model_root=model_root, run_root=run_root)

    def __init__(
        self,
        *,
        base_model: str | None,
        model_path: str | None,
        model_root: str = "/models",
        run_root: str = "/runs",
    ) -> None:
        self.base_model = base_model
        self.model_path = model_path
        self.model_root = Path(model_root)
        self.run_root = Path(run_root)
        self.manifest = _load_manifest_from_path(model_path, self.run_root) if model_path else None

    def sample(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = types.SamplingParams.model_validate(payload["sampling_params"])
        prompt = types.ModelInput.model_validate(payload["prompt"]).to_ints()
        seed = params.seed if params.seed is not None else (sum(prompt) + len(prompt))
        generator = random.Random(seed)
        sequences = []
        for index in range(payload["num_samples"]):
            completion = self._generate_tokens(prompt, params, generator, index)
            tokens = [*prompt, *completion]
            logprobs = [-0.1 - ((token % 17) / 100.0) for token in completion]
            sequences.append(
                types.SampledSequence(
                    stop_reason="length" if len(completion) >= params.max_tokens else "stop",
                    tokens=tokens,
                    logprobs=logprobs,
                )
            )
        response = types.SampleResponse(sequences=sequences)
        if payload.get("include_prompt_logprobs"):
            response.prompt_logprobs = self.compute_logprobs({"prompt": payload["prompt"]})
        topk = int(payload.get("topk_prompt_logprobs") or 0)
        if topk > 0:
            response.topk_prompt_logprobs = [
                None if index == 0 else [(max(0, token - offset), -0.5 - offset) for offset in range(topk)]
                for index, token in enumerate(prompt)
            ]
        return response.model_dump(mode="json")

    def compute_logprobs(self, payload: dict[str, Any]) -> list[float | None]:
        tokens = types.ModelInput.model_validate(payload["prompt"]).to_ints()
        if not tokens:
            return []
        return [None, *[-0.25 - ((token % 13) / 100.0) for token in tokens[1:]]]

    def _generate_tokens(
        self,
        prompt: list[int],
        params: types.SamplingParams,
        generator: random.Random,
        sample_index: int,
    ) -> list[int]:
        completion: list[int] = []
        for offset in range(params.max_tokens):
            base = prompt[-1] if prompt else 65
            token = 32 + ((base + offset + sample_index + generator.randint(0, 7)) % 95)
            completion.append(token)
        if params.stop:
            text = bytes(completion).decode("utf-8", errors="ignore")
            for stop in params.stop:
                location = text.find(stop)
                if location >= 0:
                    return list(text[:location].encode("utf-8"))
        return completion


class TransformersSamplerEngine:
    """Transformers-backed causal-LM sampler with optional PEFT adapter loading."""

    @classmethod
    def load(
        cls,
        *,
        base_model: str | None,
        model_path: str | None,
        model_root: str,
        run_root: str,
    ) -> "TransformersSamplerEngine":
        return cls(base_model=base_model, model_path=model_path, model_root=model_root, run_root=run_root)

    def __init__(
        self,
        *,
        base_model: str | None,
        model_path: str | None,
        model_root: str = "/models",
        run_root: str = "/runs",
    ) -> None:
        self.requested_base_model = base_model
        self.model_path = model_path
        self.model_root = Path(model_root)
        self.run_root = Path(run_root)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.base_model, self.adapter_dir, self.manifest = _resolve_model_reference(
            base_model=base_model,
            model_path=model_path,
            run_root=self.run_root,
        )
        if self.base_model is None:
            raise BadRequestError("Transformers sampler requires base_model or model_path")

    def sample(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = types.SamplingParams.model_validate(payload["sampling_params"])
        prompt = types.ModelInput.model_validate(payload["prompt"]).to_ints()
        num_samples = int(payload["num_samples"])
        if num_samples <= 0:
            raise BadRequestError("num_samples must be positive")

        self._ensure_model_loaded()
        sequences = []
        for sample_index in range(num_samples):
            seed = None if params.seed is None else params.seed + sample_index
            completion, stop_reason = self._generate_completion(prompt, params, seed)
            tokens = [*prompt, *completion]
            scores = self._score_token_ids(tokens)
            completion_scores = [
                float(value) if value is not None else 0.0
                for value in scores[len(prompt) :]
            ]
            sequences.append(
                types.SampledSequence(
                    stop_reason=stop_reason,
                    tokens=tokens,
                    logprobs=completion_scores,
                )
            )

        response = types.SampleResponse(sequences=sequences)
        if payload.get("include_prompt_logprobs"):
            response.prompt_logprobs = self._score_token_ids(prompt)
        topk = int(payload.get("topk_prompt_logprobs") or 0)
        if topk > 0:
            response.topk_prompt_logprobs = self._topk_prompt_logprobs(prompt, topk)
        return response.model_dump(mode="json")

    def compute_logprobs(self, payload: dict[str, Any]) -> list[float | None]:
        self._ensure_model_loaded()
        tokens = types.ModelInput.model_validate(payload["prompt"]).to_ints()
        return self._score_token_ids(tokens)

    def _generate_completion(
        self,
        prompt: list[int],
        params: types.SamplingParams,
        seed: int | None,
    ) -> tuple[list[int], str]:
        torch, _, _, _ = _backend_deps()
        if params.max_tokens <= 0:
            return [], "length"

        generation_prompt = list(prompt)
        if not generation_prompt:
            bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            if bos_token_id is None:
                raise BadRequestError("empty prompts require a tokenizer bos_token_id or eos_token_id")
            generation_prompt = [int(bos_token_id)]

        input_ids = torch.tensor([generation_prompt], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        do_sample = params.temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": params.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._pad_token_id(),
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if do_sample:
            generation_kwargs["temperature"] = max(float(params.temperature), 1e-6)
            if params.top_k is not None and params.top_k > 0:
                generation_kwargs["top_k"] = int(params.top_k)
            if params.top_p is not None and params.top_p < 1.0:
                generation_kwargs["top_p"] = float(params.top_p)

        context = _seed_context(torch, self.device, seed)
        with context:
            if seed is not None:
                torch.manual_seed(seed)
                if getattr(self.device, "type", None) == "cuda":
                    torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                output = self.model.generate(**generation_kwargs)

        sequence = output.sequences[0].detach().cpu().tolist()
        completion = sequence[len(generation_prompt) :]
        completion, stopped = self._truncate_at_stop(completion, params.stop)
        if stopped:
            stop_reason = "stop"
        elif len(completion) >= params.max_tokens:
            stop_reason = "length"
        else:
            stop_reason = "eos"
        return completion, stop_reason

    def _truncate_at_stop(self, completion: list[int], stops: list[str] | None) -> tuple[list[int], bool]:
        if not stops or not completion:
            return completion, False
        text = self.tokenizer.decode(completion, skip_special_tokens=False)
        locations = [text.find(stop) for stop in stops if stop]
        locations = [location for location in locations if location >= 0]
        if not locations:
            return completion, False
        prefix = text[: min(locations)]
        return self.tokenizer.encode(prefix, add_special_tokens=False), True

    def _score_token_ids(self, tokens: list[int]) -> list[float | None]:
        torch, _, _, _ = _backend_deps()
        if not tokens:
            return []
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        labels = input_ids.clone()
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits
            gathered, mask = token_logprobs_from_logits(logits, labels)
        scores: list[float | None] = [None]
        for value, active in zip(gathered[0], mask[0]):
            scores.append(float(value.detach().cpu()) if bool(active.detach().cpu()) else None)
        return scores

    def _topk_prompt_logprobs(self, tokens: list[int], topk: int) -> list[list[tuple[int, float]] | None]:
        torch, _, _, _ = _backend_deps()
        if not tokens:
            return []
        if len(tokens) == 1:
            return [None]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits[:, :-1, :]
            logprobs = torch.log_softmax(logits, dim=-1)[0]
            k = min(int(topk), int(logprobs.shape[-1]))
            values, indices = torch.topk(logprobs, k=k, dim=-1)
        rows: list[list[tuple[int, float]] | None] = [None]
        for row_values, row_indices in zip(values, indices):
            rows.append(
                [
                    (int(token_id.detach().cpu()), float(logprob.detach().cpu()))
                    for token_id, logprob in zip(row_indices, row_values)
                ]
            )
        return rows

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return
        torch, AutoModelForCausalLM, AutoTokenizer, PeftModel = _backend_deps()
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        pretrained_kwargs: dict[str, Any] = {"cache_dir": str(self.model_root)}
        if token:
            pretrained_kwargs["token"] = token

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, **pretrained_kwargs)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model, **pretrained_kwargs)
        except Exception as exc:
            raise BackendUnavailableError(
                f"failed to load base model/tokenizer {self.base_model!r}"
            ) from exc

        if self.tokenizer is not None and getattr(base_model.config, "pad_token_id", None) is None:
            base_model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.adapter_dir is not None:
            self.model = PeftModel.from_pretrained(base_model, str(self.adapter_dir), is_trainable=False)
        else:
            self.model = base_model
        self.model.to(self.device)
        self.model.eval()

    def _pad_token_id(self) -> int:
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        if getattr(self.model.config, "pad_token_id", None) is not None:
            return int(self.model.config.pad_token_id)
        if getattr(self.model.config, "eos_token_id", None) is not None:
            return int(self.model.config.eos_token_id)
        return 0


def load_sampler_engine(
    engine_name: str,
    *,
    base_model: str | None,
    model_path: str | None,
    model_root: str,
    run_root: str,
):
    normalized = engine_name.lower().replace("_", "-")
    if normalized in {"tiny", "deterministic"}:
        return TinySamplerEngine.load(
            base_model=base_model,
            model_path=model_path,
            model_root=model_root,
            run_root=run_root,
        )
    if normalized in {"transformers", "hf", "huggingface"}:
        return TransformersSamplerEngine.load(
            base_model=base_model,
            model_path=model_path,
            model_root=model_root,
            run_root=run_root,
        )
    if normalized in {"unsloth", "unsloth-hf"}:
        from toy_modal.backend.unsloth_engines import UnslothSamplerEngine

        return UnslothSamplerEngine.load(
            base_model=base_model,
            model_path=model_path,
            model_root=model_root,
            run_root=run_root,
        )
    raise BadRequestError(f"unsupported sampler engine: {engine_name!r}")


def tokenizer_for_reference(
    *,
    base_model: str | None,
    model_path: str | None,
    model_root: str,
    run_root: str,
):
    _, AutoTokenizer, _, _ = _tokenizer_deps()
    resolved_base_model, _, _ = _resolve_model_reference(
        base_model=base_model,
        model_path=model_path,
        run_root=Path(run_root),
    )
    if resolved_base_model is None:
        raise BadRequestError("tokenizer lookup requires base_model or model_path")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    pretrained_kwargs: dict[str, Any] = {"cache_dir": str(model_root)}
    if token:
        pretrained_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(resolved_base_model, **pretrained_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_model_reference(
    *,
    base_model: str | None,
    model_path: str | None,
    run_root: Path,
) -> tuple[str | None, Path | None, dict[str, Any] | None]:
    if not model_path:
        return base_model, None, None
    manifest = _load_manifest_from_path(model_path, run_root)
    if manifest is None:
        raise CheckpointNotFoundError(model_path)
    adapter_name = manifest.get("adapter_path", "adapter")
    parsed = parse_toy_path(model_path)
    artifact_dir = (
        run_root
        / parsed.project_id
        / parsed.run_id
        / parsed.artifact_type
        / parsed.name
    )
    adapter_dir = _wait_for_adapter_dir(artifact_dir, adapter_name)
    if adapter_dir is None:
        adapter_dir = _materialize_embedded_adapter(manifest, model_path)
    if adapter_dir is None:
        entries = _artifact_entries(artifact_dir)
        raise CheckpointNotFoundError(
            f"adapter directory missing for {model_path}; "
            f"checked {artifact_dir / adapter_name} and {artifact_dir}; "
            f"visible entries: {entries}"
        )
    return manifest.get("base_model", base_model), adapter_dir, manifest


def _wait_for_adapter_dir(artifact_dir: Path, adapter_name: str) -> Path | None:
    deadline = time.monotonic() + _artifact_wait_seconds()
    while True:
        adapter_dir = _find_adapter_dir(artifact_dir, adapter_name)
        if adapter_dir is not None:
            return adapter_dir
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.5)


def _find_adapter_dir(artifact_dir: Path, adapter_name: str) -> Path | None:
    candidates = [artifact_dir / adapter_name, artifact_dir]
    if artifact_dir.exists():
        candidates.extend(path for path in artifact_dir.iterdir() if path.is_dir())
    for candidate in candidates:
        if _looks_like_peft_adapter(candidate):
            return candidate
    return None


def _looks_like_peft_adapter(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def _materialize_embedded_adapter(manifest: dict[str, Any], model_path: str) -> Path | None:
    files = manifest.get("adapter_files") or []
    if not files:
        return None
    digest = hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:16]
    target = Path("/tmp") / "toy_modal_adapters" / digest
    target.mkdir(parents=True, exist_ok=True)
    for item in files:
        if item.get("encoding") != "base64" or not item.get("path"):
            continue
        relative_path = Path(item["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise CheckpointNotFoundError(f"unsafe embedded adapter path for {model_path}: {item['path']!r}")
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(base64.b64decode(item.get("data", "")))
    return target if _looks_like_peft_adapter(target) else None


def _artifact_wait_seconds() -> float:
    configured = os.getenv("TOY_MODAL_ARTIFACT_WAIT_SECONDS")
    if configured is not None:
        try:
            return max(0.0, float(configured))
        except ValueError:
            return 0.0
    return 30.0 if os.getenv("MODAL_ENVIRONMENT") else 0.0


def _artifact_entries(artifact_dir: Path) -> list[str]:
    if not artifact_dir.exists():
        return ["<artifact directory missing>"]
    return sorted(str(path.relative_to(artifact_dir)) for path in artifact_dir.rglob("*"))[:50]


def _load_manifest_from_path(model_path: str | None, run_root: Path) -> dict[str, Any] | None:
    if not model_path or "://" not in model_path:
        return None
    parsed = parse_toy_path(model_path)
    manifest = (
        run_root
        / parsed.project_id
        / parsed.run_id
        / parsed.artifact_type
        / parsed.name
        / "manifest.json"
    )
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        from toy_modal.backend.storage import ArtifactStore

        store = ArtifactStore.from_runs_root(run_root)
        store.raise_if_manifest_expired(payload)
        store.validate_manifest_files(payload, manifest.parent)
        return payload
    return None


def _seed_context(torch: Any, device: Any, seed: int | None):
    if seed is None:
        return nullcontext()
    if getattr(device, "type", None) != "cuda":
        return torch.random.fork_rng(devices=[])
    index = 0 if getattr(device, "index", None) is None else int(device.index)
    return torch.random.fork_rng(devices=[index])


def _backend_deps():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise BackendUnavailableError(
            "TransformersSamplerEngine requires optional backend dependencies: "
            "torch, transformers, and peft"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def _tokenizer_deps():
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BackendUnavailableError(
            "remote tokenizer lookup requires optional backend dependencies: torch and transformers"
        ) from exc
    return torch, AutoTokenizer, None, None


# Backwards-compatible name used by existing tiny-backend tests.
SamplerEngine = TinySamplerEngine
