"""Test-only fake Modal module for no-credential client workflow tests."""

from __future__ import annotations

import random
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from toy_modal import types
from toy_modal.backend.loss_inputs import validate_training_batch
from toy_modal.errors import (
    BadRequestError,
    CheckpointNotFoundError,
    DependencyFailedError,
    NotFoundError,
    RunNotFoundError,
    StaleModelSequenceError,
)
from toy_modal.paths import build_toy_path, parse_toy_path
from toy_modal.serialization import to_payload
from toy_modal.defaults import (
    DEFAULT_TRANSFORMERS_BASE_MODEL,
    DEFAULT_UNSLOTH_CAPABILITY_MODELS,
    DEFAULT_UNSLOTH_MODEL_FAMILIES,
)

DEFAULT_BASE_MODEL = DEFAULT_TRANSFORMERS_BASE_MODEL


def _uses_unsloth(engine_name: str) -> bool:
    return engine_name.lower().replace("_", "-").startswith("unsloth")


def install_fake_modal(monkeypatch) -> "FakeModalBackend":
    backend = FakeModalBackend()
    monkeypatch.setitem(sys.modules, "modal", backend.module)
    return backend


class TinyTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(max(0, min(255, token)) for token in tokens).decode(
            "utf-8",
            errors="replace",
        )


class _AsyncCallable:
    def __init__(self, callback, aio_callback=None) -> None:
        self._callback = callback
        self._aio_callback = aio_callback

    def __call__(self, *args, **kwargs):
        return self._callback(*args, **kwargs)

    async def aio(self, *args, **kwargs):
        if self._aio_callback is not None:
            self._aio_callback()
        return self._callback(*args, **kwargs)


class FakeCall:
    def __init__(self, backend: "FakeModalBackend", route: str, payload: dict[str, Any]) -> None:
        self.object_id = f"fc-{uuid4().hex}"
        self._backend = backend
        self._route = route
        self._payload = payload
        self._done = False
        self._cancelled = False
        self._result: Any = None
        self._exception: BaseException | None = None
        self.get = _AsyncCallable(self._get, backend._record_async_get)
        backend.calls[self.object_id] = self

    def _get(self, timeout: float | None = None) -> Any:
        if not self._done and not self._cancelled:
            try:
                self._result = self._backend.handle(self._route, self._payload, self.object_id)
            except BaseException as exc:
                self._exception = exc
            self._done = True
        if self._exception is not None:
            raise self._exception
        return self._result

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self._cancelled = True
        self._done = True


class FakeModalBackend:
    def __init__(self) -> None:
        self.calls: dict[str, FakeCall] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._checkpoints: dict[str, list[dict[str, Any]]] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._samplers: dict[str, dict[str, Any]] = {}
        self._tokenizer = TinyTokenizer()
        self.function_calls: list[tuple[str, str, str | None, dict[str, Any]]] = []
        self.class_calls: list[tuple[str, str, str | None, dict[str, str], str]] = []
        self.async_function_spawns = 0
        self.async_method_spawns = 0
        self.async_gets = 0

        backend = self

        class Function:
            @staticmethod
            def from_name(app_name, function_name, environment_name=None):
                return _FunctionHandle(backend, app_name, function_name, environment_name)

        class Cls:
            @staticmethod
            def from_name(app_name, class_name, environment_name=None):
                return _ClassHandle(backend, app_name, class_name, environment_name)

        class FunctionCall:
            @staticmethod
            def from_id(object_id):
                return backend.calls[object_id]

        self.module = SimpleNamespace(
            Function=Function,
            Cls=Cls,
            FunctionCall=FunctionCall,
            __version__="fake",
        )

    def _record_async_function_spawn(self) -> None:
        self.async_function_spawns += 1

    def _record_async_method_spawn(self) -> None:
        self.async_method_spawns += 1

    def _record_async_get(self) -> None:
        self.async_gets += 1

    def spawn_function(
        self,
        app_name: str,
        function_name: str,
        environment_name: str | None,
        payload: dict[str, Any],
    ) -> FakeCall:
        self.function_calls.append((app_name, function_name, environment_name, payload))
        routes = {
            "get_server_capabilities": "server.capabilities",
            "create_lora_training_run": "training.create_lora",
            "create_training_run_from_state": "training.load_state",
            "metadata_route": "metadata.route",
            "tokenizer_route": "tokenizer.route",
            "prefetch_model": "prefetch_model",
        }
        return FakeCall(self, routes[function_name], to_payload(payload))

    def spawn_method(
        self,
        app_name: str,
        class_name: str,
        environment_name: str | None,
        params: dict[str, str],
        method_name: str,
        payload: dict[str, Any],
    ) -> FakeCall:
        self.class_calls.append((app_name, class_name, environment_name, params, method_name))
        routes = {
            "forward": "training.forward",
            "forward_backward": "training.forward_backward",
            "optim_step": "training.optim_step",
            "save_state": "training.save_state",
            "load_state": "training.load_state",
            "save_weights_for_sampler": "training.save_weights_for_sampler",
            "validate_old_logprobs_sequence": "training.validate_old_logprobs_sequence",
            "sample": "sampling.sample",
            "compute_logprobs": "sampling.compute_logprobs",
        }
        return FakeCall(self, routes[method_name], to_payload(payload))

    def handle(self, route: str, payload: dict[str, Any], job_id: str) -> Any:
        if route == "metadata.route":
            return self.handle(payload["route"], payload.get("payload") or {}, job_id)
        if route == "tokenizer.route":
            operation = payload["operation"]
            return self.handle(f"tokenizer.{operation}", payload, job_id)
        handlers = {
            "server.capabilities": self._server_capabilities,
            "training.create_lora": self._create_lora_training_run,
            "training.forward": self._forward,
            "training.forward_backward": self._forward_backward,
            "training.optim_step": self._optim_step,
            "training.save_state": self._save_state,
            "training.load_state": self._load_state,
            "training.save_weights_for_sampler": self._save_weights_for_sampler,
            "training.validate_old_logprobs_sequence": self._validate_old_logprobs_sequence,
            "sampling.sample": self._sample,
            "sampling.compute_logprobs": self._compute_logprobs,
            "tokenizer.encode": self._tokenizer_encode,
            "tokenizer.decode": self._tokenizer_decode,
            "rest.get_training_run": self._get_training_run,
            "rest.get_training_run_by_toy_path": self._get_training_run_by_toy_path,
            "rest.get_weights_info_by_toy_path": self._get_weights_info_by_toy_path,
            "rest.list_training_runs": self._list_training_runs,
            "rest.list_checkpoints": self._list_checkpoints,
            "rest.list_user_checkpoints": self._list_user_checkpoints,
            "rest.get_checkpoint_archive_url": self._get_checkpoint_archive_url,
            "rest.get_checkpoint_archive_url_from_toy_path": self._get_checkpoint_archive_url_from_toy_path,
            "rest.delete_checkpoint": self._delete_checkpoint,
            "rest.delete_checkpoint_from_toy_path": self._delete_checkpoint_from_toy_path,
            "rest.set_checkpoint_ttl_from_toy_path": self._set_checkpoint_ttl,
            "rest.set_checkpoint_public": self._set_checkpoint_public,
            "rest.get_session": self._get_session,
            "rest.list_sessions": self._list_sessions,
            "rest.get_sampler": self._get_sampler,
            "prefetch_model": self._prefetch_model,
        }
        try:
            return handlers[route](payload, job_id)
        except KeyError as exc:
            raise BadRequestError(f"Unsupported route: {route}") from exc

    def _server_capabilities(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        trainer_engine = os.getenv("TOY_MODAL_TRAINER_ENGINE", "unsloth-peft")
        sampler_engine = os.getenv("TOY_MODAL_SAMPLER_ENGINE", "unsloth")
        uses_unsloth = _uses_unsloth(trainer_engine) or _uses_unsloth(sampler_engine)
        backend_profile: dict[str, Any] = {
            "uses_unsloth": uses_unsloth,
            "prefetch_gpu": os.getenv("TOY_MODAL_PREFETCH_GPU")
            or (os.getenv("TOY_MODAL_SAMPLE_GPU", "L40S") if uses_unsloth else None),
        }
        if uses_unsloth:
            backend_profile["unsloth"] = {
                "engine_config": {"load_in_4bit": True},
                "pip_packages": [
                    os.getenv("TOY_MODAL_UNSLOTH_PACKAGE", "unsloth[base]"),
                    os.getenv(
                        "TOY_MODAL_UNSLOTH_BITSANDBYTES_PACKAGE",
                        "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0",
                    ),
                ],
                "package_versions": {"unsloth": None, "unsloth_zoo": None},
                "model_families": list(DEFAULT_UNSLOTH_MODEL_FAMILIES),
            }
        supported_models = payload.get("configured_models") or (
            list(DEFAULT_UNSLOTH_CAPABILITY_MODELS)
            if uses_unsloth
            else [DEFAULT_BASE_MODEL]
        )
        return types.GetServerCapabilitiesResponse(
            supported_models=supported_models,
            supports_lora=True,
            supports_sampling=True,
            supports_importance_sampling=True,
            max_batch_size=payload.get("max_batch_size"),
            transport="modal-direct",
            trainer_engine=trainer_engine,
            sampler_engine=sampler_engine,
            backend_profile=backend_profile,
        ).model_dump(mode="json")

    def _prefetch_model(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        return {
            "model_id": payload["model_id"],
            "include_model": bool(payload.get("include_model", True)),
            "include_tokenizer": bool(payload.get("include_tokenizer", True)),
            "dry_run": bool(payload.get("dry_run", False)),
            "backend": payload.get("backend", "auto"),
            "status": "ok",
        }

    def _create_lora_training_run(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        run_id = f"run_{uuid4().hex[:12]}"
        project_id = payload.get("project_id") or "default"
        lora_config = types.LoraConfig.model_validate(payload["lora_config"])
        self._create_run_record(
            run_id=run_id,
            project_id=project_id,
            base_model=payload["base_model"],
            lora_config=lora_config,
            user_metadata=payload.get("user_metadata") or {},
            model_seq_id=0,
            optimizer_step=0,
            session_id=f"session_{run_id}",
        )
        return types.CreateTrainingRunResponse(
            training_run_id=run_id,
            project_id=project_id,
            base_model=payload["base_model"],
            lora_config=lora_config,
            user_metadata=payload.get("user_metadata") or {},
        ).model_dump(mode="json")

    def _create_run_record(
        self,
        *,
        run_id: str,
        project_id: str,
        base_model: str,
        lora_config: types.LoraConfig,
        user_metadata: dict[str, str],
        model_seq_id: int,
        optimizer_step: int,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        session_id = session_id or f"session_{run_id}"
        run = types.TrainingRun(
            training_run_id=run_id,
            project_id=project_id,
            base_model=base_model,
            lora_rank=lora_config.rank,
            created_at=now,
            updated_at=now,
            model_seq_id=model_seq_id,
            optimizer_step=optimizer_step,
            user_metadata=user_metadata,
        )
        record = {
            **run.model_dump(mode="json"),
            "lora_config": lora_config.model_dump(mode="json"),
            "latest_gradient_id": None,
            "session_id": session_id,
        }
        self._runs[run_id] = record
        self._checkpoints[run_id] = []
        self._sessions[session_id] = {
            "session_id": session_id,
            "training_run_ids": [run_id],
            "sampler_ids": [],
            "user_metadata": user_metadata,
        }
        return record

    def _run(self, run_id: str) -> dict[str, Any]:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise RunNotFoundError(run_id) from exc

    def _public_run(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        public_fields = types.TrainingRun.model_fields
        return {key: value for key, value in run.items() if key in public_fields}

    def _forward(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        return self._loss_response(payload, job_id, with_gradient=False)

    def _forward_backward(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        return self._loss_response(payload, job_id, with_gradient=True)

    def _loss_response(self, payload: dict[str, Any], job_id: str, *, with_gradient: bool) -> dict[str, Any]:
        run = self._run(payload["training_run_id"])
        expected = payload.get("expected_model_seq_id")
        if expected is not None and expected != run["model_seq_id"]:
            raise StaleModelSequenceError(
                f"expected model_seq_id {expected}, current is {run['model_seq_id']}"
            )
        self._raise_stale_old_logprobs(run, payload.get("data", []))
        validate_training_batch(payload.get("data", []), payload.get("loss_fn", "cross_entropy"))
        token_count = sum(
            len(types.ModelInput.model_validate(datum["model_input"]).to_ints())
            for datum in payload.get("data", [])
        )
        loss = max(0.01, 2.0 - (0.05 * run["optimizer_step"])) + (token_count % 17) / 100
        gradient_id = f"grad_{job_id}" if with_gradient else None
        if with_gradient:
            run["latest_gradient_id"] = gradient_id
        run["last_request_time"] = datetime.now(timezone.utc).isoformat()
        return types.ForwardBackwardOutput(
            loss=round(loss, 4),
            metrics={"fake_modal": 1.0},
            num_tokens=token_count,
            gradient_id=gradient_id,
            model_seq_id=run["model_seq_id"],
        ).model_dump(mode="json")

    def _validate_old_logprobs_sequence(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        run = self._run(payload["training_run_id"])
        validate_training_batch(payload.get("data", []), payload.get("loss_fn", "importance_sampling"))
        try:
            self._raise_stale_old_logprobs(run, payload.get("data", []))
        except StaleModelSequenceError as exc:
            return {
                "accepted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_seq_id": run["model_seq_id"],
            }
        return {
            "accepted": True,
            "error_type": None,
            "error": None,
            "model_seq_id": run["model_seq_id"],
        }

    @staticmethod
    def _raise_stale_old_logprobs(run: dict[str, Any], data: list[dict[str, Any]]) -> None:
        for datum_payload in data:
            datum = types.Datum.model_validate(datum_payload)
            old_logprobs_model_seq_id = datum.loss_fn_inputs.get("old_logprobs_model_seq_id")
            if old_logprobs_model_seq_id is not None and int(old_logprobs_model_seq_id) != run["model_seq_id"]:
                raise StaleModelSequenceError(
                    "old_logprobs_model_seq_id "
                    f"{old_logprobs_model_seq_id} does not match current model_seq_id {run['model_seq_id']}"
                )

    def _optim_step(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        dependency = payload.get("depends_on")
        if dependency:
            try:
                self.calls[dependency].get()
            except Exception as exc:
                raise DependencyFailedError(f"dependency failed: {dependency}") from exc
        run = self._run(payload["training_run_id"])
        expected = payload.get("expected_model_seq_id")
        if expected is not None and expected != run["model_seq_id"]:
            raise StaleModelSequenceError(
                f"expected model_seq_id {expected}, current is {run['model_seq_id']}"
            )
        if not run.get("latest_gradient_id"):
            raise BadRequestError("optim_step requires a prior forward_backward gradient")
        run["model_seq_id"] += 1
        run["optimizer_step"] += 1
        run["latest_gradient_id"] = None
        run["updated_at"] = datetime.now(timezone.utc).isoformat()
        return types.OptimStepResponse(
            model_seq_id=run["model_seq_id"],
            optimizer_step=run["optimizer_step"],
            metrics={"fake_modal_optimizer": 1.0},
        ).model_dump(mode="json")

    def _save_state(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        return self._save_checkpoint(payload, checkpoint_type="training")

    def _save_weights_for_sampler(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        return self._save_checkpoint(payload, checkpoint_type="sampler")

    def _save_checkpoint(self, payload: dict[str, Any], *, checkpoint_type: str) -> dict[str, Any]:
        run = self._run(payload["training_run_id"])
        project_id = run["project_id"] or "default"
        name = payload.get("name") or f"seq-{run['model_seq_id']}"
        artifact_type = "checkpoints" if checkpoint_type == "training" else "sampler_weights"
        toy_path = build_toy_path(project_id, run["training_run_id"], artifact_type, name)
        checkpoint = types.Checkpoint(
            checkpoint_id=name,
            checkpoint_type=checkpoint_type,
            toy_path=toy_path,
            size_bytes=128,
        )
        checkpoint_payload = checkpoint.model_dump(mode="json")
        self._checkpoints[run["training_run_id"]] = [
            item for item in self._checkpoints[run["training_run_id"]] if item["toy_path"] != toy_path
        ]
        self._checkpoints[run["training_run_id"]].append(checkpoint_payload)
        self._artifacts[toy_path] = {
            "checkpoint": checkpoint_payload,
            "source_run_id": run["training_run_id"],
            "project_id": project_id,
            "base_model": run["base_model"],
            "lora_config": run["lora_config"],
            "model_seq_id": run["model_seq_id"],
            "optimizer_step": run["optimizer_step"],
            "user_metadata": run["user_metadata"],
        }
        if checkpoint_type == "training":
            run["last_checkpoint"] = checkpoint_payload
        else:
            sampler_id = f"{run['training_run_id']}:sample:{len(self._samplers)}"
            session_id = run.get("session_id") or f"session_{run['training_run_id']}"
            self._samplers[sampler_id] = {
                "sampler_id": sampler_id,
                "base_model": run["base_model"],
                "model_path": toy_path,
                "sampling_session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if sampler_id not in self._sessions[session_id]["sampler_ids"]:
                self._sessions[session_id]["sampler_ids"].append(sampler_id)
            self._artifacts[toy_path]["sampler_id"] = sampler_id
            run["last_sampler_checkpoint"] = checkpoint_payload
        response_cls = types.SaveStateResponse if checkpoint_type == "training" else types.SaveWeightsForSamplerResponse
        return response_cls(
            path=toy_path,
            checkpoint_id=name,
            model_seq_id=run["model_seq_id"],
        ).model_dump(mode="json")

    def _load_state(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        parsed = parse_toy_path(payload["path"], accept_tinker_paths=payload.get("accept_tinker_paths", False))
        artifact_path = build_toy_path(parsed.project_id, parsed.run_id, parsed.artifact_type, parsed.name)
        artifact = self._artifacts.get(artifact_path)
        if artifact is None:
            raise CheckpointNotFoundError(payload["path"])
        target_run_id = payload.get("training_run_id")
        optimizer_step = artifact["optimizer_step"] if payload.get("optimizer") else 0
        if target_run_id:
            run = self._run(target_run_id)
            run["base_model"] = artifact["base_model"]
            run["lora_config"] = artifact["lora_config"]
            run["lora_rank"] = artifact["lora_config"]["rank"]
            run["model_seq_id"] = artifact["model_seq_id"]
            run["optimizer_step"] = optimizer_step
            run["updated_at"] = datetime.now(timezone.utc).isoformat()
        else:
            target_run_id = f"run_{uuid4().hex[:12]}"
            self._create_run_record(
                run_id=target_run_id,
                project_id=payload.get("project_id") or artifact["project_id"],
                base_model=artifact["base_model"],
                lora_config=types.LoraConfig.model_validate(artifact["lora_config"]),
                user_metadata={**artifact["user_metadata"], **(payload.get("user_metadata") or {})},
                model_seq_id=artifact["model_seq_id"],
                optimizer_step=optimizer_step,
                session_id=f"session_{target_run_id}",
            )
        return types.LoadStateResponse(
            path=payload["path"],
            training_run_id=target_run_id,
            model_seq_id=artifact["model_seq_id"],
            optimizer_step=optimizer_step,
            lora_config=types.LoraConfig.model_validate(artifact["lora_config"]),
        ).model_dump(mode="json")

    def _sample(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        params = types.SamplingParams.model_validate(payload["sampling_params"])
        prompt = types.ModelInput.model_validate(payload["prompt"]).to_ints()
        generator = random.Random(params.seed if params.seed is not None else sum(prompt) + len(prompt))
        sequences = []
        for sample_index in range(payload["num_samples"]):
            suffix = []
            for offset in range(params.max_tokens):
                base = prompt[-1] if prompt else 65
                suffix.append(32 + ((base + sample_index + offset + generator.randint(0, 7)) % 95))
            sequences.append(
                types.SampledSequence(
                    stop_reason="length",
                    tokens=[*prompt, *suffix],
                    logprobs=[-0.1 - ((token % 17) / 100.0) for token in suffix],
                )
            )
        response = types.SampleResponse(sequences=sequences)
        if payload.get("include_prompt_logprobs"):
            response.prompt_logprobs = [None, *([-1.0] * max(0, len(prompt) - 1))]
        topk = int(payload.get("topk_prompt_logprobs") or 0)
        if topk > 0:
            response.topk_prompt_logprobs = [
                None if index == 0 else [(max(0, token - rank), -1.0 - rank) for rank in range(topk)]
                for index, token in enumerate(prompt)
            ]
        return response.model_dump(mode="json")

    def _compute_logprobs(self, payload: dict[str, Any], job_id: str) -> list[float | None]:
        tokens = types.ModelInput.model_validate(payload["prompt"]).to_ints()
        if not tokens:
            return []
        return [None, *([-1.0] * (len(tokens) - 1))]

    def _tokenizer_encode(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        return {"tokens": self._tokenizer.encode(payload["text"])}

    def _tokenizer_decode(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        return {"text": self._tokenizer.decode(payload["tokens"])}

    def _get_training_run(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        return self._public_run(payload["training_run_id"])

    def _get_training_run_by_toy_path(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        parsed = parse_toy_path(payload["toy_path"], accept_tinker_paths=payload.get("accept_tinker_paths", False))
        return self._public_run(parsed.run_id)

    def _get_weights_info_by_toy_path(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        parsed = parse_toy_path(payload["toy_path"], accept_tinker_paths=payload.get("accept_tinker_paths", False))
        artifact_path = build_toy_path(parsed.project_id, parsed.run_id, parsed.artifact_type, parsed.name)
        artifact = self._artifacts.get(artifact_path)
        if artifact is None:
            raise CheckpointNotFoundError(payload["toy_path"])
        return types.WeightsInfoResponse(
            path=payload["toy_path"],
            base_model=artifact["base_model"],
            lora_rank=artifact["lora_config"].get("rank"),
            checkpoint_type="training" if parsed.artifact_type == "checkpoints" else "sampler",
        ).model_dump(mode="json")

    def _list_training_runs(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        limit = payload.get("limit", 20)
        offset = payload.get("offset", 0)
        runs = list(self._runs.values())
        public_fields = types.TrainingRun.model_fields
        return types.TrainingRunsResponse(
            training_runs=[
                types.TrainingRun.model_validate({key: value for key, value in run.items() if key in public_fields})
                for run in runs[offset : offset + limit]
            ],
            cursor=types.Cursor(limit=limit, offset=offset, total_count=len(runs)),
        ).model_dump(mode="json")

    def _list_checkpoints(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        self._run(payload["training_run_id"])
        checkpoints = self._checkpoints[payload["training_run_id"]]
        return types.CheckpointsListResponse(
            checkpoints=[types.Checkpoint.model_validate(item) for item in checkpoints],
            cursor=types.Cursor(limit=len(checkpoints), offset=0, total_count=len(checkpoints)),
        ).model_dump(mode="json")

    def _list_user_checkpoints(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        limit = payload.get("limit", 20)
        offset = payload.get("offset", 0)
        checkpoints = [item for run_checkpoints in self._checkpoints.values() for item in run_checkpoints]
        return types.CheckpointsListResponse(
            checkpoints=[types.Checkpoint.model_validate(item) for item in checkpoints[offset : offset + limit]],
            cursor=types.Cursor(limit=limit, offset=offset, total_count=len(checkpoints)),
        ).model_dump(mode="json")

    def _get_checkpoint_archive_url(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        self._run(payload["training_run_id"])
        if not any(item["checkpoint_id"] == payload["checkpoint_id"] for item in self._checkpoints[payload["training_run_id"]]):
            raise CheckpointNotFoundError(payload["checkpoint_id"])
        return types.CheckpointArchiveUrlResponse(
            url=f"modal-volume://fake/{payload['training_run_id']}/{payload['checkpoint_id']}"
        ).model_dump(mode="json")

    def _get_checkpoint_archive_url_from_toy_path(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        parsed = parse_toy_path(payload["toy_path"], accept_tinker_paths=payload.get("accept_tinker_paths", False))
        return self._get_checkpoint_archive_url({"training_run_id": parsed.run_id, "checkpoint_id": parsed.name}, job_id)

    def _delete_checkpoint(self, payload: dict[str, Any], job_id: str) -> None:
        checkpoints = self._checkpoints[payload["training_run_id"]]
        deleted_paths = [item["toy_path"] for item in checkpoints if item["checkpoint_id"] == payload["checkpoint_id"]]
        self._checkpoints[payload["training_run_id"]] = [
            item for item in checkpoints if item["checkpoint_id"] != payload["checkpoint_id"]
        ]
        for path in deleted_paths:
            artifact = self._artifacts.pop(path, None)
            sampler_id = artifact.get("sampler_id") if artifact else None
            if sampler_id:
                self._samplers.pop(sampler_id, None)
                for session in self._sessions.values():
                    session["sampler_ids"] = [existing for existing in session["sampler_ids"] if existing != sampler_id]
        return None

    def _delete_checkpoint_from_toy_path(self, payload: dict[str, Any], job_id: str) -> None:
        parsed = parse_toy_path(payload["toy_path"], accept_tinker_paths=payload.get("accept_tinker_paths", False))
        return self._delete_checkpoint({"training_run_id": parsed.run_id, "checkpoint_id": parsed.name}, job_id)

    def _set_checkpoint_ttl(self, payload: dict[str, Any], job_id: str) -> None:
        if payload.get("ttl_seconds") is not None and payload["ttl_seconds"] <= 0:
            raise BadRequestError("ttl_seconds must be positive or None")
        parsed = parse_toy_path(payload["toy_path"], accept_tinker_paths=payload.get("accept_tinker_paths", False))
        artifact_path = build_toy_path(parsed.project_id, parsed.run_id, parsed.artifact_type, parsed.name)
        artifact = self._artifacts.get(artifact_path)
        if artifact is None:
            raise CheckpointNotFoundError(payload["toy_path"])
        expires_at = None
        if payload.get("ttl_seconds") is not None:
            expires_at = datetime.now(timezone.utc).timestamp() + payload["ttl_seconds"]
        artifact["checkpoint"]["expires_at"] = (
            datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat() if expires_at is not None else None
        )
        self._replace_checkpoint(artifact["checkpoint"])
        return None

    def _set_checkpoint_public(self, payload: dict[str, Any], job_id: str) -> None:
        parsed = parse_toy_path(payload["toy_path"], accept_tinker_paths=payload.get("accept_tinker_paths", False))
        artifact_path = build_toy_path(parsed.project_id, parsed.run_id, parsed.artifact_type, parsed.name)
        artifact = self._artifacts.get(artifact_path)
        if artifact is None:
            raise CheckpointNotFoundError(payload["toy_path"])
        artifact["checkpoint"]["public"] = bool(payload["public"])
        self._replace_checkpoint(artifact["checkpoint"])
        return None

    def _replace_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        parsed = parse_toy_path(checkpoint["toy_path"])
        checkpoints = self._checkpoints[parsed.run_id]
        self._checkpoints[parsed.run_id] = [
            checkpoint if item["toy_path"] == checkpoint["toy_path"] else item for item in checkpoints
        ]

    def _get_session(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        try:
            session = self._sessions[payload["session_id"]]
        except KeyError as exc:
            raise NotFoundError(f"session not found: {payload['session_id']}") from exc
        return types.GetSessionResponse.model_validate(session).model_dump(mode="json")

    def _list_sessions(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        limit = payload.get("limit", 20)
        offset = payload.get("offset", 0)
        session_ids = sorted(self._sessions)
        return types.ListSessionsResponse(
            sessions=session_ids[offset : offset + limit],
            cursor=types.Cursor(limit=limit, offset=offset, total_count=len(session_ids)),
        ).model_dump(mode="json")

    def _get_sampler(self, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        try:
            sampler = self._samplers[payload["sampler_id"]]
        except KeyError as exc:
            raise NotFoundError(f"sampler not found: {payload['sampler_id']}") from exc
        return types.GetSamplerResponse.model_validate(sampler).model_dump(mode="json")


class _FunctionHandle:
    def __init__(self, backend: FakeModalBackend, app_name: str, function_name: str, environment_name: str | None) -> None:
        self.backend = backend
        self.app_name = app_name
        self.function_name = function_name
        self.environment_name = environment_name
        self.spawn = _AsyncCallable(self._spawn, backend._record_async_function_spawn)

    def _spawn(self, payload: dict[str, Any]) -> FakeCall:
        return self.backend.spawn_function(self.app_name, self.function_name, self.environment_name, payload)


class _ClassHandle:
    def __init__(self, backend: FakeModalBackend, app_name: str, class_name: str, environment_name: str | None) -> None:
        self.backend = backend
        self.app_name = app_name
        self.class_name = class_name
        self.environment_name = environment_name

    def __call__(self, **params: str) -> "_WorkerHandle":
        return _WorkerHandle(self.backend, self.app_name, self.class_name, self.environment_name, params)


class _WorkerHandle:
    def __init__(
        self,
        backend: FakeModalBackend,
        app_name: str,
        class_name: str,
        environment_name: str | None,
        params: dict[str, str],
    ) -> None:
        self.backend = backend
        self.app_name = app_name
        self.class_name = class_name
        self.environment_name = environment_name
        self.params = params

    def __getattr__(self, method_name: str):
        worker = self

        class _Method:
            def __init__(self) -> None:
                self.spawn = _AsyncCallable(self._spawn, worker.backend._record_async_method_spawn)

            def _spawn(self, payload: dict[str, Any]) -> FakeCall:
                return worker.backend.spawn_method(
                    worker.app_name,
                    worker.class_name,
                    worker.environment_name,
                    worker.params,
                    method_name,
                    payload,
                )

        return _Method()
