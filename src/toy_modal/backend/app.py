"""Modal app definitions for the toy_modal backend scaffold."""

import modal

from toy_modal.backend.config import load_config
from toy_modal.defaults import DEFAULT_UNSLOTH_MODEL_FAMILIES

config = load_config()
runtime_env = {
    "TOY_MODAL_APP_NAME": config.app_name,
    "TOY_MODAL_TRAIN_GPU": config.train_gpu,
    "TOY_MODAL_SAMPLE_GPU": config.sample_gpu,
    "TOY_MODAL_TRAINER_ENGINE": config.trainer_engine,
    "TOY_MODAL_SAMPLER_ENGINE": config.sampler_engine,
    "TOY_MODAL_MODEL_VOLUME": config.model_volume,
    "TOY_MODAL_RUN_VOLUME": config.run_volume,
    "TOY_MODAL_REGISTRY_DICT": config.registry_dict,
    "TOY_MODAL_SAMPLE_MAX_CONTAINERS": str(config.sample_max_containers),
    "TOY_MODAL_ALLOW_UNSAFE_CUSTOM_LOSS": "1" if config.allow_unsafe_custom_loss else "0",
    "TOY_MODAL_HTTP_LARGE_RESULT_INLINE_BYTES": str(config.http_large_result_inline_bytes),
}
if config.resolved_prefetch_gpu:
    runtime_env["TOY_MODAL_PREFETCH_GPU"] = config.resolved_prefetch_gpu
if config.supported_models:
    runtime_env["TOY_MODAL_SUPPORTED_MODELS"] = " ".join(config.supported_models)
if config.uses_unsloth:
    runtime_env.update(
        {
            "TOY_MODAL_UNSLOTH_LOAD_IN_4BIT": "1" if config.unsloth_load_in_4bit else "0",
            "TOY_MODAL_UNSLOTH_LOAD_IN_8BIT": "1" if config.unsloth_load_in_8bit else "0",
            "TOY_MODAL_UNSLOTH_LOAD_IN_16BIT": "1" if config.unsloth_load_in_16bit else "0",
            "TOY_MODAL_UNSLOTH_MAX_SEQ_LENGTH": str(config.unsloth_max_seq_length),
            "TOY_MODAL_UNSLOTH_USE_GRADIENT_CHECKPOINTING": config.unsloth_use_gradient_checkpointing,
            "TOY_MODAL_UNSLOTH_TRUST_REMOTE_CODE": "1" if config.unsloth_trust_remote_code else "0",
            "TOY_MODAL_UNSLOTH_FAST_INFERENCE": "1" if config.unsloth_fast_inference else "0",
            "TOY_MODAL_UNSLOTH_GPU_MEMORY_UTILIZATION": str(config.unsloth_gpu_memory_utilization),
            "TOY_MODAL_UNSLOTH_USE_EXACT_MODEL_NAME": "1" if config.unsloth_use_exact_model_name else "0",
            "TOY_MODAL_UNSLOTH_PACKAGE": config.unsloth_package,
            "TOY_MODAL_UNSLOTH_BITSANDBYTES_PACKAGE": config.unsloth_bitsandbytes_package,
            "TOY_MODAL_UNSLOTH_EXTRA_PIP_PACKAGES": " ".join(config.unsloth_extra_pip_packages),
        }
    )
    if config.unsloth_load_in_fp8:
        runtime_env["TOY_MODAL_UNSLOTH_LOAD_IN_FP8"] = config.unsloth_load_in_fp8
    if config.unsloth_dtype:
        runtime_env["TOY_MODAL_UNSLOTH_DTYPE"] = config.unsloth_dtype
if config.hf_secret_name:
    runtime_env["TOY_MODAL_HF_SECRET_NAME"] = config.hf_secret_name
if config.http_api_key:
    runtime_env["TOY_MODAL_HTTP_API_KEY"] = config.http_api_key

app = modal.App(config.app_name)

model_cache_volume = modal.Volume.from_name(config.model_volume, create_if_missing=True)
run_volume = modal.Volume.from_name(config.run_volume, create_if_missing=True)
registry = modal.Dict.from_name(config.registry_dict, create_if_missing=True)
secrets = [modal.Secret.from_name(config.hf_secret_name)] if config.hf_secret_name else []

image_packages = [
    "pydantic>=2.8,<3",
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "safetensors",
    "fastapi[standard]",
]
if config.uses_unsloth:
    image_packages.extend(config.unsloth_pip_packages)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*image_packages)
    .add_local_python_source("toy_modal")
)

prefetch_function_options = {
    "image": image,
    "volumes": {"/models": model_cache_volume},
    "env": runtime_env,
    "secrets": secrets,
    "timeout": 60 * 60,
}
if config.resolved_prefetch_gpu:
    prefetch_function_options["gpu"] = config.resolved_prefetch_gpu


@app.function(image=image, env=runtime_env)
def get_server_capabilities(payload: dict) -> dict:
    backend_profile = {
        "uses_unsloth": config.uses_unsloth,
        "prefetch_gpu": config.resolved_prefetch_gpu,
    }
    if config.uses_unsloth:
        from toy_modal.backend.unsloth_config import unsloth_runtime_versions

        backend_profile["unsloth"] = {
            "engine_config": config.unsloth_engine_config().manifest(),
            "pip_packages": list(config.unsloth_pip_packages),
            "package_versions": unsloth_runtime_versions(),
            "model_families": list(DEFAULT_UNSLOTH_MODEL_FAMILIES),
        }
    return {
        "supported_models": payload.get("configured_models") or list(config.resolved_supported_models),
        "supports_lora": True,
        "supports_full_finetune": False,
        "supports_sampling": True,
        "supports_importance_sampling": True,
        "max_batch_size": None,
        "transport": "modal-direct",
        "trainer_engine": config.trainer_engine,
        "sampler_engine": config.sampler_engine,
        "backend_profile": backend_profile,
    }


@app.function(image=image, volumes={"/runs": run_volume}, env=runtime_env)
def create_lora_training_run(payload: dict) -> dict:
    from toy_modal.backend.registry import create_run

    run_volume.reload()
    result = create_run(payload, registry=registry, run_root="/runs")
    run_volume.commit()
    return result


@app.function(image=image, volumes={"/runs": run_volume}, env=runtime_env)
def create_training_run_from_state(payload: dict) -> dict:
    from toy_modal.backend.metadata import create_training_run_from_state as create_from_state

    run_volume.reload()
    result = create_from_state(payload, registry=registry, run_root="/runs")
    run_volume.commit()
    return result


@app.function(image=image, volumes={"/runs": run_volume}, env=runtime_env)
def metadata_route(payload: dict) -> dict:
    from toy_modal.backend.metadata import handle_metadata_route

    run_volume.reload()
    result = handle_metadata_route(
        payload["route"],
        payload.get("payload") or {},
        registry=registry,
        run_root="/runs",
        archive_url_prefix=f"modal-volume://{config.run_volume}",
    )
    if payload["route"] in {
        "rest.delete_checkpoint",
        "rest.delete_checkpoint_from_toy_path",
        "rest.set_checkpoint_ttl_from_toy_path",
        "rest.set_checkpoint_public",
        "rest.get_checkpoint_archive_url",
        "rest.get_checkpoint_archive_url_from_toy_path",
    }:
        run_volume.commit()
    return result


@app.function(image=image, volumes={"/models": model_cache_volume, "/runs": run_volume}, env=runtime_env)
def tokenizer_route(payload: dict) -> dict:
    from toy_modal.backend.sampler_worker import tokenizer_for_reference

    run_volume.reload()
    tokenizer = tokenizer_for_reference(
        base_model=payload.get("base_model"),
        model_path=payload.get("model_path"),
        model_root="/models",
        run_root="/runs",
    )
    if payload["operation"] == "encode":
        return {"tokens": tokenizer.encode(payload["text"], add_special_tokens=False)}
    if payload["operation"] == "decode":
        return {"text": tokenizer.decode(payload["tokens"], skip_special_tokens=False)}
    raise ValueError(f"unsupported tokenizer operation: {payload['operation']!r}")


@app.function(**prefetch_function_options)
def prefetch_model(payload: dict) -> dict:
    from toy_modal.backend.model_cache import prefetch_model as run_prefetch

    model_cache_volume.reload()
    result = run_prefetch(
        payload["model_id"],
        model_root="/models",
        include_model=payload.get("include_model", True),
        include_tokenizer=payload.get("include_tokenizer", True),
        dry_run=payload.get("dry_run", False),
        local_files_only=payload.get("local_files_only", False),
        backend=payload.get("backend") or ("unsloth" if config.uses_unsloth else "transformers"),
    )
    if not payload.get("dry_run", False):
        model_cache_volume.commit()
    return result


@app.cls(
    image=image,
    gpu=config.train_gpu,
    volumes={"/models": model_cache_volume, "/runs": run_volume},
    env=runtime_env,
    secrets=secrets,
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    timeout=60 * 60,
)
class TrainerWorker:
    run_id: str = modal.parameter()

    @modal.enter()
    def setup(self):
        from toy_modal.backend.trainer_worker import load_trainer_engine

        run_volume.reload()
        self.engine = load_trainer_engine(
            config.trainer_engine,
            run_id=self.run_id,
            registry=registry,
            model_root="/models",
            run_root="/runs",
        )

    @modal.method()
    def forward(self, payload: dict) -> dict:
        self._wait_for_dependency(payload)
        run_volume.reload()
        result = self.engine.forward(payload)
        run_volume.commit()
        return result

    @modal.method()
    def forward_backward(self, payload: dict) -> dict:
        self._wait_for_dependency(payload)
        run_volume.reload()
        result = self.engine.forward_backward(payload)
        run_volume.commit()
        return result

    @modal.method()
    def optim_step(self, payload: dict) -> dict:
        self._wait_for_dependency(payload)
        run_volume.reload()
        result = self.engine.optim_step(payload)
        run_volume.commit()
        return result

    @modal.method()
    def save_state(self, payload: dict) -> dict:
        self._wait_for_dependency(payload)
        run_volume.reload()
        result = self.engine.save_state(payload)
        run_volume.commit()
        return result

    @modal.method()
    def load_state(self, payload: dict) -> dict:
        self._wait_for_dependency(payload)
        run_volume.reload()
        result = self.engine.load_state(payload)
        run_volume.commit()
        return result

    @modal.method()
    def save_weights_for_sampler(self, payload: dict) -> dict:
        self._wait_for_dependency(payload)
        run_volume.reload()
        result = self.engine.save_weights_for_sampler(payload)
        run_volume.commit()
        return result

    @modal.method()
    def validate_old_logprobs_sequence(self, payload: dict) -> dict:
        self._wait_for_dependency(payload)
        run_volume.reload()
        return self.engine.validate_old_logprobs_sequence(payload)

    def _wait_for_dependency(self, payload: dict) -> None:
        dependency = payload.get("depends_on")
        if dependency:
            modal.FunctionCall.from_id(dependency).get()


@app.cls(
    image=image,
    gpu=config.sample_gpu,
    volumes={"/models": model_cache_volume, "/runs": run_volume},
    env=runtime_env,
    secrets=secrets,
    min_containers=0,
    max_containers=config.sample_max_containers,
    scaledown_window=300,
    timeout=30 * 60,
)
class SamplerWorker:
    model_path: str = modal.parameter(default="")
    base_model: str = modal.parameter(default="")

    @modal.enter()
    def setup(self):
        from toy_modal.backend.sampler_worker import load_sampler_engine

        run_volume.reload()
        self.engine = load_sampler_engine(
            config.sampler_engine,
            base_model=self.base_model or None,
            model_path=self.model_path or None,
            model_root="/models",
            run_root="/runs",
        )

    @modal.method()
    def sample(self, payload: dict) -> dict:
        return self.engine.sample(payload)

    @modal.method()
    def compute_logprobs(self, payload: dict) -> list[float | None]:
        return self.engine.compute_logprobs(payload)


@app.function(image=image, env=runtime_env)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def http_gateway():
    from toy_modal.backend.http_gateway import create_app
    from toy_modal.transport.modal_direct import ModalDirectTransport

    return create_app(
        transport=ModalDirectTransport(app_name=config.app_name),
        large_result_inline_bytes=config.http_large_result_inline_bytes,
    )
