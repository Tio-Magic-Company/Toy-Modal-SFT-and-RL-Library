# Unsloth Backend

Toy Modal can use Unsloth Core as an accelerated backend while keeping the same
Tinker-style SDK surface. The selected engines are:

```text
TOY_MODAL_TRAINER_ENGINE=unsloth-peft
TOY_MODAL_SAMPLER_ENGINE=unsloth
```

`UnslothTrainerEngine` subclasses the existing PEFT trainer contract. It still
accepts `types.LoraConfig`, writes `toy-modal://` checkpoints through Modal
Volumes, returns APIFuture-compatible worker results, and keeps custom Python
loss callables disabled by default. The difference is that base-model loading,
quantized modes, LoRA wrapping, and training/inference patching go through
`unsloth.FastLanguageModel`.

`UnslothSamplerEngine` preserves `SamplingClient.sample`, `compute_logprobs`,
prompt logprobs, top-k prompt logprobs, and saved-adapter sampling. It can load
the same sampler-weight manifests written by `save_weights_for_sampler`.

Public CLI, cookbook, recipe, and example defaults use
`toy_modal.DEFAULT_BASE_MODEL`, currently `unsloth/tinyllama-bnb-4bit`, so the
default model follows the default backend. Use
`toy_modal.DEFAULT_TRANSFORMERS_BASE_MODEL` or an explicit `--base-model` when
reproducing the plain PEFT/Transformers baseline.

## Configuration

The default Unsloth settings favor a practical LoRA deployment shape:

```text
TOY_MODAL_UNSLOTH_LOAD_IN_4BIT=1
TOY_MODAL_UNSLOTH_LOAD_IN_8BIT=0
TOY_MODAL_UNSLOTH_LOAD_IN_16BIT=0
TOY_MODAL_UNSLOTH_MAX_SEQ_LENGTH=2048
TOY_MODAL_UNSLOTH_USE_GRADIENT_CHECKPOINTING=unsloth
TOY_MODAL_UNSLOTH_TRUST_REMOTE_CODE=0
TOY_MODAL_UNSLOTH_FAST_INFERENCE=0
TOY_MODAL_UNSLOTH_GPU_MEMORY_UTILIZATION=0.5
TOY_MODAL_UNSLOTH_USE_EXACT_MODEL_NAME=0
TOY_MODAL_UNSLOTH_PACKAGE=unsloth[base]
TOY_MODAL_UNSLOTH_BITSANDBYTES_PACKAGE=bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0
TOY_MODAL_UNSLOTH_EXTRA_PIP_PACKAGES=
```

Only one quantized mode may be enabled at a time. `TOY_MODAL_UNSLOTH_LOAD_IN_FP8`
may be set to `1`, `true`, or an Unsloth-supported string such as `block`.
`TOY_MODAL_UNSLOTH_DTYPE` may be set to a PyTorch dtype name such as `bfloat16`.

The package profile is intentionally configurable because Unsloth's supported
Torch, CUDA, Triton, and bitsandbytes combinations move faster than this SDK.
The default Modal image installs `unsloth[base]` plus a Linux bitsandbytes range
that matches the current Unsloth package metadata. Override
`TOY_MODAL_UNSLOTH_PACKAGE`, clear or pin
`TOY_MODAL_UNSLOTH_BITSANDBYTES_PACKAGE`, and add whitespace-separated
`TOY_MODAL_UNSLOTH_EXTRA_PIP_PACKAGES` when validating a specific GPU image.

## Fallbacks

Use the plain PEFT/Transformers baseline when debugging dependency issues or
comparing behavior:

```text
TOY_MODAL_TRAINER_ENGINE=peft
TOY_MODAL_SAMPLER_ENGINE=transformers
```

Use the tiny engines for deterministic route tests that avoid model loading:

```text
TOY_MODAL_TRAINER_ENGINE=tiny
TOY_MODAL_SAMPLER_ENGINE=tiny
```

## Model Prefetch

`toy-modal backend prefetch-model --backend auto` follows the configured backend
engines. With the current defaults, a model prefetch uses Unsloth Core and a
tokenizer-only prefetch uses Transformers `AutoTokenizer`:

```bash
toy-modal backend prefetch-model Qwen/Qwen3.5-4B --dry-run --backend unsloth
toy-modal backend prefetch-model Qwen/Qwen3.5-4B --backend unsloth --app-name toy-modal-backend
```

Use `--backend transformers` when validating the plain baseline cache path.
Set `TOY_MODAL_PREFETCH_BACKEND=transformers` to make `--backend auto` choose
the plain cache path even when an Unsloth engine is configured.
The deployed prefetch function uses `TOY_MODAL_PREFETCH_GPU` when set. Otherwise
Unsloth deployments use the sampling GPU for full model prefetches because
`FastLanguageModel.from_pretrained` performs real model loading.

## Operational Evidence

`ServiceClient.get_server_capabilities()` includes the deployed
`trainer_engine`, `sampler_engine`, and a `backend_profile`. For Unsloth
deployments, that profile records the Unsloth engine settings and Modal image
package profile, plus runtime versions for `unsloth`, `unsloth_zoo`, `torch`,
`transformers`, `peft`, `trl`, and `bitsandbytes`. Checkpoint manifests saved
by `UnslothTrainerEngine` record the same version snapshot alongside
`backend="unsloth-peft"` and the Unsloth load config. This is intentionally
separate from training-run metadata: it lets validation reports prove which
backend the live app is configured to use before any cost-bearing training or
sampling step starts.

The same capabilities response advertises an advisory model list. By default,
Unsloth deployments include `unsloth/tinyllama-bnb-4bit` and representative
Llama, Qwen, and Gemma 4-bit model IDs. This is not an allow-list: users can
still pass any compatible Hugging Face model ID. Set `TOY_MODAL_SUPPORTED_MODELS`
to a shell-split or comma-separated list when an app should advertise a narrower
local catalog. The Modal app propagates this value into remote function env so
the runtime capabilities response matches the deployment configuration.

## Export

`toy_modal.weights.build_hf_model(..., backend="auto")` follows the default
Unsloth profile. It loads the base model through `FastLanguageModel`, attaches
the adapter with PEFT, and uses `save_pretrained_merged` when the patched model
exposes it. The default Unsloth save method is `lora` for adapter export and
`merged_16bit` when `merge=True`.

Use `backend="transformers"` to reproduce the historical
`AutoModelForCausalLM` plus PEFT export path. `dry_run=True` writes a manifest
with the resolved backend and dependency requirements without importing either
heavy backend stack.

## Validation Status

The current checked-in deployed evidence is still the 2026-05-27 tiny-model
PEFT/Transformers `modal-direct` baseline. The Unsloth backend has local unit
coverage for configuration, engine selection, LoRA mapping, runtime wiring,
checkpoint manifests, saved-adapter sampler loading, prefetch planning, export
planning, and Unsloth-first dependency import ordering. Real Unsloth import,
large-model throughput, cookbook parity, HTTP gateway validation, and
production archive behavior still require explicit cost-bearing Modal
validation.

The no-credential validation wrapper includes an Unsloth dependency probe when
an Unsloth engine is selected:

```bash
python dev_notes/validation/run_modal_validation.py
python dev_notes/validation/run_modal_validation.py --require-unsloth-import
```

Without `--require-unsloth-import`, missing local Unsloth packages are reported
but do not fail the no-deploy validation run. With the flag, the wrapper fails
before pytest or Modal phases if `unsloth.FastLanguageModel` cannot import.
Deployed validation scripts also assert that Unsloth capabilities include an
engine profile, model families, and the selected base model in
`supported_models`. Use `--supported-models` to advertise custom models through
the wrapper, or `--skip-supported-model-check` for intentionally advisory-only
catalogs.
