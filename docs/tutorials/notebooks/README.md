# toy_modal Tutorials

These tutorials are Marimo notebooks stored as `.py` files. They are designed
to look and feel like the original notebook tutorial series while using
clean-room Toy Modal APIs and a user-owned Modal backend.

Install the notebook dependencies:

```bash
python -m pip install -e '.[backend,tutorials]'
```

Start with setup:

```bash
marimo edit docs/tutorials/notebooks/000_setup_tutorial.py
```

Then open the first SDK tutorial:

```bash
marimo edit docs/tutorials/notebooks/101_hello_toy_modal.py
```

## Notebook Model

Each file defines:

- `import marimo`
- `app = marimo.App()`
- markdown teaching cells with `mo.md(...)`
- executable code cells
- Modal app/base-model UI controls
- explicit cost acknowledgement and remote-run checkboxes
- a visible limitations section

The files are not CLI scripts anymore. Do not run them as
`python docs/tutorials/notebooks/102_first_sft.py --i-understand-costs`; open them with
Marimo and use the notebook UI.

## Learning Path

| Files | Topic |
| --- | --- |
| `000_setup_tutorial.py` | Deploy and smoke-check the Modal backend. |
| `101` to `104` | Client hierarchy, sampling, first SFT loop, async futures, first RL loop. |
| `201` to `205` | Renderers, loss functions, completers, weights, and evaluations. |
| `301` to `304` | Cookbook environment abstractions, custom environments, SFT config, RL config. |
| `401` to `407` | Hyperparameters, KL controls, DPO, sequence extension, multi-agent scoring, distillation, RLHF composition. |
| `501` to `503` | Download/export, LoRA adapter construction, model cards, guarded Hub publishing. |

## Cost Guard

The notebooks teach the Modal-backed path and use `transport="modal-direct"`.
Remote cells are gated by checkboxes inside the notebook. Publishing to
Hugging Face is also guarded by explicit UI controls and credential checks.

The tutorial defaults now match the original model targets where practical.
Preflight large or gated models before running remote cells:

```bash
toy-modal backend prefetch-model Qwen/Qwen3.5-4B --dry-run
```

For gated models such as `meta-llama/Llama-3.1-8B`, create a Modal Secret first:

```bash
modal secret create huggingface-token HF_TOKEN="$HF_TOKEN"
export TOY_MODAL_HF_SECRET_NAME=huggingface-token
```

Normal unit tests statically inspect notebook structure and prose. They do not
import Marimo or run Modal jobs unless the corresponding opt-in environment
variables are set.

## Documentation Site Direction

The notebooks are the source of truth. Narrative documentation lives under
`docs/tutorials/` so it can later be included in a MkDocs Material
site and published to GitHub Pages.
