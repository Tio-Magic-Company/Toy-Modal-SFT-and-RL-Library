# Tutorial Series

The tutorial series is the Modal-first path through `toy_modal`. Each tutorial
is a Marimo notebook stored as a `.py` file in [`notebooks/`](notebooks/). The notebooks
contain the teaching prose, UI controls, code cells, summaries, and limitations;
these documentation pages organize the sequence and future docs-site shape.

Install the notebook dependencies:

```bash
python -m pip install -e '.[backend,tutorials]'
```

Start with backend setup:

```bash
marimo edit docs/tutorials/notebooks/000_setup_tutorial.py
```

Then open the first SDK notebook:

```bash
marimo edit docs/tutorials/notebooks/101_hello_toy_modal.py
```

## Recommended Order

| Stage | Tutorials | What you learn |
| --- | --- | --- |
| Setup | `000` | Modal app ownership, backend env vars, deployment, smoke checks. |
| SDK basics | `101` to `104` | Service, sampling, training, futures, and the first RL loop. |
| Core utilities | `201` to `205` | Rendering, losses, completers, checkpoints, and evaluators. |
| Cookbook abstractions | `301` to `304` | Problem environments, rollouts, custom envs, SFT config, RL config. |
| Advanced workflows | `401` to `407` | Sweeps, KL, DPO, sequence extension, multi-agent scoring, distillation, RLHF composition. |
| Export | `501` to `503` | Artifact download, LoRA export, model cards, and guarded Hub publishing. |

## Why Marimo

The original tutorial set is a collection of Marimo notebooks stored as Python
files. Toy Modal follows that model so readers get notebook-style explanatory
sections, interactive controls, and reactive execution while keeping the files
reviewable in Git.

The notebooks still teach normal Python control loops. They are just presented
cell-by-cell so setup, local data construction, remote Modal calls, inspection,
and limitations are easier to read.

## Future Docs Site

The current Markdown structure is ready for a MkDocs Material site. A later docs
publishing change can add `mkdocs.yml`, point the navigation at
`docs/`, include this tutorial index, and publish the generated
site to GitHub Pages. That should be a documentation deployment step, not a
dependency of running the notebooks.
