# Project Layout

Important directories:

| Path | Purpose |
| --- | --- |
| `src/toy_modal/types.py` | Public Pydantic models and request/response types. |
| `src/toy_modal/futures.py` | Shared `APIFuture` abstraction. |
| `src/toy_modal/clients/` | Public SDK clients. |
| `src/toy_modal/transport/` | `modal-direct` and deployed HTTP transport code. |
| `src/toy_modal/backend/` | Modal app, worker, metadata, storage, and loss helpers. |
| `docs/examples/` | Small Modal-backed examples that mirror the SDK shape. |
| `docs/recipes/` | Clean-room cookbook recipes and workflow scripts. |
| `tests/unit/` | Fast tests using fakes; they do not require Modal credentials or GPUs. |
| `docs/technical/` | User-facing technical notes and compatibility appendices. |
| `reference_docs/` | Checked-in Modal and Tinker reference snapshots. |
| `dev_notes/` | Private state-of-repo report, validation scripts, and validation reports. |

The package name is `toy_modal`. Use `import toy_modal as tinker` only as an
import alias in user code.
