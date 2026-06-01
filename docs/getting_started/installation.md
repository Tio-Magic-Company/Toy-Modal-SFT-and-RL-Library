# Installation

`toy_modal` requires Python 3.11 or newer according to `pyproject.toml`.

From a checkout, install the SDK in editable mode:

```bash
python -m pip install -e .
```

For no-credential unit tests, install the dev extra if your environment does not
already have `pytest`:

```bash
python -m pip install -e ".[dev]"
```

Modal client support is available through the `modal` extra:

```bash
python -m pip install -e ".[modal]"
```

Backend deployment and development use heavier dependencies:

```bash
python -m pip install -e ".[backend]"
```

Do not deploy the Modal backend as part of installation. Deployment and
Modal-backed examples are user-owned and cost-bearing.

Next: [`quickstart.md`](quickstart.md).
