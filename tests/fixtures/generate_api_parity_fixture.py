"""Generate the API parity fixture from checked-in Tinker Markdown docs.

The docs define which classes and methods are part of the compatibility target.
The local implementation supplies the currently asserted parameter names so the
fixture catches accidental SDK surface drift without forcing wrapper internals to
copy every upstream annotation style.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import toy_modal as tinker
from toy_modal import types


DOCS_ROOT = REPO_ROOT / "reference_docs" / "tinker" / "api"
FIXTURE_PATH = Path(__file__).with_name("api_parity_fixture.json")

CLIENT_DOCS = {
    "ServiceClient": "serviceclient.md",
    "TrainingClient": "trainingclient.md",
    "SamplingClient": "samplingclient.md",
    "RestClient": "restclient.md",
}

TYPE_FIELD_NAMES = (
    "AdamParams",
    "Checkpoint",
    "ForwardBackwardOutput",
    "GetServerCapabilitiesResponse",
    "ImageChunk",
    "ImageAssetPointerChunk",
    "TensorData",
)

EXPORTS = (
    "APIFuture",
    "RestClient",
    "SamplingClient",
    "ServiceClient",
    "TrainingClient",
    "TinkerError",
    "types",
)


def build_fixture(docs_root: Path = DOCS_ROOT) -> dict[str, Any]:
    class_map = {
        "ServiceClient": tinker.ServiceClient,
        "TrainingClient": tinker.TrainingClient,
        "SamplingClient": tinker.SamplingClient,
        "RestClient": tinker.RestClient,
    }
    return {
        "exports": list(EXPORTS),
        "classes": {
            class_name: {
                "methods": _documented_method_params(
                    docs_root / doc_name,
                    class_map[class_name],
                )
            }
            for class_name, doc_name in CLIENT_DOCS.items()
        },
        "types": {
            type_name: list(getattr(types, type_name).model_fields)
            for type_name in TYPE_FIELD_NAMES
        },
    }


def dump_fixture(fixture: dict[str, Any]) -> str:
    return _format_json(fixture) + "\n"


def _format_json(value: Any, *, indent: int = 0) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        items = list(value.items())
        for index, (key, item) in enumerate(items):
            suffix = "," if index < len(items) - 1 else ""
            rendered = _format_json(item, indent=indent + 2)
            lines.append(f"{' ' * (indent + 2)}{json.dumps(key)}: {rendered}{suffix}")
        lines.append(f"{' ' * indent}" + "}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return "[]"
        if indent >= 4 and all(not isinstance(item, (dict, list)) for item in value):
            return "[" + ", ".join(json.dumps(item) for item in value) + "]"
        lines = ["["]
        for index, item in enumerate(value):
            suffix = "," if index < len(value) - 1 else ""
            lines.append(f"{' ' * (indent + 2)}{_format_json(item, indent=indent + 2)}{suffix}")
        lines.append(f"{' ' * indent}]")
        return "\n".join(lines)
    return json.dumps(value)


def _documented_methods(path: Path, cls) -> list[str]:
    text = path.read_text(encoding="utf-8")
    names = []
    for match in re.finditer(r"^#### `([^`]+)`", text, flags=re.MULTILINE):
        name = match.group(1)
        if name == "__reduce__":
            continue
        names.append(name)
        async_name = f"{name}_async"
        if not name.endswith("_async") and hasattr(cls, async_name):
            names.append(async_name)
    deduped = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _documented_method_params(path: Path, cls) -> dict[str, list[str]]:
    methods: dict[str, list[str]] = {}
    missing: list[str] = []
    for method_name in _documented_methods(path, cls):
        if hasattr(cls, method_name):
            methods[method_name] = _implementation_params(getattr(cls, method_name))
        else:
            missing.append(method_name)
    if missing:
        raise RuntimeError(f"{cls.__name__} is missing documented methods: {missing}")
    return methods


def _implementation_params(method) -> list[str]:
    return [
        name
        for name in inspect.signature(method).parameters
        if name != "self"
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", action="store_true", help="print the generated fixture")
    parser.add_argument("--check", action="store_true", help="fail if the checked-in fixture is stale")
    args = parser.parse_args(argv)

    rendered = dump_fixture(build_fixture())
    if args.stdout:
        print(rendered, end="")
        return 0
    if args.check:
        current = FIXTURE_PATH.read_text(encoding="utf-8") if FIXTURE_PATH.exists() else ""
        if current != rendered:
            print(f"{FIXTURE_PATH} is stale; run {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    FIXTURE_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
