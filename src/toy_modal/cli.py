"""CLI for Modal smoke tests, backend guidance, and metadata operations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from toy_modal import ServiceClient, types
from toy_modal.cookbook import (
    DEFAULT_MODAL_BASE_MODEL,
    RECIPE_NAMES,
    RecipeConfig,
    run_many_smoke_recipes,
    run_smoke_recipe,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="toy-modal")
    parser.add_argument("--project-id", default="cli")
    parser.add_argument("--transport", default="modal-direct")
    parser.add_argument("--base-url")
    parser.add_argument("--app-name", default="toy-modal-backend")
    parser.add_argument("--environment-name")
    parser.add_argument("--api-key")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backend = subparsers.add_parser("backend")
    backend_subparsers = backend.add_subparsers(dest="backend_command", required=True)
    deploy = backend_subparsers.add_parser("deploy")
    deploy.add_argument("--module", default="toy_modal.backend.app")
    check = backend_subparsers.add_parser("check")
    check.add_argument("--base-model", default=DEFAULT_MODAL_BASE_MODEL)
    check.add_argument("--app-name", default=argparse.SUPPRESS)
    check.add_argument("--environment-name", default=argparse.SUPPRESS)
    check.add_argument("--api-key", default=argparse.SUPPRESS)
    check.add_argument("--prefetch-tokenizer", action="store_true")
    prefetch = backend_subparsers.add_parser("prefetch-model")
    prefetch.add_argument("model_id")
    prefetch.add_argument("--app-name", default=None)
    prefetch.add_argument("--environment-name", default=None)
    prefetch.add_argument("--model-root", default=".toy_modal_model_cache")
    prefetch.add_argument("--dry-run", action="store_true")
    prefetch.add_argument("--tokenizer-only", action="store_true")
    prefetch.add_argument("--local-files-only", action="store_true")
    prefetch.add_argument("--backend", choices=["auto", "unsloth", "transformers"], default="auto")

    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--base-model", default=DEFAULT_MODAL_BASE_MODEL)
    smoke.add_argument("--app-name", default=argparse.SUPPRESS)
    smoke.add_argument("--environment-name", default=argparse.SUPPRESS)
    smoke.add_argument("--api-key", default=argparse.SUPPRESS)

    run = subparsers.add_parser("run")
    run_subparsers = run.add_subparsers(dest="run_command", required=True)
    run_subparsers.add_parser("list")
    run_info = run_subparsers.add_parser("info")
    run_info.add_argument("run_id")
    run_stop = run_subparsers.add_parser("stop")
    run_stop.add_argument("run_id")

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint_subparsers = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_list = checkpoint_subparsers.add_parser("list")
    checkpoint_list.add_argument("run_id")
    checkpoint_download = checkpoint_subparsers.add_parser("download")
    checkpoint_download.add_argument("path")
    checkpoint_download.add_argument("destination")
    checkpoint_delete = checkpoint_subparsers.add_parser("delete")
    checkpoint_delete.add_argument("path")

    cookbook = subparsers.add_parser("cookbook")
    cookbook_subparsers = cookbook.add_subparsers(dest="cookbook_command", required=True)
    cookbook_subparsers.add_parser("list")
    cookbook_smoke = cookbook_subparsers.add_parser("smoke")
    cookbook_smoke.add_argument("recipe", nargs="?")
    cookbook_smoke.add_argument("--all", action="store_true")
    cookbook_smoke.add_argument("--base-model", default=DEFAULT_MODAL_BASE_MODEL)
    cookbook_smoke.add_argument("--app-name", default=argparse.SUPPRESS)
    cookbook_smoke.add_argument("--environment-name", default=argparse.SUPPRESS)
    cookbook_smoke.add_argument("--api-key", default=argparse.SUPPRESS)
    cookbook_smoke.add_argument("--log-path")

    args = parser.parse_args(argv)

    if args.command == "backend" and args.backend_command == "deploy":
        return subprocess.call([sys.executable, "-m", "modal", "deploy", "-m", args.module])

    if args.command == "backend" and args.backend_command == "prefetch-model":
        return _prefetch_model_command(args)

    if args.command == "backend" and args.backend_command == "check":
        return _backend_check_command(args)

    client = _client(args)

    if args.command == "smoke-test":
        return _smoke_test(client, args.base_model)

    if args.command == "run":
        return _run_command(client, args)

    if args.command == "checkpoint":
        return _checkpoint_command(client, args)

    if args.command == "cookbook":
        return _cookbook_command(args)

    parser.error("unhandled command")
    return 2


def _client(args) -> ServiceClient:
    kwargs = {
        "project_id": args.project_id,
        "transport": args.transport,
        "app_name": args.app_name,
        "environment_name": args.environment_name,
    }
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.api_key:
        kwargs["api_key"] = args.api_key
    return ServiceClient(**kwargs)


def _smoke_test(client: ServiceClient, base_model: str) -> int:
    training = client.create_lora_training_client(base_model=base_model, rank=4)
    tokenizer = training.get_tokenizer()
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokenizer.encode("Question: 2+2? Answer: 4")),
        loss_fn_inputs={"target_tokens": [4], "weights": [1]},
    )
    loss = training.forward_backward([datum], "cross_entropy")
    step = training.optim_step(types.AdamParams(learning_rate=1e-4))
    print(f"loss={loss.result().loss}")
    print(f"step={step.result().optimizer_step}")
    return 0


def _prefetch_model_command(args) -> int:
    payload = {
        "model_id": args.model_id,
        "include_model": not args.tokenizer_only,
        "include_tokenizer": True,
        "dry_run": bool(args.dry_run),
        "local_files_only": bool(args.local_files_only),
        "backend": args.backend,
    }
    if args.dry_run:
        from toy_modal.backend.model_cache import prefetch_model

        result = prefetch_model(
            args.model_id,
            model_root=args.model_root,
            include_model=payload["include_model"],
            include_tokenizer=True,
            dry_run=True,
            local_files_only=args.local_files_only,
            backend=args.backend,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    try:
        import modal
    except ImportError as exc:
        raise RuntimeError("real model prefetch requires the modal package") from exc

    app_name = args.app_name or "toy-modal-backend"
    function = modal.Function.from_name(
        app_name,
        "prefetch_model",
        environment_name=args.environment_name,
    )
    result = function.spawn(payload).get()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _backend_check_command(args) -> int:
    client = _client(args)
    capabilities = client.get_server_capabilities()
    training = client.create_lora_training_client(
        base_model=args.base_model,
        rank=4,
        train_unembed=False,
        user_metadata={"probe": "backend-check"},
    )
    result = {
        "app_name": args.app_name,
        "environment_name": args.environment_name,
        "transport": args.transport,
        "project_id": args.project_id,
        "base_model": args.base_model,
        "capabilities": capabilities.model_dump(mode="json"),
        "training_run_id": training.training_run_id,
        "model_seq_id": training.model_seq_id,
        "optimizer_step": training.optimizer_step,
    }
    if args.prefetch_tokenizer:
        try:
            import modal
        except ImportError as exc:
            raise RuntimeError("tokenizer prefetch requires the modal package") from exc
        function = modal.Function.from_name(
            args.app_name,
            "prefetch_model",
            environment_name=args.environment_name,
        )
        result["prefetch"] = function.spawn(
            {
                "model_id": args.base_model,
                "include_model": False,
                "include_tokenizer": True,
                "dry_run": False,
                "local_files_only": False,
            }
        ).get()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_command(client: ServiceClient, args) -> int:
    rest = client.create_rest_client()
    if args.run_command == "list":
        response = rest.list_training_runs().result()
        print(json.dumps(response.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    if args.run_command == "info":
        response = rest.get_training_run(args.run_id).result()
        print(json.dumps(response.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    if args.run_command == "stop":
        print(f"stop requested for {args.run_id}; backend cancellation is transport-specific")
        return 0
    raise ValueError(f"unknown run command: {args.run_command}")


def _checkpoint_command(client: ServiceClient, args) -> int:
    rest = client.create_rest_client()
    if args.checkpoint_command == "list":
        response = rest.list_checkpoints(args.run_id).result()
        print(json.dumps(response.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    if args.checkpoint_command == "download":
        archive = rest.get_checkpoint_archive_url_from_toy_path(args.path).result()
        destination = Path(args.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if archive.url.startswith(("http://", "https://")):
            with urllib.request.urlopen(archive.url) as response:
                destination.write_bytes(response.read())
        else:
            destination.write_text(json.dumps(archive.model_dump(mode="json"), indent=2, sort_keys=True))
            print(
                "checkpoint archive is Volume-backed metadata; use `modal volume get` "
                "with the recorded modal-volume:// path to download the tarball.",
                file=sys.stderr,
            )
        print(str(destination))
        return 0
    if args.checkpoint_command == "delete":
        rest.delete_checkpoint_from_toy_path(args.path).result()
        print(f"deleted {args.path}")
        return 0
    raise ValueError(f"unknown checkpoint command: {args.checkpoint_command}")


def _cookbook_command(args) -> int:
    if args.cookbook_command == "list":
        for name in RECIPE_NAMES:
            print(name)
        return 0
    if args.cookbook_command == "smoke":
        if args.all:
            results = run_many_smoke_recipes(
                transport=args.transport,
                base_model=args.base_model,
                project_id=args.project_id,
                app_name=args.app_name,
                environment_name=args.environment_name,
                base_url=args.base_url,
                api_key=args.api_key,
                log_path=args.log_path,
            )
        else:
            name = args.recipe or "sl_loop"
            results = [
                run_smoke_recipe(
                    RecipeConfig(
                        name=name,
                        transport=args.transport,
                        base_model=args.base_model,
                        project_id=args.project_id,
                        app_name=args.app_name,
                        environment_name=args.environment_name,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        log_path=args.log_path,
                    )
                )
            ]
        for result in results:
            print(json.dumps(result.to_record(), sort_keys=True))
        return 0
    raise ValueError(f"unknown cookbook command: {args.cookbook_command}")


if __name__ == "__main__":
    sys.exit(main())
