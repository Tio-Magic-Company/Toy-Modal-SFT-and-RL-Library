"""Shared argument helpers for deployed examples."""

from __future__ import annotations

import argparse
import os

import toy_modal as tinker

DEFAULT_BASE_MODEL = tinker.DEFAULT_BASE_MODEL


def add_service_args(parser: argparse.ArgumentParser, *, project_id: str) -> None:
    parser.add_argument("--project-id", default=project_id)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--transport", default="modal-direct")
    parser.add_argument("--app-name", default=os.getenv("TOY_MODAL_APP_NAME", "toy-modal-backend"))
    parser.add_argument("--environment-name", default=os.getenv("TOY_MODAL_ENVIRONMENT"))
    parser.add_argument("--base-url", default=os.getenv("TOY_MODAL_HTTP_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("TOY_MODAL_HTTP_API_KEY"))


def service_from_args(args: argparse.Namespace, **kwargs) -> tinker.ServiceClient:
    service_kwargs = {
        "project_id": args.project_id,
        "transport": args.transport,
        "app_name": args.app_name,
        "environment_name": args.environment_name,
        **kwargs,
    }
    if args.transport == "http":
        service_kwargs["base_url"] = args.base_url
        service_kwargs["api_key"] = args.api_key
    return tinker.ServiceClient(**service_kwargs)
