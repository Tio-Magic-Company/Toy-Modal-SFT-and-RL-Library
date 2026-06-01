"""Checkpoint-shaped workflow against a deployed backend."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import add_service_args, service_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_service_args(parser, project_id="demo")
    args = parser.parse_args()

    service_client = service_from_args(args, accept_tinker_paths=True)
    training_client = service_client.create_lora_training_client(args.base_model)
    checkpoint = training_client.save_state("initial").result()
    resumed = service_client.create_training_client_from_state_with_optimizer(checkpoint.path)
    print(resumed.get_info().training_run_id)


if __name__ == "__main__":
    main()
