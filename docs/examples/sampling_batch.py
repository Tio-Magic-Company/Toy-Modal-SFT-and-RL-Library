"""Minimal sampling call against a deployed backend."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from toy_modal import types
from common import add_service_args, service_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_service_args(parser, project_id="demo")
    args = parser.parse_args()

    service_client = service_from_args(args)
    sampling_client = service_client.create_sampling_client(base_model=args.base_model)
    tokenizer = sampling_client.get_tokenizer()
    response = sampling_client.sample(
        prompt=types.ModelInput.from_ints(tokenizer.encode("Hello")),
        num_samples=2,
        sampling_params=types.SamplingParams(max_tokens=3),
    ).result()
    for sequence in response.sequences:
        print(tokenizer.decode(sequence.tokens))


if __name__ == "__main__":
    main()
