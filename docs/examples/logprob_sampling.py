"""Sampling example with generated logprobs, prompt logprobs, and top-k prompt scores."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from toy_modal import types
from common import add_service_args, service_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_service_args(parser, project_id="demo-logprobs")
    args = parser.parse_args()

    service = service_from_args(args)
    sampler = service.create_sampling_client(base_model=args.base_model)
    tokenizer = sampler.get_tokenizer()
    prompt = types.ModelInput.from_ints(tokenizer.encode("Explain 2+2:"))
    response = sampler.sample(
        prompt,
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=5, seed=3),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=3,
    ).result()
    sequence = response.sequences[0]
    print(
        json.dumps(
            {
                "text": tokenizer.decode(sequence.tokens),
                "generated_logprobs": sequence.logprobs,
                "prompt_logprobs": response.prompt_logprobs,
                "topk_prompt_logprobs_at_1": response.topk_prompt_logprobs[1]
                if response.topk_prompt_logprobs
                else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
