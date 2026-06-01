"""RL-shaped scaffold against a deployed backend."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from toy_modal import types
from toy_modal.backend.loss_inputs import validate_training_batch
from common import add_service_args, service_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_service_args(parser, project_id="demo")
    args = parser.parse_args()

    service_client = service_from_args(args)
    training_client = service_client.create_lora_training_client(args.base_model)
    tokenizer = training_client.get_tokenizer()
    prompt = types.ModelInput.from_ints(tokenizer.encode("State: easy math\nAction:"))

    sampler = training_client.save_weights_and_get_sampling_client("policy-0")
    rollout = sampler.sample(prompt, 1, types.SamplingParams(max_tokens=4)).result()
    sequence = rollout.sequences[0]
    completion = sequence.tokens[prompt.length() :]
    old_logprobs = sequence.logprobs or [0.0] * len(completion)

    datum = types.Datum(
        model_input=types.ModelInput.from_ints(prompt.to_ints() + completion),
        loss_fn_inputs={
            "target_tokens": completion,
            "weights": [1.0] * len(completion),
            "old_logprobs": old_logprobs,
            "advantages": [1.0] * len(completion),
        },
    )
    validate_training_batch([datum], "importance_sampling")
    result = training_client.forward_backward([datum], "importance_sampling").result()
    print(result.loss)


if __name__ == "__main__":
    main()
