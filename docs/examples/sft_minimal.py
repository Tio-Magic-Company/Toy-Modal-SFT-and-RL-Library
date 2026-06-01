"""Minimal SFT-shaped loop against a deployed backend."""

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
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=8,
        train_unembed=False,
    )

    tokenizer = training_client.get_tokenizer()
    prompt = tokenizer.encode("Question: 2+2? Answer:")
    answer = tokenizer.encode(" 4")
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(prompt + answer),
        loss_fn_inputs={
            "target_tokens": answer,
            "weights": [0] * len(prompt) + [1] * len(answer),
        },
    )
    validate_training_batch([datum], "cross_entropy")

    fwdbwd_future = training_client.forward_backward([datum], "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

    print(f"loss: {fwdbwd_future.result().loss}")
    print(f"step: {optim_future.result().optimizer_step}")

    sampling_client = training_client.save_weights_and_get_sampling_client("demo-sft-step-1")
    sample = sampling_client.sample(
        types.ModelInput.from_ints(tokenizer.encode("Question: 3+5? Answer:")),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=8, temperature=0.0),
    ).result()
    print(tokenizer.decode(sample.sequences[0].tokens))


if __name__ == "__main__":
    main()
