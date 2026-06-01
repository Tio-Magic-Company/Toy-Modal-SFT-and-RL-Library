"""Inspect run, checkpoint, session, and sampler metadata through RestClient."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from toy_modal import types
from common import add_service_args, service_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_service_args(parser, project_id="demo-rest")
    args = parser.parse_args()

    service = service_from_args(args)
    training = service.create_lora_training_client(
        args.base_model,
        user_metadata={"example": "rest_metadata"},
    )
    tokenizer = training.get_tokenizer()
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokenizer.encode("metadata")),
        loss_fn_inputs={"target_tokens": [1], "weights": [1.0]},
    )
    training.forward_backward([datum], "cross_entropy").result()
    training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    checkpoint = training.save_state("metadata-checkpoint").result()
    sampler_checkpoint = training.save_weights_for_sampler("metadata-sampler").result()

    rest = service.create_rest_client()
    sessions = rest.list_sessions().result()
    session = rest.get_session(sessions.sessions[0]).result()
    sampler = rest.get_sampler(session.sampler_ids[0]).result()
    archive = rest.get_checkpoint_archive_url_from_toy_path(checkpoint.path).result()
    print(
        json.dumps(
            {
                "checkpoint": checkpoint.path,
                "sampler_checkpoint": sampler_checkpoint.path,
                "session_id": session.session_id,
                "sampler_id": sampler.sampler_id,
                "archive_url": archive.url,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
