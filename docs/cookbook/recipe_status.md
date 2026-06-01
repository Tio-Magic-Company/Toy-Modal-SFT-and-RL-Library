# Recipe Status

| Recipe family | Status | Prerequisites before promotion |
| --- | --- | --- |
| preference/DPO | Smoke scaffold | Choose a structured DPO backend loss or reviewed trusted custom-loss strategy. |
| prompt/model distillation | Recommended next promotion | Add teacher sampling/logprob collection and student KL or SFT training path. This should be promoted before unsafe custom-loss work because it can use existing sampler/logprob primitives. |
| tool use | Smoke scaffold | Stabilize tool-call rendering, environment contracts, and evaluator hooks. |
| multi-agent | Smoke scaffold | Stabilize sampler sessions, self-play/cross-play orchestration, and trajectory storage. |
| verifier/rubric | Smoke scaffold | Define external judge integration and logging without backend code execution. |
| VLM | Smoke scaffold | Validate image input parity and model support. |
| Harbor/agent RL | Smoke scaffold | Review sandboxing, cost controls, and environment lifecycle. |
| SDFT | Smoke scaffold | Promote top-k/logprob driven objective and eval flow. |
| True Thinking Score | Smoke scaffold | Keep as evaluation-only until scoring methodology and validation are checked in. |

These families should remain local, deterministic scaffolds until their required
sampler, session, eval, loss, and safety primitives are implemented and tested.

Before claiming deployed cookbook parity, archive a Modal report directory that
includes the HTTP gateway probe, `chat_sft.py`, `math_rl.py --loss-fn ppo`,
`math_rl.py --loss-fn cispo`, and `on_policy_rl_workflow.py` running with
`--transport modal-direct`.
