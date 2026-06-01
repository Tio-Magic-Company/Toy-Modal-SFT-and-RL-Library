"""Reusable cookbook loops for Modal-backed examples, CLI smoke tests, and recipes."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Protocol, Sequence

from toy_modal import types
from toy_modal.backend.loss_inputs import validate_training_batch
from toy_modal.clients.service_client import ServiceClient
from toy_modal.defaults import DEFAULT_BASE_MODEL
from toy_modal.renderers import Renderer, TokenWeights, TrainOnWhat


RECIPE_NAMES: tuple[str, ...] = (
    "sl_loop",
    "rl_loop",
    "sl_basic",
    "rl_basic",
    "chat_sft",
    "math_rl",
    "code_rl",
    "preference_dpo",
    "prompt_distillation",
    "model_distillation",
    "tool_use",
    "multi_agent",
    "rubric_grading",
    "verifier_environment",
    "vlm_image_classification",
    "harbor_rl",
    "sdft",
    "true_thinking_score",
    "eval_scaffold",
    "tiny_sft_workflow",
    "on_policy_rl_workflow",
)

RL_RECIPE_NAMES = {
    "rl_loop",
    "rl_basic",
    "math_rl",
    "code_rl",
    "tool_use",
    "multi_agent",
    "harbor_rl",
    "on_policy_rl_workflow",
}

SMOKE_SCAFFOLD_RECIPE_NAMES = RL_RECIPE_NAMES | {
    "preference_dpo",
    "prompt_distillation",
    "model_distillation",
}

RLLossFn = Literal["importance_sampling", "ppo", "cispo"]
DEFAULT_MODAL_BASE_MODEL = DEFAULT_BASE_MODEL


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class RenderedExample:
    model_input: types.ModelInput
    target_tokens: list[int]
    weights: list[float]
    text: str

    def to_datum(self) -> types.Datum:
        return types.Datum(
            model_input=self.model_input,
            loss_fn_inputs={
                "target_tokens": self.target_tokens,
                "weights": self.weights,
            },
        )


class ChatTemplateRenderer:
    """Small clean-room renderer for chat JSONL recipes.

    The default template is deliberately simple and tokenizer-agnostic. Model
    families with stricter templates can pass explicit role prefixes.
    """

    def __init__(
        self,
        *,
        role_prefixes: dict[str, str] | None = None,
        assistant_roles: Sequence[str] = ("assistant",),
    ) -> None:
        self.role_prefixes = role_prefixes or {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
        }
        self.assistant_roles = tuple(assistant_roles)

    def render_messages(self, messages: Sequence[Message], *, add_generation_prompt: bool = False) -> str:
        parts = []
        for message in messages:
            prefix = self.role_prefixes.get(message.role, message.role.title())
            parts.append(f"{prefix}: {message.content.strip()}")
        if add_generation_prompt:
            prefix = self.role_prefixes.get("assistant", "Assistant")
            parts.append(f"{prefix}:")
        return "\n".join(parts)

    def render_supervised(self, tokenizer, messages: Sequence[Message]) -> RenderedExample:
        if not messages:
            raise ValueError("messages must not be empty")
        target_start = _last_role_index(messages, self.assistant_roles)
        if target_start is None:
            raise ValueError("messages must contain at least one assistant message")

        prompt_messages = messages[:target_start]
        prompt_text = self.render_messages(prompt_messages, add_generation_prompt=bool(prompt_messages))
        target_text = messages[target_start].content.strip()
        if prompt_text:
            target_text = f" {target_text}"
            full_text = f"{prompt_text}{target_text}"
        else:
            full_text = target_text
        prompt_tokens = tokenizer.encode(prompt_text + ("\n" if prompt_text else ""))
        target_tokens = tokenizer.encode(target_text)
        return RenderedExample(
            model_input=types.ModelInput.from_ints([*prompt_tokens, *target_tokens]),
            target_tokens=target_tokens,
            weights=[0.0] * len(prompt_tokens) + [1.0] * len(target_tokens),
            text=full_text,
        )


@dataclass(frozen=True)
class TrainLoopConfig:
    loss_fn: types.LossFnType = "cross_entropy"
    learning_rate: float = 1e-4
    steps: int = 1
    checkpoint_every: int = 1
    checkpoint_prefix: str = "checkpoint"


@dataclass(frozen=True)
class TrainLoopResult:
    losses: list[float | None]
    optimizer_step: int
    checkpoints: list[str]


@dataclass(frozen=True)
class StepResult:
    observation: str
    reward: float
    done: bool = True
    info: dict[str, object] | None = None

    @property
    def episode_done(self) -> bool:
        return self.done

    @property
    def metrics(self) -> dict[str, object]:
        return dict(self.info or {})


class Env(Protocol):
    def initial_observation(self) -> str: ...

    def step(self, action: str) -> StepResult: ...


class EnvGroupBuilder(Protocol):
    def build_group(self, prompt: str, group_size: int) -> Sequence[Env]: ...


@dataclass(frozen=True)
class RLDataset:
    prompts: list[str]

    def __iter__(self) -> Iterable[str]:
        return iter(self.prompts)


@dataclass(frozen=True)
class Trajectory:
    prompt: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    old_logprobs: list[float]
    reward: float
    text: str = ""
    metadata: dict[str, object] | None = None

    @property
    def total_reward(self) -> float:
        return self.reward


@dataclass(frozen=True)
class TrajectoryGroup:
    prompt: str
    trajectories: list[Trajectory]

    @property
    def trajectories_G(self) -> list[Trajectory]:
        return self.trajectories

    def get_total_rewards(self) -> list[float]:
        return [trajectory.reward for trajectory in self.trajectories]


class ProblemEnv:
    """Base class for single-step, problem/answer RL environments."""

    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: Renderer,
        *,
        convo_prefix: Sequence[Message | dict[str, object]] | None = None,
        format_coef: float = 0.1,
    ) -> None:
        self.problem = problem
        self.answer = answer
        self.renderer = renderer
        self.convo_prefix = list(convo_prefix or [])
        self.format_coef = format_coef
        self._spent = False

    def get_question(self) -> str:
        return self.problem

    def get_reference_answer(self) -> str:
        return self.answer

    def check_answer(self, text: str) -> bool:
        return self.answer.strip() in text

    def check_format(self, text: str) -> bool:
        return bool(text.strip())

    async def initial_observation(self) -> tuple[types.ModelInput, list[str | int]]:
        messages = [
            *self.convo_prefix,
            {"role": "user", "content": self.get_question()},
        ]
        prompt = await asyncio.to_thread(self.renderer.build_generation_prompt, messages)
        return prompt, self.renderer.get_stop_sequences()

    async def step(self, action: Sequence[int] | str) -> StepResult:
        if self._spent:
            raise RuntimeError("ProblemEnv instances are single-use")
        self._spent = True
        tokenizer = getattr(self.renderer, "tokenizer", None)
        if isinstance(action, str):
            text = action
        elif tokenizer is not None:
            text = await asyncio.to_thread(tokenizer.decode, list(action))
        else:
            text = " ".join(str(token) for token in action)
        correct_answer = self.check_answer(text)
        correct_format = self.check_format(text)
        reward = (1.0 if correct_answer else 0.0) + self.format_coef * (
            0.0 if correct_format else -1.0
        )
        return StepResult(
            observation="",
            reward=reward,
            done=True,
            info={
                "correct_answer": correct_answer,
                "correct_format": correct_format,
                "reference_answer": self.get_reference_answer(),
                "text": text,
            },
        )


@dataclass(frozen=True)
class ProblemGroupBuilder:
    env_thunk: Callable[[], ProblemEnv]
    num_envs: int

    async def make_envs(self) -> list[ProblemEnv]:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(self, trajectories: Sequence[Trajectory]) -> list[float]:
        return [0.0 for _ in trajectories]

    async def cleanup(self) -> None:
        return None


class RolloutStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_group(self, group: TrajectoryGroup) -> None:
        _append_jsonl(
            self.path,
            {
                "prompt": group.prompt,
                "trajectories": [asdict(item) for item in group.trajectories],
            },
        )

    def read_groups(self) -> list[TrajectoryGroup]:
        if not self.path.exists():
            return []
        groups = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            groups.append(
                TrajectoryGroup(
                    prompt=record["prompt"],
                    trajectories=[Trajectory(**item) for item in record["trajectories"]],
                )
            )
        return groups


class Evaluator(Protocol):
    def evaluate(self, sampler) -> dict[str, float]: ...


@dataclass(frozen=True)
class NLLEvaluator:
    name: str
    datums: list[types.Datum]

    def evaluate(self, training, *, loss_fn: types.LossFnType = "cross_entropy") -> dict[str, float]:
        if not self.datums:
            return {f"{self.name}/nll": 0.0}
        result = training.forward(self.datums, loss_fn).result()
        return {f"{self.name}/nll": float(result.loss or 0.0)}


@dataclass(frozen=True)
class SamplingEvaluator:
    name: str
    prompts: list[str]
    grader: Callable[[str, str], float]
    max_tokens: int = 4

    def evaluate(self, sampler, tokenizer) -> dict[str, float]:
        return LocalBenchmark(self.name, self.prompts, self.grader).run(
            sampler,
            tokenizer,
            max_tokens=self.max_tokens,
        )


@dataclass(frozen=True)
class LocalBenchmark:
    name: str
    prompts: list[str]
    grader: Callable[[str, str], float]

    def run(self, sampler, tokenizer, *, max_tokens: int = 4) -> dict[str, float]:
        scores = []
        for prompt in self.prompts:
            response = sampler.sample(
                types.ModelInput.from_ints(tokenizer.encode(prompt)),
                1,
                types.SamplingParams(max_tokens=max_tokens, temperature=0.0),
            ).result()
            completion = response.samples[0].tokens[-max_tokens:]
            scores.append(float(self.grader(prompt, tokenizer.decode(completion))))
        return {f"{self.name}/score": sum(scores) / max(1, len(scores))}


@dataclass(frozen=True)
class RecipeConfig:
    name: str
    transport: str = "modal-direct"
    base_model: str = DEFAULT_MODAL_BASE_MODEL
    project_id: str = "recipe"
    app_name: str = field(default_factory=lambda: os.getenv("TOY_MODAL_APP_NAME", "toy-modal-backend"))
    environment_name: str | None = field(default_factory=lambda: os.getenv("TOY_MODAL_ENVIRONMENT"))
    base_url: str | None = None
    api_key: str | None = None
    log_path: str | Path | None = None
    max_tokens: int = 3
    seed: int = 7


@dataclass(frozen=True)
class RecipeResult:
    recipe: str
    training_run_id: str
    model_path: str
    loss: float | None
    optimizer_step: int
    sample_tokens: list[int]
    sample_text: str
    prompt_logprobs: list[float | None] | None
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def list_recipes() -> tuple[str, ...]:
    return RECIPE_NAMES


def build_service(config: RecipeConfig) -> ServiceClient:
    kwargs = {
        "project_id": config.project_id,
        "transport": config.transport,
        "app_name": config.app_name or os.getenv("TOY_MODAL_APP_NAME", "toy-modal-backend"),
        "environment_name": config.environment_name or os.getenv("TOY_MODAL_ENVIRONMENT"),
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return ServiceClient(**kwargs)


def run_smoke_recipe(
    config: RecipeConfig,
    *,
    service: ServiceClient | None = None,
) -> RecipeResult:
    if config.name not in RECIPE_NAMES:
        raise ValueError(f"unknown recipe: {config.name}")

    service = service or build_service(config)
    training = service.create_lora_training_client(
        config.base_model,
        rank=4,
        user_metadata={"recipe": config.name},
    )
    tokenizer = training.get_tokenizer()
    prompt_text = f"{config.name}: answer briefly"
    prompt_tokens = tokenizer.encode(prompt_text)
    target_tokens = _target_tokens(config.name)
    datum = _datum_for_recipe(config.name, prompt_tokens, target_tokens)

    loss_fn: types.LossFnType = (
        "importance_sampling" if config.name in RL_RECIPE_NAMES else "cross_entropy"
    )
    validate_training_batch([datum], loss_fn)
    loss = training.forward_backward([datum], loss_fn).result()
    step = training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    sampler = training.save_weights_and_get_sampling_client(f"{config.name}-sampler")
    sample = sampler.sample(
        types.ModelInput.from_ints(tokenizer.encode(config.name)),
        1,
        types.SamplingParams(max_tokens=config.max_tokens, seed=config.seed),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=2,
    ).result()
    sequence = sample.samples[0]
    generated = sequence.tokens[-config.max_tokens :] if config.max_tokens else []
    result = RecipeResult(
        recipe=config.name,
        training_run_id=training.training_run_id,
        model_path=sampler.model_path or "",
        loss=loss.loss,
        optimizer_step=step.optimizer_step,
        sample_tokens=sequence.tokens,
        sample_text=tokenizer.decode(generated),
        prompt_logprobs=sample.prompt_logprobs,
        topk_prompt_logprobs=sample.topk_prompt_logprobs,
    )
    if config.log_path is not None:
        write_recipe_outputs(config.log_path, result)
    return result


def run_many_smoke_recipes(
    names: Sequence[str] | None = None,
    *,
    transport: str = "modal-direct",
    base_model: str = DEFAULT_MODAL_BASE_MODEL,
    project_id: str = "recipe",
    app_name: str | None = None,
    environment_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    log_path: str | Path | None = None,
) -> list[RecipeResult]:
    selected = tuple(names or RECIPE_NAMES)
    return [
        run_smoke_recipe(
            RecipeConfig(
                name=name,
                transport=transport,
                base_model=base_model,
                project_id=project_id,
                app_name=app_name or os.getenv("TOY_MODAL_APP_NAME", "toy-modal-backend"),
                environment_name=environment_name or os.getenv("TOY_MODAL_ENVIRONMENT"),
                base_url=base_url,
                api_key=api_key,
                log_path=(Path(log_path) / name if log_path else None),
            )
        )
        for name in selected
    ]


def run_supervised_train_loop(
    training,
    datums: Sequence[types.Datum],
    config: TrainLoopConfig,
) -> TrainLoopResult:
    if not datums:
        raise ValueError("datums must not be empty")
    losses: list[float | None] = []
    checkpoints: list[str] = []
    for step_index in range(config.steps):
        validate_training_batch(list(datums), config.loss_fn)
        loss = training.forward_backward(list(datums), config.loss_fn).result()
        optim = training.optim_step(types.AdamParams(learning_rate=config.learning_rate)).result()
        losses.append(loss.loss)
        if config.checkpoint_every and (step_index + 1) % config.checkpoint_every == 0:
            checkpoint = training.save_state(f"{config.checkpoint_prefix}-{step_index + 1}").result()
            checkpoints.append(checkpoint.path)
    return TrainLoopResult(
        losses=losses,
        optimizer_step=optim.optimizer_step if losses else training.optimizer_step,
        checkpoints=checkpoints,
    )


def run_rl_train_loop(
    training,
    datums: Sequence[types.Datum],
    config: TrainLoopConfig,
    *,
    loss_fn_config: dict[str, float] | None = None,
) -> TrainLoopResult:
    if config.loss_fn not in {"importance_sampling", "ppo", "cispo"}:
        raise ValueError("run_rl_train_loop requires an RL loss_fn")
    if not datums:
        raise ValueError("datums must not be empty")
    losses: list[float | None] = []
    checkpoints: list[str] = []
    for step_index in range(config.steps):
        validate_training_batch(list(datums), config.loss_fn)
        loss = training.forward_backward(
            list(datums),
            config.loss_fn,
            loss_fn_config=loss_fn_config,
        ).result()
        optim = training.optim_step(types.AdamParams(learning_rate=config.learning_rate)).result()
        losses.append(loss.loss)
        if config.checkpoint_every and (step_index + 1) % config.checkpoint_every == 0:
            checkpoint = training.save_state(f"{config.checkpoint_prefix}-{step_index + 1}").result()
            checkpoints.append(checkpoint.path)
    return TrainLoopResult(
        losses=losses,
        optimizer_step=optim.optimizer_step,
        checkpoints=checkpoints,
    )


def load_conversation_jsonl(path: str | Path) -> list[list[Message]]:
    conversations: list[list[Message]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "messages" in record:
                conversations.append(
                    [
                        Message(role=str(item["role"]), content=str(item["content"]))
                        for item in record["messages"]
                    ]
                )
            elif "prompt" in record and "completion" in record:
                conversations.append(
                    [
                        Message(role="user", content=str(record["prompt"])),
                        Message(role="assistant", content=str(record["completion"])),
                    ]
                )
            else:
                raise ValueError(
                    f"{path}:{line_number} must contain messages or prompt/completion fields"
                )
    return conversations


def render_conversation_datums(
    tokenizer,
    conversations: Sequence[Sequence[Message]],
    *,
    renderer: ChatTemplateRenderer | None = None,
) -> list[types.Datum]:
    renderer = renderer or ChatTemplateRenderer()
    return [renderer.render_supervised(tokenizer, messages).to_datum() for messages in conversations]


def collect_grouped_rollouts(
    *,
    sampler,
    tokenizer,
    prompts: Sequence[str],
    group_size: int,
    sampling_params: types.SamplingParams,
    reward_fn: Callable[[str, str], float],
    store: RolloutStore | None = None,
) -> list[TrajectoryGroup]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    groups: list[TrajectoryGroup] = []
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt)
        response = sampler.sample(
            types.ModelInput.from_ints(prompt_tokens),
            group_size,
            sampling_params,
        ).result()
        trajectories = []
        for sequence in response.samples:
            completion = sequence.tokens[len(prompt_tokens) :]
            text = tokenizer.decode(completion)
            old_logprobs = sequence.logprobs or [0.0] * len(completion)
            trajectories.append(
                Trajectory(
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion,
                    old_logprobs=[float(value) for value in old_logprobs],
                    reward=float(reward_fn(prompt, text)),
                    text=text,
                )
            )
        group = TrajectoryGroup(prompt=prompt, trajectories=trajectories)
        if store is not None:
            store.append_group(group)
        groups.append(group)
    return groups


def group_relative_advantages(
    rewards: Sequence[float],
    *,
    skip_degenerate: bool = True,
    eps: float = 1e-8,
) -> list[float] | None:
    if not rewards:
        return None
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = variance ** 0.5
    if std <= eps:
        if skip_degenerate:
            return None
        return [0.0 for _ in rewards]
    return [(reward - mean) / std for reward in rewards]


def grpo_datums_from_trajectory_groups(
    groups: Sequence[TrajectoryGroup],
    *,
    loss_fn: RLLossFn = "importance_sampling",
    model_seq_id: int | None = None,
    skip_degenerate: bool = True,
) -> list[types.Datum]:
    if loss_fn not in {"importance_sampling", "ppo", "cispo"}:
        raise ValueError(f"unsupported GRPO loss_fn: {loss_fn!r}")
    datums: list[types.Datum] = []
    for group in groups:
        rewards = [trajectory.reward for trajectory in group.trajectories]
        advantages = group_relative_advantages(rewards, skip_degenerate=skip_degenerate)
        if advantages is None:
            continue
        for trajectory, advantage in zip(group.trajectories, advantages):
            if not trajectory.completion_tokens:
                continue
            inputs: dict[str, object] = {
                "target_tokens": trajectory.completion_tokens,
                "old_logprobs": trajectory.old_logprobs,
                "logprobs": trajectory.old_logprobs,
                "advantages": [float(advantage)] * len(trajectory.completion_tokens),
                "weights": [1.0] * len(trajectory.completion_tokens),
                "masks": [1.0] * len(trajectory.completion_tokens),
            }
            if model_seq_id is not None:
                inputs["old_logprobs_model_seq_id"] = model_seq_id
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(
                    [*trajectory.prompt_tokens, *trajectory.completion_tokens]
                ),
                loss_fn_inputs=inputs,
            )
            validate_training_batch([datum], loss_fn)
            datums.append(datum)
    return datums


async def do_group_rollout(group_builder, policy) -> TrajectoryGroup:
    """Run one group of single-step environments with a token completer."""

    envs = await _maybe_await(group_builder.make_envs())
    trajectories: list[Trajectory] = []
    prompt_text = ""
    try:
        for env in envs:
            observation, stop = await _maybe_await(env.initial_observation())
            prompt_tokens = observation.to_ints()
            token_result = await policy(observation, stop=stop)
            step = await _maybe_await(env.step(token_result.tokens))
            prompt_text = getattr(env, "problem", prompt_text)
            trajectories.append(
                Trajectory(
                    prompt=prompt_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=list(token_result.tokens),
                    old_logprobs=list(token_result.logprobs),
                    reward=float(step.reward),
                    text=str(step.metrics.get("text", "")),
                    metadata=dict(step.metrics),
                )
            )
        if hasattr(group_builder, "compute_group_rewards"):
            group_rewards = await _maybe_await(group_builder.compute_group_rewards(trajectories))
            trajectories = [
                trajectory.__class__(
                    prompt=trajectory.prompt,
                    prompt_tokens=trajectory.prompt_tokens,
                    completion_tokens=trajectory.completion_tokens,
                    old_logprobs=trajectory.old_logprobs,
                    reward=trajectory.reward + float(extra_reward),
                    text=trajectory.text,
                    metadata=trajectory.metadata,
                )
                for trajectory, extra_reward in zip(trajectories, group_rewards)
            ]
        return TrajectoryGroup(prompt=prompt_text, trajectories=trajectories)
    finally:
        if hasattr(group_builder, "cleanup"):
            await _maybe_await(group_builder.cleanup())


async def do_group_rollout_and_filter_constant_reward(group_builder, policy) -> TrajectoryGroup | None:
    group = await do_group_rollout(group_builder, policy)
    return None if group_relative_advantages(group.get_total_rewards()) is None else group


def compute_advantages(groups: Sequence[TrajectoryGroup]) -> list[TokenWeights]:
    advantages = []
    for group in groups:
        centered = group_relative_advantages(group.get_total_rewards(), skip_degenerate=False)
        advantages.append(TokenWeights(centered or []))
    return advantages


def remove_constant_reward_groups(groups: Sequence[TrajectoryGroup]) -> list[TrajectoryGroup]:
    return [
        group
        for group in groups
        if group_relative_advantages(group.get_total_rewards()) is not None
    ]


def trajectory_to_data(
    trajectory: Trajectory,
    advantage: float,
    *,
    model_seq_id: int | None = None,
) -> types.Datum:
    inputs: dict[str, object] = {
        "target_tokens": trajectory.completion_tokens,
        "old_logprobs": trajectory.old_logprobs,
        "logprobs": trajectory.old_logprobs,
        "advantages": [float(advantage)] * len(trajectory.completion_tokens),
        "weights": [1.0] * len(trajectory.completion_tokens),
        "masks": [1.0] * len(trajectory.completion_tokens),
    }
    if model_seq_id is not None:
        inputs["old_logprobs_model_seq_id"] = model_seq_id
    return types.Datum(
        model_input=types.ModelInput.from_ints(
            [*trajectory.prompt_tokens, *trajectory.completion_tokens]
        ),
        loss_fn_inputs=inputs,
    )


def assemble_training_data(
    groups: Sequence[TrajectoryGroup],
    advantages: Sequence[Sequence[float]],
    *,
    model_seq_id: int | None = None,
) -> tuple[list[types.Datum], list[dict[str, object]]]:
    datums: list[types.Datum] = []
    metadata: list[dict[str, object]] = []
    for group, group_advantages in zip(groups, advantages):
        for trajectory, advantage in zip(group.trajectories, group_advantages):
            if not trajectory.completion_tokens:
                continue
            datums.append(trajectory_to_data(trajectory, float(advantage), model_seq_id=model_seq_id))
            metadata.append(
                {
                    "prompt": group.prompt,
                    "reward": trajectory.reward,
                    "advantage": float(advantage),
                    **(trajectory.metadata or {}),
                }
            )
    return datums, metadata


@dataclass(frozen=True)
class ChatDatasetBuilderCommonConfig:
    model_name_for_tokenizer: str
    renderer_name: str = "role_colon"
    max_length: int = 2048
    batch_size: int = 1
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE


class SupervisedDataset:
    def __iter__(self) -> Iterable[list[types.Datum]]: ...


@dataclass(frozen=True)
class InMemorySupervisedDataset:
    datums: list[types.Datum]
    batch_size: int = 1

    def __iter__(self) -> Iterable[list[types.Datum]]:
        for index in range(0, len(self.datums), self.batch_size):
            yield self.datums[index : index + self.batch_size]

    def __len__(self) -> int:
        return (len(self.datums) + self.batch_size - 1) // self.batch_size


class ChatDatasetBuilder(Protocol):
    common_config: ChatDatasetBuilderCommonConfig

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]: ...


@dataclass(frozen=True)
class KLReferenceConfig:
    coef: float = 0.0
    reference_model: str | None = None


@dataclass(frozen=True)
class SupervisedTrainConfig:
    dataset_builder: ChatDatasetBuilder
    model_name: str = DEFAULT_MODAL_BASE_MODEL
    transport: str = "modal-direct"
    project_id: str = "tutorial"
    app_name: str = field(default_factory=lambda: os.getenv("TOY_MODAL_APP_NAME", "toy-modal-backend"))
    environment_name: str | None = field(default_factory=lambda: os.getenv("TOY_MODAL_ENVIRONMENT"))
    base_url: str | None = None
    api_key: str | None = None
    learning_rate: float = 1e-4
    lora_rank: int = 4
    max_steps: int = 1
    save_every: int = 1
    log_path: str | Path | None = None


@dataclass(frozen=True)
class RLTrainConfig:
    prompts: list[str]
    model_name: str = DEFAULT_MODAL_BASE_MODEL
    transport: str = "modal-direct"
    project_id: str = "tutorial-rl"
    app_name: str = field(default_factory=lambda: os.getenv("TOY_MODAL_APP_NAME", "toy-modal-backend"))
    environment_name: str | None = field(default_factory=lambda: os.getenv("TOY_MODAL_ENVIRONMENT"))
    base_url: str | None = None
    api_key: str | None = None
    loss_fn: RLLossFn = "importance_sampling"
    learning_rate: float = 1e-4
    lora_rank: int = 4
    group_size: int = 2
    max_steps: int = 1
    max_tokens: int = 2
    kl_reference: KLReferenceConfig | None = None
    log_path: str | Path | None = None


@dataclass(frozen=True)
class SweepResult:
    params: dict[str, object]
    metrics: dict[str, float]


def get_lr(model_name: str, *, default: float = 1e-4) -> float:
    lowered = model_name.lower()
    if "tiny" in lowered or "local" in lowered:
        return default
    if "30b" in lowered or "70b" in lowered:
        return 5e-5
    return default


def linear_lr(step: int, total_steps: int, peak_lr: float) -> float:
    if total_steps <= 1:
        return peak_lr
    return peak_lr * max(0.0, 1.0 - (step / (total_steps - 1)))


def grid_sweep(param_grid: dict[str, Sequence[object]]) -> list[dict[str, object]]:
    items = list(param_grid.items())
    if not items:
        return [{}]
    results: list[dict[str, object]] = [{}]
    for key, values in items:
        results = [{**base, key: value} for base in results for value in values]
    return results


def estimate_lora_parameters(
    *,
    hidden_size: int,
    num_layers: int,
    rank: int,
    projections_per_layer: int = 4,
) -> int:
    if min(hidden_size, num_layers, rank, projections_per_layer) < 0:
        raise ValueError("LoRA parameter dimensions must be non-negative")
    return 2 * hidden_size * rank * projections_per_layer * num_layers


def run_supervised_config(config: SupervisedTrainConfig) -> TrainLoopResult:
    service_kwargs = {
        "project_id": config.project_id,
        "transport": config.transport,
        "app_name": config.app_name,
        "environment_name": config.environment_name,
    }
    if config.base_url:
        service_kwargs["base_url"] = config.base_url
    if config.api_key:
        service_kwargs["api_key"] = config.api_key
    service = ServiceClient(**service_kwargs)
    training = service.create_lora_training_client(config.model_name, rank=config.lora_rank)
    train_dataset, _eval_dataset = config.dataset_builder()
    losses: list[float | None] = []
    checkpoints: list[str] = []
    optimizer_step = training.optimizer_step
    for step_index, batch in enumerate(train_dataset):
        if step_index >= config.max_steps:
            break
        result = training.forward_backward(batch, "cross_entropy").result()
        step = training.optim_step(types.AdamParams(learning_rate=config.learning_rate)).result()
        losses.append(result.loss)
        optimizer_step = step.optimizer_step
        if config.save_every and (step_index + 1) % config.save_every == 0:
            checkpoint = training.save_state(f"config-step-{step_index + 1}").result()
            checkpoints.append(checkpoint.path)
    loop_result = TrainLoopResult(losses=losses, optimizer_step=optimizer_step, checkpoints=checkpoints)
    if config.log_path is not None:
        path = Path(config.log_path)
        path.mkdir(parents=True, exist_ok=True)
        _append_jsonl(path / "metrics.jsonl", asdict(loop_result))
    return loop_result


def run_rl_config(
    config: RLTrainConfig,
    *,
    reward_fn: Callable[[str, str], float] | None = None,
) -> TrainLoopResult:
    service_kwargs = {
        "project_id": config.project_id,
        "transport": config.transport,
        "app_name": config.app_name,
        "environment_name": config.environment_name,
    }
    if config.base_url:
        service_kwargs["base_url"] = config.base_url
    if config.api_key:
        service_kwargs["api_key"] = config.api_key
    service = ServiceClient(**service_kwargs)
    training = service.create_lora_training_client(config.model_name, rank=config.lora_rank)
    tokenizer = training.get_tokenizer()
    sampler = training.save_weights_and_get_sampling_client("rl-config-initial")
    groups = collect_grouped_rollouts(
        sampler=sampler,
        tokenizer=tokenizer,
        prompts=config.prompts,
        group_size=config.group_size,
        sampling_params=types.SamplingParams(max_tokens=config.max_tokens, seed=13),
        reward_fn=reward_fn or (lambda _prompt, completion: 1.0 if completion else 0.0),
    )
    datums = grpo_datums_from_trajectory_groups(
        groups,
        loss_fn=config.loss_fn,
        model_seq_id=training.model_seq_id,
        skip_degenerate=False,
    )
    result = run_rl_train_loop(
        training,
        datums,
        TrainLoopConfig(
            loss_fn=config.loss_fn,
            learning_rate=config.learning_rate,
            steps=config.max_steps,
            checkpoint_prefix="rl-config",
        ),
        loss_fn_config={"kl_coef": config.kl_reference.coef} if config.kl_reference else None,
    )
    if config.log_path is not None:
        path = Path(config.log_path)
        path.mkdir(parents=True, exist_ok=True)
        _append_jsonl(path / "metrics.jsonl", asdict(result))
    return result


@dataclass(frozen=True)
class Transition:
    observation: types.ModelInput
    action_tokens: list[int]
    reward: float
    done: bool = True
    metadata: dict[str, object] | None = None


class MessageEnv:
    def __init__(
        self,
        messages: Sequence[dict[str, object]],
        *,
        reward_fn: Callable[[dict[str, object]], float] | None = None,
    ) -> None:
        self.messages = [dict(message) for message in messages]
        self.reward_fn = reward_fn or (lambda message: 1.0 if message.get("content") else 0.0)

    def initial_messages(self) -> list[dict[str, object]]:
        return [dict(message) for message in self.messages]

    def step_message(self, message: dict[str, object]) -> StepResult:
        reward = float(self.reward_fn(message))
        return StepResult(
            observation=str(message.get("content", "")),
            reward=reward,
            done=True,
            info={"message": message},
        )


class EnvFromMessageEnv(ProblemEnv):
    def __init__(self, message_env: MessageEnv, renderer: Renderer) -> None:
        self.message_env = message_env
        super().__init__("", "", renderer)

    async def initial_observation(self) -> tuple[types.ModelInput, list[str | int]]:
        return (
            self.renderer.build_generation_prompt(self.message_env.initial_messages()),
            self.renderer.get_stop_sequences(),
        )

    async def step(self, action: Sequence[int] | str) -> StepResult:
        tokenizer = getattr(self.renderer, "tokenizer", None)
        content = action if isinstance(action, str) else tokenizer.decode(list(action))
        return self.message_env.step_message({"role": "assistant", "content": content})


@dataclass(frozen=True)
class Comparison:
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class LabeledComparison(Comparison):
    label: int = 1


class ComparisonRenderer:
    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer

    def to_datums(self, tokenizer, comparison: Comparison) -> tuple[types.Datum, types.Datum]:
        chosen = [
            {"role": "user", "content": comparison.prompt},
            {"role": "assistant", "content": comparison.chosen},
        ]
        rejected = [
            {"role": "user", "content": comparison.prompt},
            {"role": "assistant", "content": comparison.rejected},
        ]
        return (
            self.renderer.conversation_to_datum(chosen),  # type: ignore[attr-defined]
            self.renderer.conversation_to_datum(rejected),  # type: ignore[attr-defined]
        )


class ComparisonRendererFromChatRenderer(ComparisonRenderer):
    pass


class PreferenceModel:
    def score(self, prompt: str, completion: str) -> float:
        return float(len(completion) - len(prompt)) / max(1, len(completion))


class PreferenceModelFromChatRenderer(PreferenceModel):
    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer


def build_dpo_datums(
    tokenizer,
    comparisons: Sequence[Comparison],
    renderer: Renderer,
) -> list[types.Datum]:
    comparison_renderer = ComparisonRenderer(renderer)
    datums: list[types.Datum] = []
    for comparison in comparisons:
        chosen, rejected = comparison_renderer.to_datums(tokenizer, comparison)
        chosen.loss_fn_inputs["preference_label"] = 1
        chosen.loss_fn_inputs["rejected_tokens"] = rejected.model_input.to_ints()
        chosen.loss_fn_inputs["rejected_target_tokens"] = rejected.loss_fn_inputs["target_tokens"]
        chosen.loss_fn_inputs.setdefault("beta", 0.1)
        chosen.loss_fn_inputs.setdefault("reference_chosen_logprob", 0.0)
        chosen.loss_fn_inputs.setdefault("reference_rejected_logprob", 0.0)
        datums.append(chosen)
    return datums


def preference_reward(model: PreferenceModel, prompt: str, completion: str) -> float:
    return model.score(prompt, completion)


@dataclass(frozen=True)
class RLHFPipelineResult:
    sft: TrainLoopResult
    preference_datums: int
    rl: TrainLoopResult


def run_rlhf_pipeline(
    *,
    service: ServiceClient,
    renderer: Renderer,
    conversations: Sequence[Sequence[dict[str, object]]],
    comparisons: Sequence[Comparison],
) -> RLHFPipelineResult:
    training = service.create_lora_training_client(DEFAULT_MODAL_BASE_MODEL, rank=4)
    tokenizer = training.get_tokenizer()
    sft_datums = [
        renderer.conversation_to_datum(messages)  # type: ignore[attr-defined]
        for messages in conversations
    ]
    sft = run_supervised_train_loop(training, sft_datums, TrainLoopConfig(steps=1))
    preference_datums = build_dpo_datums(tokenizer, comparisons, renderer)
    if preference_datums:
        training.forward_backward(preference_datums, "dpo").result()
        training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    rl = run_rl_train_loop(
        training,
        grpo_datums_from_trajectory_groups(
            [
                TrajectoryGroup(
                    prompt="rlhf",
                    trajectories=[Trajectory("rlhf", [1], [2], [-0.1], 1.0)],
                )
            ],
            model_seq_id=training.model_seq_id,
            skip_degenerate=False,
        ),
        TrainLoopConfig(loss_fn="importance_sampling", steps=1, checkpoint_prefix="rlhf-rl"),
    )
    return RLHFPipelineResult(sft=sft, preference_datums=len(preference_datums), rl=rl)


def write_recipe_outputs(log_path: str | Path, result: RecipeResult) -> None:
    path = Path(log_path)
    path.mkdir(parents=True, exist_ok=True)
    _append_jsonl(path / "metrics.jsonl", result.to_record())
    _append_jsonl(
        path / "checkpoints.jsonl",
        {
            "recipe": result.recipe,
            "training_run_id": result.training_run_id,
            "model_path": result.model_path,
            "optimizer_step": result.optimizer_step,
        },
    )


def _datum_for_recipe(name: str, prompt_tokens: list[int], target_tokens: list[int]) -> types.Datum:
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(target_tokens)
    inputs: dict[str, object] = {
        "target_tokens": target_tokens,
        "weights": weights,
    }
    if name in RL_RECIPE_NAMES:
        inputs.update(
            {
                "logprobs": [-0.5] * max(1, len(target_tokens)),
                "old_logprobs": [-0.5] * max(1, len(target_tokens)),
                "advantages": [1.0] * max(1, len(target_tokens)),
            }
        )
    return types.Datum(
        model_input=types.ModelInput.from_ints([*prompt_tokens, *target_tokens]),
        loss_fn_inputs=inputs,
    )


def _target_tokens(name: str) -> list[int]:
    base = sum(name.encode("utf-8")) % 17
    return [base + 1, base + 2, base + 3]


def _last_role_index(messages: Sequence[Message], roles: Sequence[str]) -> int | None:
    role_set = set(roles)
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role in role_set:
            return index
    return None


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
