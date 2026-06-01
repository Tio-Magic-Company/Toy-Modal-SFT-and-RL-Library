"""Shared public defaults for Toy Modal examples and CLI entrypoints."""

DEFAULT_UNSLOTH_BASE_MODEL = "unsloth/tinyllama-bnb-4bit"
DEFAULT_TRANSFORMERS_BASE_MODEL = "hf-internal-testing/tiny-random-gpt2"

# The framework default follows the default backend engines.
DEFAULT_BASE_MODEL = DEFAULT_UNSLOTH_BASE_MODEL

# Advisory capability metadata for freshly deployed backends. These are not a
# closed allow-list; users can still pass any compatible Hugging Face model ID.
DEFAULT_UNSLOTH_CAPABILITY_MODELS = (
    DEFAULT_UNSLOTH_BASE_MODEL,
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-3-4B-it-bnb-4bit",
)
DEFAULT_TRANSFORMERS_CAPABILITY_MODELS = (
    DEFAULT_TRANSFORMERS_BASE_MODEL,
)
DEFAULT_UNSLOTH_MODEL_FAMILIES = (
    "Llama",
    "Mistral",
    "Qwen",
    "Gemma",
    "DeepSeek",
    "Phi",
    "Mixtral",
    "Yi",
)
