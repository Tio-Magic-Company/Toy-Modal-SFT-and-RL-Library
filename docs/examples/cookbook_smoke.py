"""Run one clean-room cookbook smoke recipe through the reusable helper."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from toy_modal.cookbook import RecipeConfig, run_smoke_recipe


def main() -> None:
    result = run_smoke_recipe(RecipeConfig(name="sl_loop", project_id="demo-cookbook"))
    print(json.dumps(result.to_record(), sort_keys=True))


if __name__ == "__main__":
    main()
