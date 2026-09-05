"""Validate the deterministic evaluation case contract."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path(__file__).parents[1] / "eval_cases.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"name", "category", "input", "expected_tools"}
    assert len(cases) >= 8
    for case in cases:
        assert required <= set(case), case
        assert isinstance(case["expected_tools"], list), case
    print(f"validated {len(cases)} evaluation cases")


if __name__ == "__main__":
    main()

