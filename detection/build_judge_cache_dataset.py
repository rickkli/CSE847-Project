# Modified by Cursor: one-off script to build a full-dataset NIM judge cache without LR metadata.
"""
Reads ``data/llm_hallucination_dataset_v1.csv``, calls NVIDIA NIM once per row (text-only judge
prompt), writes ``data/judge_cache_dataset_v1.json``.

Requires NVIDIA_API_KEY (or .env). Run:

  python build_judge_cache_dataset.py

If the output file already exists, only missing ``record_id`` keys are fetched (resume after
interruption). Delete the output file to rebuild from scratch.

NIM calls are batched: after every 40 requests the script sleeps 61 seconds to stay under 40 RPM.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from llm_hallucination_judge import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_SAMPLE_KEYS,
    call_nim_judge,
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = _DATA_DIR / "llm_hallucination_dataset_v1.csv"
OUTPUT_PATH = _DATA_DIR / "judge_cache_dataset_v1.json"
NIM_JUDGE_MODEL = os.getenv("NIM_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

# Stay under 40 RPM: pause after each batch of 40 NIM calls (>60s cooldown).
REQUESTS_PER_BATCH = 40
BATCH_COOLDOWN_SEC = 61.0


class NimRateLimiter:
    """Pause after every ``REQUESTS_PER_BATCH`` API calls to stay below 40 requests/minute."""

    def __init__(self) -> None:
        self._requests_in_batch = 0

    def after_request(self) -> None:
        self._requests_in_batch += 1
        if self._requests_in_batch < REQUESTS_PER_BATCH:
            return
        print(
            f"Rate limit: {REQUESTS_PER_BATCH} NIM calls in this window; "
            f"sleeping {BATCH_COOLDOWN_SEC:.0f}s..."
        )
        time.sleep(BATCH_COOLDOWN_SEC)
        self._requests_in_batch = 0


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Cache must be a JSON object keyed by record_id: {path}")
    return data


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
        f.write("\n")


def row_sample(row: pd.Series) -> dict[str, str]:
    return {c: str(row[c]) for c in JUDGE_SAMPLE_KEYS if c in row.index and pd.notna(row[c])}


def build_text_only_judge_messages(sample: Mapping[str, Any]) -> list[dict[str, str]]:
    from llm_hallucination_judge import _sanitize_sample

    fields = _sanitize_sample(sample)
    meta_lines = [
        f"- record_id: {fields['record_id']}" if "record_id" in fields else None,
        f"- prompt_type: {fields['prompt_type']}" if "prompt_type" in fields else None,
        f"- task_type: {fields['task_type']}" if "task_type" in fields else None,
        f"- domain: {fields['domain']}" if "domain" in fields else None,
        f"- language: {fields['language']}" if "language" in fields else None,
        f"- model_name: {fields['model_name']}" if "model_name" in fields else None,
    ]
    meta_block = "\n".join(line for line in meta_lines if line)

    user_content = f"""You are evaluating whether an LLM response contains a factual hallucination relative to the user prompt.

Label semantics:
- hallucination_label 0: response is factually supported / not a hallucination
- hallucination_label 1: response contains a factual error, fabrication, or unsupported claim

Metadata:
{meta_block if meta_block else "(none)"}

User prompt:
\"\"\"
{fields["prompt_text"]}
\"\"\"

Model response:
\"\"\"
{fields["response_text"]}
\"\"\"

Respond with STRICTLY a single JSON object (no markdown, no extra text) with keys:
- "hallucination_label": integer 0 or 1
- "confidence": float between 0 and 1 (your confidence in the label)
- "rationale": brief string explaining the decision
"""

    return [
        {
            "role": "system",
            "content": (
                "You are a careful hallucination detection judge. "
                "Reply with exactly one JSON object and no other text before or after it. "
                "Do not use markdown fences or reasoning prose."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def fetch_judge_entry(sample: Mapping[str, Any]) -> dict[str, Any]:
    messages = build_text_only_judge_messages(sample)
    judge_out = call_nim_judge(messages, model=NIM_JUDGE_MODEL)
    return {
        "hallucination_label": int(judge_out["hallucination_label"]),
        "judge_confidence": float(judge_out["confidence"]),
        "rationale": judge_out["rationale"],
        "nim_model": NIM_JUDGE_MODEL,
    }


def main() -> int:
    if not DATASET_PATH.is_file():
        print(f"Dataset not found: {DATASET_PATH}", file=sys.stderr)
        return 1

    df = pd.read_csv(DATASET_PATH)
    if "record_id" not in df.columns:
        print("Dataset must include record_id column.", file=sys.stderr)
        return 1

    cache = load_cache(OUTPUT_PATH)
    n_rows = len(df)
    n_cached = len(cache)
    to_fetch = sum(1 for _, row in df.iterrows() if str(row["record_id"]) not in cache)

    print(f"Dataset: {DATASET_PATH.name} ({n_rows} rows)")
    print(f"Output: {OUTPUT_PATH.name} ({n_cached} entries on disk)")
    print(f"NIM model: {NIM_JUDGE_MODEL}")
    print(f"Calling NIM for {to_fetch} row(s)...")

    fetched = 0
    limiter = NimRateLimiter()
    for _, row in df.iterrows():
        rid = str(row["record_id"])
        if rid in cache:
            continue
        sample = row_sample(row)
        cache[rid] = fetch_judge_entry(sample)
        fetched += 1
        save_cache(OUTPUT_PATH, cache)
        print(f"  [{fetched}/{to_fetch}] {rid}")
        limiter.after_request()

    save_cache(OUTPUT_PATH, cache)
    print(f"Done. {len(cache)} entries in {OUTPUT_PATH.name} (fetched {fetched} this run).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())