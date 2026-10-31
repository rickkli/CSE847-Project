# Modified by Cursor: tiered hallucination classifier with NVIDIA NIM LLM-as-judge fallback.

"""
Escalate low-confidence logistic-regression predictions to an LLM judge via NVIDIA NIM.

Requires: requests, python-dotenv; set NVIDIA_API_KEY in the environment or .env.
Optional: NIM_JUDGE_MODEL (default nvidia/nemotron-3-ultra-550b-a55b).
Uses OpenAI-compatible response_format json_schema on the integrate REST API.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_JUDGE_MODEL = os.getenv("NIM_JUDGE_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

JUDGE_SAMPLE_KEYS = (
    "record_id",
    "prompt_text",
    "response_text",
    "prompt_type",
    "task_type",
    "domain",
    "language",
    "model_name",
)

FORBIDDEN_JUDGE_KEYS = frozenset(
    {
        "hallucination_label",
        "hallucination_type",
        "hallucination_span",
        "correct_information",
        "correction_text",
        "notes",
        "annotation_confidence",
    }
)

MAX_FIELD_CHARS = 8000

JUDGE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "hallucination_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "hallucination_label": {"type": "integer", "enum": [0, 1]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
            "required": ["hallucination_label", "confidence", "rationale"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class TieredClassificationResult:
    final_label: int
    decision_source: str
    lr_pred: int
    lr_proba: list[float]
    lr_confidence: float
    confidence_threshold: float
    escalated: bool
    judge_confidence: float | None = None
    judge_rationale: str | None = None
    record_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_proba(lr_proba: Sequence[float]) -> list[float]:
    proba = [float(p) for p in lr_proba]
    if len(proba) != 2:
        raise ValueError("lr_proba must have length 2: [P(label=0), P(label=1)]")
    if any(p < 0 or p > 1 for p in proba):
        raise ValueError("lr_proba values must be in [0, 1]")
    return proba


def lr_confidence(lr_proba: Sequence[float], lr_pred: int | None = None) -> float:
    proba = _validate_proba(lr_proba)
    conf = max(proba)
    if lr_pred is not None and lr_pred in (0, 1):
        if int(lr_pred) != int(proba.index(max(proba))):
            conf = proba[int(lr_pred)]
    return conf


def should_escalate(
    lr_proba: Sequence[float],
    lr_pred: int,
    threshold: float,
) -> bool:
    if lr_pred not in (0, 1):
        raise ValueError("lr_pred must be 0 or 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    return lr_confidence(lr_proba, lr_pred) < threshold


def _truncate(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _sanitize_sample(sample: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in JUDGE_SAMPLE_KEYS:
        if key in sample and sample[key] is not None and str(sample[key]).strip():
            out[key] = _truncate(str(sample[key]))
    if "prompt_text" not in out or "response_text" not in out:
        raise ValueError("sample must include prompt_text and response_text")
    leaked = FORBIDDEN_JUDGE_KEYS & set(sample.keys())
    if leaked:
        pass
    return out


def build_judge_messages(
    sample: Mapping[str, Any],
    lr_pred: int,
    lr_proba: Sequence[float],
) -> list[dict[str, str]]:
    fields = _sanitize_sample(sample)
    proba = _validate_proba(lr_proba)
    conf = lr_confidence(proba, lr_pred)

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

A separate ML model produced a tentative prediction (may be wrong):
- ml_hallucination_label: {lr_pred}
- ml_confidence: {conf:.4f}
- ml_proba_not_hallucination: {proba[0]:.4f}
- ml_proba_hallucination: {proba[1]:.4f}

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


def _message_text(message: Mapping[str, Any]) -> str:
    """Combine visible content and Nemotron reasoning fields for JSON extraction."""
    chunks: list[str] = []
    for key in ("content", "reasoning_content"):
        value = message.get(key)
        if value is not None and str(value).strip():
            chunks.append(str(value).strip())
    return "\n".join(chunks).strip()


def _extract_balanced_json_object(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def extract_json_from_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("No JSON object found in the response string.")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    for start in (idx for idx, ch in enumerate(text) if ch == "{"):
        candidate = _extract_balanced_json_object(text, start)
        if not candidate:
            continue
        if "hallucination_label" in candidate:
            return candidate

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No JSON object found in the response string.")


def parse_judge_response(text: str) -> dict[str, Any]:
    json_string = extract_json_from_text(text.strip())
    data = json.loads(json_string)
    label = int(data["hallucination_label"])
    if label not in (0, 1):
        raise ValueError("hallucination_label must be 0 or 1")
    confidence = float(data["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    rationale = str(data.get("rationale", "")).strip()
    return {
        "hallucination_label": label,
        "confidence": confidence,
        "rationale": rationale,
    }


def _nim_payload(model_name: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> dict[str, Any]:
    """Build NIM chat/completions body. Uses json_schema structured output (integrate REST API)."""
    return {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": JUDGE_RESPONSE_FORMAT,
    }


def call_nim_judge(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
) -> dict[str, Any]:
    key = api_key or NVIDIA_API_KEY
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not found in environment.")

    model_name = model or DEFAULT_JUDGE_MODEL
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    attempt_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            payload = _nim_payload(model_name, attempt_messages, temperature, max_tokens)
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if not response.ok:
                detail = response.text[:500]
                raise requests.HTTPError(
                    f"{response.status_code} for model {model_name!r}: {detail}",
                    response=response,
                )
            result = response.json()
            message = result.get("choices", [{}])[0].get("message", {})
            content = _message_text(message)
            if not content:
                raise ValueError("Model returned an empty response (no content or reasoning_content).")
            return parse_judge_response(content)
        except (json.JSONDecodeError, ValueError, requests.RequestException) as exc:
            last_error = exc
            attempt_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was not valid JSON. "
                        'Reply with ONLY this object shape: '
                        '{"hallucination_label": 0, "confidence": 0.0, "rationale": "..."}'
                    ),
                }
            ]
            if attempt + 1 < max_retries:
                time.sleep(retry_delay_sec * (attempt + 1))
            continue
    assert last_error is not None
    raise last_error


def classify_tiered(
    sample: Mapping[str, Any],
    lr_pred: int,
    lr_proba: Sequence[float],
    confidence_threshold: float = 0.7,
    *,
    model: str | None = None,
    api_key: str | None = None,
    skip_network: bool = False,
) -> TieredClassificationResult:
    proba = _validate_proba(lr_proba)
    conf = lr_confidence(proba, lr_pred)
    record_id = sample.get("record_id")
    record_id_str = str(record_id) if record_id is not None else None

    base = TieredClassificationResult(
        final_label=int(lr_pred),
        decision_source="lr",
        lr_pred=int(lr_pred),
        lr_proba=proba,
        lr_confidence=conf,
        confidence_threshold=float(confidence_threshold),
        escalated=False,
        record_id=record_id_str,
    )

    if not should_escalate(proba, int(lr_pred), confidence_threshold):
        return base

    if skip_network:
        raise RuntimeError(
            "Escalation required but skip_network=True (no NVIDIA NIM call made)."
        )

    messages = build_judge_messages(sample, int(lr_pred), proba)
    judge = call_nim_judge(messages, model=model, api_key=api_key)

    base.escalated = True
    base.decision_source = "llm_judge"
    base.final_label = int(judge["hallucination_label"])
    base.judge_confidence = float(judge["confidence"])
    base.judge_rationale = judge["rationale"]
    return base


def _parse_lr_proba_arg(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("lr-proba must be two comma-separated floats, e.g. 0.4,0.6")
    return [float(parts[0]), float(parts[1])]


def _load_json_records(path: str | None) -> list[dict[str, Any]]:
    if path and path != "-":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("JSON input must be an object or array of objects")


def _run_self_checks() -> None:
    proba = [0.4, 0.6]
    pred = 1
    conf = lr_confidence(proba, pred)
    assert abs(conf - 0.6) < 1e-9
    assert should_escalate(proba, pred, 0.7) is True
    assert should_escalate([0.65, 0.35], 0, 0.7) is True
    assert should_escalate([0.75, 0.25], 0, 0.7) is False
    assert should_escalate([0.7, 0.3], 0, 0.7) is False

    sample = {
        "prompt_text": "Who invented the telephone?",
        "response_text": "Nikola Tesla invented it.",
        "correct_information": "Alexander Graham Bell",
    }
    messages = build_judge_messages(sample, lr_pred=1, lr_proba=[0.4, 0.6])
    blob = json.dumps(messages)
    for forbidden in FORBIDDEN_JUDGE_KEYS:
        assert forbidden not in blob or forbidden == "hallucination_label"
    assert "correct_information" not in blob

    prose = (
        "Analysis without braces at first. "
        '{"hallucination_label": 1, "confidence": 0.9, "rationale": "wrong fact"}'
    )
    parsed = parse_judge_response(prose)
    assert parsed["hallucination_label"] == 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiered hallucination classification with NIM judge.")
    parser.add_argument("--self-check", action="store_true", help="Run local threshold/prompt checks (no network).")
    parser.add_argument("--skip-network", action="store_true", help="Do not call NIM even if escalation is required.")
    parser.add_argument("--input-json", help="Path to JSON file, or '-' for stdin (array or single object).")
    parser.add_argument("--lr-pred", type=int, choices=[0, 1])
    parser.add_argument("--lr-proba", type=_parse_lr_proba_arg)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--prompt-text")
    parser.add_argument("--response-text")
    parser.add_argument("--prompt-type")
    parser.add_argument("--task-type")
    parser.add_argument("--domain")
    parser.add_argument("--language")
    parser.add_argument("--model-name")
    parser.add_argument("--record-id")
    parser.add_argument("--nim-model", default=None, help="Override NIM_JUDGE_MODEL for this run.")
    args = parser.parse_args()

    if args.self_check:
        _run_self_checks()
        print("Self-checks passed.")
        return 0

    results: list[dict[str, Any]] = []

    if args.input_json:
        records = _load_json_records(args.input_json)
        for rec in records:
            if "lr_pred" not in rec or "lr_proba" not in rec:
                raise SystemExit("Each JSON record must include lr_pred and lr_proba.")
            proba = _validate_proba(rec["lr_proba"])
            threshold = float(rec.get("confidence_threshold", args.threshold))
            out = classify_tiered(
                rec,
                int(rec["lr_pred"]),
                proba,
                threshold,
                model=args.nim_model,
                skip_network=args.skip_network,
            )
            results.append(out.to_dict())
    else:
        if args.lr_pred is None or args.lr_proba is None:
            parser.error("Provide --lr-pred and --lr-proba, or use --input-json.")
        if not args.prompt_text or not args.response_text:
            parser.error("Provide --prompt-text and --response-text for single-row mode.")
        sample = {
            "prompt_text": args.prompt_text,
            "response_text": args.response_text,
        }
        for key, arg_val in (
            ("prompt_type", args.prompt_type),
            ("task_type", args.task_type),
            ("domain", args.domain),
            ("language", args.language),
            ("model_name", args.model_name),
            ("record_id", args.record_id),
        ):
            if arg_val is not None:
                sample[key] = arg_val
        out = classify_tiered(
            sample,
            args.lr_pred,
            args.lr_proba,
            args.threshold,
            model=args.nim_model,
            skip_network=args.skip_network,
        )
        results.append(out.to_dict())

    print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
