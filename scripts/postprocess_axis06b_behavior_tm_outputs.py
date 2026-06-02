#!/usr/bin/env python3
"""Postprocess AxisTilted2 BEHAVIOR transition-modeling outputs.

This script only uses the current run's raw model outputs and the official
prompt order. It normalizes the model text into evaluator-friendly JSON strings
without reading prior submissions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


PROMPT_FILE = "behavior_transition_modeling_prompts.json"
OUTPUT_FILE = "behavior_transition_modeling_outputs.json"
ACTION_RE = re.compile(r"\(:action\s+([^\s\)]+)")
PROMPT_ACTION_RE = re.compile(
    r"\(:action\s+([^\s\)]+)\)?\s*"
    r":parameters\s*\((.*?)\)\s*"
    r":precondition\s*\(\)\s*"
    r":effect\s*\(\)\s*\)",
    re.S,
)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"```(?:json|pddl|python)?", "", text, flags=re.I)
    return text.replace("```", "").strip()


def balanced_json_slice(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    quote = ""
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char in {'"', "'"}:
            if not in_string:
                in_string = True
                quote = char
            elif quote == char:
                in_string = False
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_jsonish(text: str) -> Any:
    candidates = [text]
    sliced = balanced_json_slice(text)
    if sliced:
        candidates.insert(0, sliced)
    match = re.search(r"\{.*\"output\".*\}", text, flags=re.S)
    if match:
        candidates.insert(0, match.group(0))

    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def find_balanced_end(text: str, start_index: int) -> int | None:
    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def extract_action_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    cursor = 0
    while True:
        start = text.find("(:action", cursor)
        if start < 0:
            return blocks
        end = find_balanced_end(text, start)
        if end is None:
            cursor = start + len("(:action")
            continue
        blocks.append(text[start:end].strip())
        cursor = end


def normalize_output_text(raw: str | None) -> tuple[str, bool]:
    cleaned = clean_text(raw)
    parsed = parse_jsonish(cleaned)
    parsed_json = False
    if isinstance(parsed, dict) and isinstance(parsed.get("output"), str):
        cleaned = clean_text(parsed["output"])
        parsed_json = True
    elif isinstance(parsed, dict) and isinstance(parsed.get("output"), list):
        cleaned = "\n\n".join(str(item) for item in parsed["output"])
        parsed_json = True

    blocks = extract_action_blocks(cleaned)
    if blocks:
        return "\n\n".join(blocks), parsed_json
    return cleaned, parsed_json


def final_problem_part(prompt: str) -> str:
    return prompt.rsplit("Output:", 1)[0].rsplit("Input:", 1)[-1]


def prompt_action_names(prompt: str) -> list[str]:
    return [match.group(1) for match in PROMPT_ACTION_RE.finditer(final_problem_part(prompt))]


def output_action_names(output: str) -> list[str]:
    return ACTION_RE.findall(output)


def validate_output(output: str) -> dict[str, int]:
    return {
        "empty": int(not output.strip()),
        "balanced": int(output.count("(") == output.count(")") and bool(output.strip())),
        "action_blocks": len(extract_action_blocks(output)),
        "think_tags": int("<think>" in output or "</think>" in output),
        "markdown_fences": int("```" in output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output-dir", required=True)
    parser.add_argument("--prompt-dir", default="llm_prompts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    prompt_rows = load_rows(Path(args.prompt_dir) / PROMPT_FILE)
    if args.limit is not None:
        prompt_rows = prompt_rows[: args.limit]
    raw_rows = load_rows(Path(args.raw_output_dir) / OUTPUT_FILE)
    raw_by_id = {row["identifier"]: row.get("llm_output", "") for row in raw_rows}

    stats = {
        "task": "behavior_transition_modeling",
        "total": len(prompt_rows),
        "raw_rows": len(raw_rows),
        "raw_missing": 0,
        "json_parseable_raw": 0,
        "empty_outputs": 0,
        "balanced_outputs": 0,
        "outputs_with_action_blocks": 0,
        "think_tag_outputs": 0,
        "markdown_fence_outputs": 0,
        "prompt_action_sequence_mismatches": 0,
    }
    mismatch_examples: list[dict[str, Any]] = []
    out_rows: list[dict[str, str]] = []

    for prompt_row in prompt_rows:
        identifier = prompt_row["identifier"]
        raw = raw_by_id.get(identifier, "")
        if not raw:
            stats["raw_missing"] += 1
        output, parsed_json = normalize_output_text(raw)
        stats["json_parseable_raw"] += int(parsed_json)

        checks = validate_output(output)
        stats["empty_outputs"] += checks["empty"]
        stats["balanced_outputs"] += checks["balanced"]
        stats["outputs_with_action_blocks"] += int(checks["action_blocks"] > 0)
        stats["think_tag_outputs"] += checks["think_tags"]
        stats["markdown_fence_outputs"] += checks["markdown_fences"]

        prompt_actions = prompt_action_names(prompt_row["llm_prompt"])
        output_actions = output_action_names(output)
        if prompt_actions and output_actions and prompt_actions != output_actions:
            stats["prompt_action_sequence_mismatches"] += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append(
                    {
                        "identifier": identifier,
                        "prompt_actions": prompt_actions,
                        "output_actions": output_actions,
                    }
                )

        out_rows.append(
            {
                "identifier": identifier,
                "llm_output": json.dumps({"output": output}, ensure_ascii=False),
            }
        )

    output_path = Path(args.output_dir) / OUTPUT_FILE
    atomic_write_json(output_path, out_rows)

    report = {
        **stats,
        "output_path": str(output_path),
        "mismatch_examples": mismatch_examples,
    }
    report_path = Path(args.report_path) if args.report_path else Path(args.output_dir) / "behavior_transition_modeling_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
