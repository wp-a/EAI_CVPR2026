#!/usr/bin/env python3
"""Whitespace-only postprocess for VirtualHome TM PDDL outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from vtm_schema_utils import (
    action_name,
    decode_llm_output,
    extract_action_blocks,
    first_balanced_expr_after,
    prompt_action_names,
    read_json,
    write_json,
)


def normalize_pddl_expr(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def render_compact_block(block: str) -> str:
    name = action_name(block)
    parameters = normalize_pddl_expr(first_balanced_expr_after(block, ":parameters"))
    precondition = normalize_pddl_expr(first_balanced_expr_after(block, ":precondition"))
    effect = normalize_pddl_expr(first_balanced_expr_after(block, ":effect"))
    return (
        f"(:action {name}\n"
        f"  :parameters {parameters}\n"
        f"  :precondition {precondition}\n"
        f"  :effect {effect}\n"
        f")"
    )


def postprocess_output_text(text: str) -> str:
    blocks = extract_action_blocks(text)
    if not blocks:
        return normalize_pddl_expr(text)
    return "\n".join(render_compact_block(block) for block in blocks)


def row_output(row: dict[str, Any]) -> str:
    return decode_llm_output(row["llm_output"])


def normalized_for_compare(text: str) -> str:
    return normalize_pddl_expr(text)


def validate_rows(prompts: list[dict[str, str]] | None, rows: list[dict[str, str]]) -> dict[str, int]:
    checks = {
        "rows": len(rows),
        "bad_json": 0,
        "bad_balance": 0,
        "bad_action_order": 0,
        "contains_hold_singular": 0,
    }
    for index, row in enumerate(rows):
        try:
            output_text = row_output(row)
        except Exception:
            checks["bad_json"] += 1
            continue
        if output_text.count("(") != output_text.count(")"):
            checks["bad_balance"] += 1
        if "hold_rh" in output_text or "hold_lh" in output_text:
            checks["contains_hold_singular"] += 1
        if prompts is not None:
            actual = [action_name(block) for block in extract_action_blocks(output_text)]
            expected = prompt_action_names(prompts[index]["llm_prompt"])
            if actual != expected:
                checks["bad_action_order"] += 1
    return checks


def compare_rows(rows: list[dict[str, str]], compare_rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) != len(compare_rows):
        raise ValueError(f"Compare row count mismatch: {len(rows)} != {len(compare_rows)}")
    return {
        "rows": len(rows),
        "identifier_order_same": [row["identifier"] for row in rows]
        == [row["identifier"] for row in compare_rows],
        "row_json_exact": sum(row == other for row, other in zip(rows, compare_rows)),
        "row_output_exact": sum(row_output(row) == row_output(other) for row, other in zip(rows, compare_rows)),
        "row_output_pddl_whitespace_normalized_exact": sum(
            normalized_for_compare(row_output(row)) == normalized_for_compare(row_output(other))
            for row, other in zip(rows, compare_rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize PDDL whitespace in VirtualHome TM outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--compare-output", type=Path, default=None)
    args = parser.parse_args()

    rows = read_json(args.input)
    output_rows: list[dict[str, str]] = []
    changed_rows = 0
    for row in rows:
        output_text = row_output(row)
        normalized_text = postprocess_output_text(output_text)
        if normalized_text != output_text:
            changed_rows += 1
        output_rows.append(
            {
                "identifier": row["identifier"],
                "llm_output": json.dumps({"output": normalized_text}, ensure_ascii=False),
            }
        )

    write_json(args.output, output_rows)

    prompts = read_json(args.prompts) if args.prompts else None
    report: dict[str, Any] = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(output_rows),
        "changed_rows": changed_rows,
        "format_checks": validate_rows(prompts, output_rows),
        "policy": "PDDL whitespace only: collapse runs of whitespace and remove spaces adjacent to parentheses.",
    }
    if args.compare_output:
        report["comparison"] = compare_rows(output_rows, read_json(args.compare_output))
    write_json(args.report or args.output.with_name("postprocess_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
