#!/usr/bin/env python3
"""Strict, conservative postprocess for VirtualHome SD outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vsd_common import parse_prompt_item, stable_json_output, normalize_candidate, hard_issue_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="llm_prompts/virtualhome_subgoal_decomposition_prompts.json")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--split-safe-and", action="store_true")
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    outputs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_by_id = {row["identifier"]: row.get("llm_output", "") for row in outputs}

    rows = []
    report_rows = []
    summary = {
        "total": 0,
        "parse_repaired": 0,
        "changed": 0,
        "missing_goals_added": 0,
        "required_actions_added": 0,
        "invalid_atoms_removed": 0,
        "hard_issues_after": 0,
    }

    for prompt in prompts:
        info = parse_prompt_item(prompt)
        raw = output_by_id.get(info.identifier, "")
        normalized, row_report = normalize_candidate(info, raw, split_and=args.split_safe_and)
        hard_count, hard_issues = hard_issue_count(info, normalized)
        out_text = stable_json_output(normalized)
        rows.append({"identifier": info.identifier, "llm_output": out_text})

        changed = out_text != raw.strip()
        summary["total"] += 1
        summary["parse_repaired"] += 0 if row_report["parse_ok"] else 1
        summary["changed"] += int(changed)
        summary["missing_goals_added"] += len(row_report["missing_goals_added"])
        summary["required_actions_added"] += len(row_report["required_actions_added"])
        summary["invalid_atoms_removed"] += len(row_report["invalid_atoms"])
        summary["hard_issues_after"] += hard_count
        report_rows.append(
            {
                "identifier": info.identifier,
                "changed": changed,
                "hard_count_after": hard_count,
                "hard_issues_after": hard_issues,
                **row_report,
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps({"summary": summary, "rows": report_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

