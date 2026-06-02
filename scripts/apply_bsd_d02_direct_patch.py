#!/usr/bin/env python3
"""Conservatively complete direct BEHAVIOR BSD goals from prompts.

This postprocess is prompt driven: it reads each task's compact BSD prompt,
extracts simple direct goal atoms from the goal-state section, and appends
missing atoms only for short direct-goal outputs. It does not expand
forall/exists/forpairs/forn goals and does not special-case identifiers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


SIMPLE_GOAL_RE = re.compile(r"^(?:not\s+)?[A-Za-z_][A-Za-z0-9_]*\([^()]+\)$")
COMPLEX_GOAL_MARKERS = ("forall", "exists", "forpairs", "forn", "fornucleated")


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_goal_section(prompt: str) -> str:
    parts = [part.strip() for part in prompt.split("--")]
    if len(parts) < 4:
        return ""
    return parts[3]


def simple_direct_goals(prompt: str) -> list[str]:
    section = compact_goal_section(prompt)
    lowered = section.lower()
    if any(marker in lowered for marker in COMPLEX_GOAL_MARKERS):
        return []
    goals: list[str] = []
    for line in section.splitlines():
        atom = line.strip()
        if SIMPLE_GOAL_RE.match(atom):
            goals.append(atom)
    return goals


def prompt_goals_by_id(prompt_path: Path) -> dict[str, list[str]]:
    rows = read_json(prompt_path)
    return {row["identifier"]: simple_direct_goals(row.get("llm_prompt", "")) for row in rows}


def atomic_write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bsd", required=True, help="Source behavior_subgoal_decomposition_outputs.json")
    parser.add_argument("--output-bsd", required=True, help="Patched behavior_subgoal_decomposition_outputs.json")
    parser.add_argument(
        "--prompt-file",
        default="llm_prompts_axis06b_behavior_compact/behavior_subgoal_decomposition_prompts.json",
        help="Compact BSD prompt file used to derive direct goals.",
    )
    parser.add_argument("--copy-from-dir", default="", help="Optional directory containing the other submission JSON files.")
    parser.add_argument("--output-dir", default="", help="Optional full submission output directory.")
    parser.add_argument("--report", default="", help="Optional JSON report path.")
    parser.add_argument(
        "--max-direct-goals",
        type=int,
        default=4,
        help="Only complete rows whose prompt has at most this many direct goals.",
    )
    parser.add_argument(
        "--max-current-output-atoms",
        type=int,
        default=5,
        help="Only complete rows whose model output is this short or shorter.",
    )
    args = parser.parse_args()

    rows = read_json(Path(args.input_bsd))
    goals_by_id = prompt_goals_by_id(Path(args.prompt_file))
    changed: list[str] = []
    skipped: dict[str, int] = {
        "no_prompt_direct_goals": 0,
        "too_many_direct_goals": 0,
        "output_not_list": 0,
        "output_too_long": 0,
        "already_complete": 0,
    }
    added_atoms = 0
    for row in rows:
        identifier = row.get("identifier", "")
        goals = goals_by_id.get(identifier, [])
        if not goals:
            skipped["no_prompt_direct_goals"] += 1
            continue
        parsed = json.loads(row.get("llm_output", ""))
        output = parsed.get("output")
        if not isinstance(output, list):
            skipped["output_not_list"] += 1
            continue
        if len(goals) > args.max_direct_goals:
            skipped["too_many_direct_goals"] += 1
            continue
        if len(output) > args.max_current_output_atoms:
            skipped["output_too_long"] += 1
            continue
        additions = [goal for goal in goals if goal not in output]
        if not additions:
            skipped["already_complete"] += 1
            continue
        before = list(output)
        for atom in additions:
            output.append(atom)
            added_atoms += 1
        if output != before:
            changed.append(identifier)
        row["llm_output"] = json.dumps({"output": output}, ensure_ascii=False, separators=(",", ":"))

    output_bsd = Path(args.output_bsd)
    atomic_write_json(output_bsd, rows)

    if args.copy_from_dir and args.output_dir:
        copy_from = Path(args.copy_from_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(copy_from.glob("*_outputs.json")):
            dst = output_dir / src.name
            if src.name == "behavior_subgoal_decomposition_outputs.json":
                continue
            shutil.copyfile(src, dst)
        shutil.copyfile(output_bsd, output_dir / "behavior_subgoal_decomposition_outputs.json")

    report = {
        "policy": "prompt-derived short direct-goal completion; no identifier-specific patches",
        "input_bsd": args.input_bsd,
        "prompt_file": args.prompt_file,
        "output_bsd": str(output_bsd),
        "max_direct_goals": args.max_direct_goals,
        "max_current_output_atoms": args.max_current_output_atoms,
        "changed_rows": len(changed),
        "added_atoms": added_atoms,
        "changed_identifiers": changed,
        "skipped": skipped,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
