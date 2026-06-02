#!/usr/bin/env python3
"""Build conservative VirtualHome SD ablation candidates.

The candidates are meant for EvalAI ablations: each variant changes a narrow
class of postprocessing edits so score deltas can be attributed.
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from vsd_common import (
    atom_keys_in_output,
    atoms_in_expr,
    hard_issue_count,
    normalize_candidate,
    parse_prompt_item,
    stable_json_output,
)


ROOM_RE = re.compile(r"\b(?:bathroom|bedroom|kitchen|dining_room|living_room|home_office)\.\d+\b")


def read_rows(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_outputs(
    rows: list[dict[str, str]],
    prompt_info: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    parsed = {}
    for row in rows:
        identifier = row["identifier"]
        try:
            parsed[identifier] = json.loads(row["llm_output"])
        except Exception:
            if prompt_info and identifier in prompt_info:
                parsed[identifier], _report = normalize_candidate(prompt_info[identifier], row.get("llm_output", ""))
            else:
                parsed[identifier] = {
                    "necessity_to_use_action": "no",
                    "actions_to_include": [],
                    "output": [],
                }
    return parsed


def action_names(items: list[str]) -> list[str]:
    names: list[str] = []
    for item in items:
        for name, _args, _key in atoms_in_expr(item):
            if name not in names:
                names.append(name)
    return names


def has_atom(output: list[str], item: str) -> bool:
    wanted = {key for _name, _args, key in atoms_in_expr(item)}
    if not wanted:
        return item in output
    return wanted.issubset(atom_keys_in_output(output))


def insert_like_baseline(output: list[str], item: str, baseline_output: list[str]) -> None:
    if has_atom(output, item):
        return
    baseline_idx = baseline_output.index(item) if item in baseline_output else len(baseline_output)
    insert_at = min(baseline_idx, len(output))
    output.insert(insert_at, item)


def remove_item_atoms(output: list[str], item: str) -> list[str]:
    remove_keys = {key for _name, _args, key in atoms_in_expr(item)}
    if not remove_keys:
        return [x for x in output if x != item]
    kept = []
    for existing in output:
        existing_keys = {key for _name, _args, key in atoms_in_expr(existing)}
        if existing == item or (existing_keys and existing_keys.issubset(remove_keys)):
            continue
        kept.append(existing)
    return kept


def restore_original_items(
    original_output: list[str],
    candidate_output: list[str],
    restore_items: set[str],
) -> list[str]:
    result = list(candidate_output)
    for idx, item in enumerate(original_output):
        if item not in restore_items or has_atom(result, item):
            continue
        insert_at = min(idx, len(result))
        result.insert(insert_at, item)
    return result


def copy_obj(obj: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(obj)


def write_variant(
    name: str,
    out_dir: Path,
    identifiers: list[str],
    objects: dict[str, dict[str, Any]],
    prompt_info: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    hard_total = 0
    hard_rows = 0
    for identifier in identifiers:
        obj = objects[identifier]
        info = prompt_info[identifier]
        hard, _issues = hard_issue_count(info, obj)
        hard_total += hard
        hard_rows += int(hard > 0)
        rows.append({"identifier": identifier, "llm_output": stable_json_output(obj)})

    target_dir = out_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "virtualhome_subgoal_decomposition_outputs.json"
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "name": name,
        "path": str(target),
        "hard_total": hard_total,
        "hard_rows": hard_rows,
    }


def diff_count(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], identifiers: list[str]) -> int:
    return sum(a[i] != b[i] for i in identifiers)


def build_selective_refiner(
    baseline: dict[str, dict[str, Any]],
    refiner_rows: list[dict[str, str]],
    prompt_info: dict[str, Any],
    identifiers: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    refiner_by_id = {row["identifier"]: row.get("llm_output", "") for row in refiner_rows}
    chosen = {identifier: copy_obj(baseline[identifier]) for identifier in identifiers}
    accepted = []
    rejected = 0

    for identifier in identifiers:
        raw = refiner_by_id.get(identifier, "")
        if not raw:
            continue
        info = prompt_info[identifier]
        base = baseline[identifier]
        candidate, candidate_report = normalize_candidate(info, raw)
        if not candidate_report["parse_ok"]:
            rejected += 1
            continue

        base_hard, _ = hard_issue_count(info, base)
        candidate_hard, _ = hard_issue_count(info, candidate)
        if candidate_hard > base_hard:
            rejected += 1
            continue

        base_output = base.get("output", [])
        candidate_output = candidate.get("output", [])
        base_atoms = atom_keys_in_output(base_output)
        candidate_atoms = atom_keys_in_output(candidate_output)
        removed_atoms = base_atoms - candidate_atoms
        added_atoms = candidate_atoms - base_atoms
        length_delta = len(candidate_output) - len(base_output)

        if removed_atoms:
            rejected += 1
            continue
        if length_delta < 0 or length_delta > 3:
            rejected += 1
            continue
        if len(added_atoms) > 3:
            rejected += 1
            continue
        if base.get("necessity_to_use_action") != candidate.get("necessity_to_use_action"):
            rejected += 1
            continue
        if set(base.get("actions_to_include", [])) - set(candidate.get("actions_to_include", [])):
            rejected += 1
            continue

        chosen[identifier] = candidate
        accepted.append(
            {
                "identifier": identifier,
                "base_hard": base_hard,
                "candidate_hard": candidate_hard,
                "added_atoms": sorted(added_atoms),
                "length_delta": length_delta,
            }
        )

    return chosen, {"accepted": len(accepted), "rejected": rejected, "accepted_rows": accepted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--refiner")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    prompt_info = {row["identifier"]: parse_prompt_item(row) for row in prompts}
    identifiers = [row["identifier"] for row in prompts]

    original_rows = read_rows(Path(args.original))
    baseline_rows = read_rows(Path(args.baseline))
    original = parse_outputs(original_rows, prompt_info)
    baseline = parse_outputs(baseline_rows, prompt_info)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    report_by_id = {row["identifier"]: row for row in report["rows"]}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, dict[str, dict[str, Any]]] = {}

    goals_only = {identifier: copy_obj(original[identifier]) for identifier in identifiers}
    goals_actions_safe = {identifier: copy_obj(original[identifier]) for identifier in identifiers}
    baseline_restore_grab = {identifier: copy_obj(baseline[identifier]) for identifier in identifiers}
    baseline_restore_extra_actions = {identifier: copy_obj(baseline[identifier]) for identifier in identifiers}
    baseline_no_added_actions = {identifier: copy_obj(baseline[identifier]) for identifier in identifiers}
    baseline_no_weird_added_actions = {identifier: copy_obj(baseline[identifier]) for identifier in identifiers}

    stats = {
        "skipped_weird_required_actions": [],
        "restored_grab_items": 0,
        "restored_extra_action_items": 0,
        "removed_added_action_items": 0,
        "removed_weird_added_action_items": 0,
    }

    for identifier in identifiers:
        row_report = report_by_id[identifier]
        original_obj = original[identifier]
        baseline_obj = baseline[identifier]
        original_output = list(original_obj.get("output", []))
        baseline_output = list(baseline_obj.get("output", []))

        for goal in row_report.get("missing_goals_added") or []:
            insert_like_baseline(goals_only[identifier]["output"], goal, baseline_output)
            insert_like_baseline(goals_actions_safe[identifier]["output"], goal, baseline_output)

        for action in row_report.get("required_actions_added") or []:
            if ROOM_RE.search(action) or "()" not in action and re.search(r"\(\s*(?:,\s*)?\)", action):
                stats["skipped_weird_required_actions"].append({"identifier": identifier, "action": action})
                continue
            insert_like_baseline(goals_actions_safe[identifier]["output"], action, baseline_output)
            for name in action_names([action]):
                current = goals_actions_safe[identifier].setdefault("actions_to_include", [])
                if name not in current:
                    current.append(name)

        extra_items = set()
        grab_items = set()
        for removed in row_report.get("extra_actions_removed") or []:
            for item in original_output:
                if removed in {key for _name, _args, key in atoms_in_expr(item)}:
                    extra_items.add(item)
                    if removed.startswith("GRAB("):
                        grab_items.add(item)

        before = list(baseline_restore_extra_actions[identifier]["output"])
        baseline_restore_extra_actions[identifier]["output"] = restore_original_items(
            original_output,
            before,
            extra_items,
        )
        stats["restored_extra_action_items"] += len(baseline_restore_extra_actions[identifier]["output"]) - len(before)

        before = list(baseline_restore_grab[identifier]["output"])
        baseline_restore_grab[identifier]["output"] = restore_original_items(original_output, before, grab_items)
        stats["restored_grab_items"] += len(baseline_restore_grab[identifier]["output"]) - len(before)

        for action in row_report.get("required_actions_added") or []:
            before_len = len(baseline_no_added_actions[identifier]["output"])
            baseline_no_added_actions[identifier]["output"] = remove_item_atoms(
                baseline_no_added_actions[identifier]["output"],
                action,
            )
            stats["removed_added_action_items"] += before_len - len(baseline_no_added_actions[identifier]["output"])
            for name in action_names([action]):
                original_actions = set(original_obj.get("actions_to_include", []))
                if name not in original_actions:
                    baseline_no_added_actions[identifier]["actions_to_include"] = [
                        x for x in baseline_no_added_actions[identifier].get("actions_to_include", []) if x != name
                    ]

            if ROOM_RE.search(action):
                before_len = len(baseline_no_weird_added_actions[identifier]["output"])
                baseline_no_weird_added_actions[identifier]["output"] = remove_item_atoms(
                    baseline_no_weird_added_actions[identifier]["output"],
                    action,
                )
                stats["removed_weird_added_action_items"] += before_len - len(
                    baseline_no_weird_added_actions[identifier]["output"]
                )
                for name in action_names([action]):
                    original_actions = set(original_obj.get("actions_to_include", []))
                    if name not in original_actions:
                        baseline_no_weird_added_actions[identifier]["actions_to_include"] = [
                            x
                            for x in baseline_no_weird_added_actions[identifier].get("actions_to_include", [])
                            if x != name
                        ]

    variants["goals_only_from_original"] = goals_only
    variants["goals_safe_actions_no_delete"] = goals_actions_safe
    variants["baseline_restore_grab"] = baseline_restore_grab
    variants["baseline_restore_extra_actions"] = baseline_restore_extra_actions
    variants["baseline_no_added_actions"] = baseline_no_added_actions
    variants["baseline_no_weird_added_actions"] = baseline_no_weird_added_actions

    selective_report = None
    if args.refiner:
        selective, selective_report = build_selective_refiner(
            baseline,
            read_rows(Path(args.refiner)),
            prompt_info,
            identifiers,
        )
        variants["baseline_selective_refiner_superset"] = selective

    summaries = []
    for name, objects in variants.items():
        summary = write_variant(name, out_dir, identifiers, objects, prompt_info)
        summary["diff_from_original"] = diff_count(original, objects, identifiers)
        summary["diff_from_baseline"] = diff_count(baseline, objects, identifiers)
        summaries.append(summary)

    manifest = {
        "inputs": {
            "prompts": args.prompts,
            "original": args.original,
            "baseline": args.baseline,
            "report": args.report,
            "refiner": args.refiner,
        },
        "stats": stats,
        "selective_refiner": selective_report,
        "variants": summaries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
