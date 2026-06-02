#!/usr/bin/env python3
"""Validate and summarize VirtualHome SD controlled experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {"necessity_to_use_action", "actions_to_include", "output"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def parse_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    rows = load_json(path)
    parsed_by_id: dict[str, dict[str, Any]] = {}
    stats = {
        "path": str(path),
        "exists": path.exists(),
        "md5": md5(path),
        "rows": len(rows) if isinstance(rows, list) else 0,
        "unique_identifiers": 0,
        "parse_ok": 0,
        "missing_required_keys": 0,
        "bad_output_type": 0,
        "think_tag_rows": 0,
        "avg_output_len": None,
    }
    if not isinstance(rows, list):
        return [], parsed_by_id, stats

    output_lens: list[int] = []
    identifiers = []
    for row in rows:
        identifier = row.get("identifier") if isinstance(row, dict) else None
        if identifier:
            identifiers.append(identifier)
        raw = row.get("llm_output", "") if isinstance(row, dict) else ""
        if "<think>" in raw.lower() or "</think>" in raw.lower():
            stats["think_tag_rows"] += 1
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        stats["parse_ok"] += 1
        if REQUIRED_KEYS - set(obj):
            stats["missing_required_keys"] += 1
        if not isinstance(obj.get("output"), list):
            stats["bad_output_type"] += 1
        else:
            output_lens.append(len(obj["output"]))
        if identifier:
            parsed_by_id[identifier] = obj

    stats["unique_identifiers"] = len(set(identifiers))
    if output_lens:
        stats["avg_output_len"] = sum(output_lens) / len(output_lens)
    return rows, parsed_by_id, stats


def semantic_diff_count(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> int:
    ids = set(a) & set(b)
    return sum(a[identifier] != b[identifier] for identifier in ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--output", required=True)
    parser.add_argument("--file", action="append", default=[], help="name=path")
    parser.add_argument("--postprocess-report", action="append", default=[], help="name=path")
    parser.add_argument("--ablation-manifest", action="append", default=[], help="name=path")
    args = parser.parse_args()

    prompts = load_json(Path(args.prompts))
    prompt_ids = [row["identifier"] for row in prompts]
    prompt_id_set = set(prompt_ids)

    reference_by_id = None
    reference_stats = None
    if args.reference:
        _rows, reference_by_id, reference_stats = parse_rows(Path(args.reference))

    files: dict[str, Any] = {}
    for item in args.file:
        name, raw_path = item.split("=", 1)
        rows, parsed_by_id, stats = parse_rows(Path(raw_path))
        row_ids = [row.get("identifier") for row in rows if isinstance(row, dict)]
        stats["identifiers_match_prompts"] = row_ids == prompt_ids
        stats["missing_prompt_ids"] = sorted(prompt_id_set - set(row_ids))[:20]
        stats["extra_ids"] = sorted(set(row_ids) - prompt_id_set)[:20]
        if reference_by_id is not None:
            stats["diff_from_reference"] = semantic_diff_count(reference_by_id, parsed_by_id)
        files[name] = stats

    postprocess_reports = {}
    for item in args.postprocess_report:
        name, raw_path = item.split("=", 1)
        report = load_json(Path(raw_path))
        postprocess_reports[name] = report.get("summary", report)

    ablation_manifests = {}
    for item in args.ablation_manifest:
        name, raw_path = item.split("=", 1)
        manifest = load_json(Path(raw_path))
        stats = manifest.get("stats", {})
        restore_variant = next(
            (
                variant
                for variant in manifest.get("variants", [])
                if variant.get("name") == "baseline_restore_grab"
            ),
            None,
        )
        ablation_manifests[name] = {
            "restored_grab_items": stats.get("restored_grab_items"),
            "baseline_restore_grab": restore_variant,
        }

    output = {
        "prompts": {
            "path": args.prompts,
            "rows": len(prompt_ids),
            "unique_identifiers": len(prompt_id_set),
        },
        "reference": reference_stats,
        "files": files,
        "postprocess_reports": postprocess_reports,
        "ablation_manifests": ablation_manifests,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
