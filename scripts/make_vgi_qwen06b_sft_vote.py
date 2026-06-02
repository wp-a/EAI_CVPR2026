#!/usr/bin/env python3
"""Vote Qwen0.6B VGI SFT candidates without using old best fallback."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from make_vgi_e4_learned_rules_vote import (
    dedupe_and_sort,
    parse_candidate,
    task_key,
    extract_goal_name,
    template_goals,
)
from make_vgi_vote_submission import (
    CandidateGoals,
    check_allowed_path,
    discover_candidate_files,
    load_json,
    parse_jsonish,
    parse_prompt_constraints,
    render_goals,
    sort_actions,
    sort_edges,
    sort_nodes,
    validate_rows,
    write_json,
)


DISALLOWED_CANDIDATE_PARTS = {"outputs/best", "evalai_valid_submissions", "high_score_62.69", "sample_submission"}


def check_candidate_source(path: Path) -> None:
    text = str(path)
    blocked = [part for part in DISALLOWED_CANDIDATE_PARTS if part in text]
    if blocked:
        raise SystemExit(f"Refusing candidate path {path}: contains {blocked}")


def load_output_map(path: Path) -> dict[str, str]:
    rows = load_json(path)
    return {row["identifier"]: row.get("llm_output", "") for row in rows}


def vote(candidates: list[CandidateGoals], constraints, task: str, template: CandidateGoals) -> CandidateGoals:
    node_counts: Counter[tuple[str, str]] = Counter()
    edge_counts: Counter[tuple[str, str, str]] = Counter()
    action_counts: Counter[str] = Counter()
    action_desc: dict[str, Counter[str]] = defaultdict(Counter)
    for goals in candidates:
        node_counts.update(set(goals.nodes))
        edge_counts.update(set(goals.edges))
        seen_actions = {action for action, _ in goals.actions}
        action_counts.update(seen_actions)
        for action, description in goals.actions:
            action_desc[action][description or action] += 1

    node_threshold = 2 if len(candidates) >= 3 else 1
    edge_threshold = 2 if len(candidates) >= 3 else 1
    action_threshold = 2 if len(candidates) >= 3 else 1
    nodes = [node for node, count in node_counts.items() if count >= node_threshold]
    edges = [edge for edge, count in edge_counts.items() if count >= edge_threshold]
    actions = []
    for action, count in action_counts.items():
        if count >= action_threshold:
            actions.append((action, action_desc[action].most_common(1)[0][0]))
    goals = CandidateGoals(
        nodes=sort_nodes(nodes, constraints),
        edges=sort_edges(edges, constraints),
        actions=sort_actions(actions, constraints)[:2],
    )
    if goals.is_empty():
        return template
    return dedupe_and_sort(goals, constraints, task)


def shape_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    totals = Counter()
    empty = action_gt_2 = self_loop = parse_ok = 0
    for row in rows:
        decoded = parse_jsonish(row.get("llm_output", ""))
        if decoded is None:
            continue
        parse_ok += 1
        nodes = decoded.get("node goals") or []
        edges = decoded.get("edge goals") or []
        actions = decoded.get("action goals") or []
        totals["node"] += len(nodes)
        totals["edge"] += len(edges)
        totals["action"] += len(actions)
        empty += int(not nodes and not edges and not actions)
        action_gt_2 += int(len(actions) > 2)
        for edge in edges:
            self_loop += int(edge.get("from_name") == edge.get("to_name"))
    count = len(rows) or 1
    return {
        "rows": len(rows),
        "parse_ok": parse_ok,
        "total_node": totals["node"],
        "total_edge": totals["edge"],
        "total_action": totals["action"],
        "avg_node": totals["node"] / count,
        "avg_edge": totals["edge"] / count,
        "avg_action": totals["action"] / count,
        "empty": empty,
        "action_gt_2": action_gt_2,
        "self_loop": self_loop,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", default="llm_prompts/virtualhome_goal_interpretation_prompts.json")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    prompt_file = Path(args.prompt_file)
    candidate_dir = Path(args.candidate_dir)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_name(output_path.stem + "_report.json")
    check_allowed_path(prompt_file, "prompt-file")
    check_candidate_source(candidate_dir)
    check_allowed_path(output_path, "output")
    check_allowed_path(report_path, "report")

    prompt_rows = load_json(prompt_file)
    identifiers = [row["identifier"] for row in prompt_rows]
    constraints_by_id = {row["identifier"]: parse_prompt_constraints(row["llm_prompt"]) for row in prompt_rows}
    task_by_id = {row["identifier"]: task_key(extract_goal_name(row["llm_prompt"])) for row in prompt_rows}
    candidate_files = discover_candidate_files([], [str(candidate_dir)])
    if not candidate_files:
        raise SystemExit("No candidate files found")
    for path in candidate_files:
        check_candidate_source(path)
    candidate_maps = [(path, load_output_map(path)) for path in candidate_files]
    report: dict[str, Any] = {
        "source_policy": "Qwen0.6B SFT candidates plus prompt-derived task template fallback only; no old best fallback",
        "candidate_files": [str(path) for path in candidate_files],
        "parsed_candidates": 0,
        "unparsed_candidates": 0,
        "empty_after_vote": 0,
        "invalid": Counter(),
    }
    output_rows = []
    for identifier in identifiers:
        constraints = constraints_by_id[identifier]
        task = task_by_id[identifier]
        template = template_goals(constraints, task)
        parsed_candidates = []
        for _, output_map in candidate_maps:
            goals, parsed, stats = parse_candidate(output_map.get(identifier, ""), constraints, task)
            if parsed:
                report["parsed_candidates"] += 1
                if not goals.is_empty():
                    parsed_candidates.append(goals)
            else:
                report["unparsed_candidates"] += 1
            report["invalid"].update(stats)
        goals = vote(parsed_candidates, constraints, task, template)
        if goals.is_empty():
            report["empty_after_vote"] += 1
        output_rows.append({"identifier": identifier, "llm_output": render_goals(goals)})
    report["invalid"] = dict(report["invalid"])
    report["output_validation"] = validate_rows(output_rows, identifiers)
    report["output_shape"] = shape_stats(output_rows)
    write_json(output_path, output_rows)
    write_json(report_path, report)
    print(output_path)
    print(json.dumps(report["output_validation"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(report["output_shape"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
