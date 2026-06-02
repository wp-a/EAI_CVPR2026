#!/usr/bin/env python3
"""Vote VirtualHome GI candidates and build a submission directory."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


OUTPUT_FILE = "virtualhome_goal_interpretation_outputs.json"
REQUIRED_FILES = {
    "behavior_action_sequencing_outputs.json": 100,
    "behavior_goal_interpretation_outputs.json": 100,
    "behavior_subgoal_decomposition_outputs.json": 100,
    "behavior_transition_modeling_outputs.json": 100,
    "virtualhome_action_sequencing_outputs.json": 1500,
    "virtualhome_goal_interpretation_outputs.json": 1500,
    "virtualhome_subgoal_decomposition_outputs.json": 1500,
    "virtualhome_transition_modeling_outputs.json": 1500,
}
DISALLOWED_PATH_PARTS = {"evalai_valid_submissions", "high_score_62.69"}


@dataclass
class PromptConstraints:
    objects: list[str]
    object_states: dict[str, set[str]]
    relations: list[str]
    relation_targets: dict[str, set[str]]
    actions: list[str]


@dataclass
class CandidateGoals:
    nodes: list[tuple[str, str]] = field(default_factory=list)
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    actions: list[tuple[str, str]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.nodes and not self.edges and not self.actions


@dataclass
class ParseResult:
    goals: CandidateGoals
    parsed_json: bool
    invalid_nodes: int = 0
    invalid_edges: int = 0
    invalid_actions: int = 0


def check_allowed_path(path: Path, role: str) -> None:
    parts = set(path.parts)
    blocked = sorted(parts & DISALLOWED_PATH_PARTS)
    if blocked:
        raise SystemExit(f"Refusing to use disallowed {role} path {path}: contains {blocked}")


def load_json(path: Path) -> Any:
    check_allowed_path(path, "input")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_literal_after(prompt: str, anchor: str, end_anchor: str | None = None) -> Any | None:
    start = prompt.find(anchor)
    if start < 0:
        return None
    start = prompt.find("{", start)
    if start < 0:
        return None
    if end_anchor:
        end_limit = prompt.find(end_anchor, start)
        search_text = prompt[start:end_limit if end_limit >= 0 else len(prompt)]
    else:
        search_text = prompt[start:]

    depth = 0
    for offset, char in enumerate(search_text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                literal = search_text[: offset + 1]
                try:
                    return ast.literal_eval(literal)
                except Exception:
                    return None
    return None


def parse_object_constraints(prompt: str) -> tuple[list[str], dict[str, set[str]]]:
    start = prompt.find("Relevant objects in the scene are:")
    end = prompt.find("All possible relationships", start)
    if start < 0 or end < 0:
        return [], {}
    block = prompt[start:end]
    objects: list[str] = []
    object_states: dict[str, set[str]] = {}
    for line in block.splitlines():
        line = line.strip()
        match = re.match(r"^([^,\n]+),\s*initial states:.*?possible states:\s*(\[.*\])\s*$", line)
        if not match:
            continue
        name = match.group(1).strip()
        try:
            states = {str(state).upper() for state in ast.literal_eval(match.group(2))}
        except Exception:
            states = set()
        objects.append(name)
        object_states[name] = states
    return objects, object_states


def parse_prompt_constraints(prompt: str) -> PromptConstraints:
    objects, object_states = parse_object_constraints(prompt)
    relation_dict = extract_literal_after(
        prompt,
        "All possible relationships are the keys",
        "Symbolic goals format",
    )
    relation_targets = extract_literal_after(
        prompt,
        "Each relation has a fixed set of objects",
        "Action goals is a list",
    )
    action_dict = extract_literal_after(
        prompt,
        "Below is a dictionary of possible actions",
        "Goal name and goal description",
    )

    relations = [str(relation).upper() for relation in (relation_dict or {}).keys()]
    targets = {
        str(relation).upper(): {str(target) for target in target_set}
        for relation, target_set in (relation_targets or {}).items()
    }
    actions = [str(action).upper() for action in (action_dict or {}).keys()]
    return PromptConstraints(objects=objects, object_states=object_states, relations=relations, relation_targets=targets, actions=actions)


def strip_noise(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"```(?:json|python)?", "", text, flags=re.I)
    return text.replace("```", "").strip()


def find_json_object(text: str) -> str | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        quote = ""
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote:
                    in_string = False
                continue
            if char in {"'", '"'}:
                in_string = True
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def parse_jsonish(text: str) -> dict[str, Any] | None:
    text = strip_noise(text)
    candidate = find_json_object(text)
    if candidate is None:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            decoded = loader(candidate)
        except Exception:
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


def canonical_key(raw_key: str) -> str:
    return re.sub(r"[^a-z]", "", str(raw_key).lower())


def list_for_key(decoded: dict[str, Any], aliases: set[str]) -> list[Any]:
    for key, value in decoded.items():
        if canonical_key(key) in aliases and isinstance(value, list):
            return value
    return []


def normalize_item_value(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in item and item[key] is not None:
            return str(item[key]).strip()
    lowered = {canonical_key(key): value for key, value in item.items()}
    for key in keys:
        normalized = canonical_key(key)
        if normalized in lowered and lowered[normalized] is not None:
            return str(lowered[normalized]).strip()
    return ""


def parse_candidate_output(text: str, constraints: PromptConstraints, strict_relation_targets: bool) -> ParseResult:
    decoded = parse_jsonish(text)
    if decoded is None:
        return ParseResult(goals=CandidateGoals(), parsed_json=False)

    object_set = set(constraints.objects)
    relation_set = set(constraints.relations)
    action_set = set(constraints.actions)
    goals = CandidateGoals()
    invalid_nodes = invalid_edges = invalid_actions = 0

    node_items = list_for_key(decoded, {"nodegoals", "nodes", "nodegoal"})
    for item in node_items:
        if not isinstance(item, dict):
            invalid_nodes += 1
            continue
        name = normalize_item_value(item, "name", "object", "object_name")
        state = normalize_item_value(item, "state").upper()
        if name not in object_set or state not in constraints.object_states.get(name, set()):
            invalid_nodes += 1
            continue
        goals.nodes.append((name, state))

    edge_items = list_for_key(decoded, {"edgegoals", "edges", "edgegoal"})
    for item in edge_items:
        if not isinstance(item, dict):
            invalid_edges += 1
            continue
        from_name = normalize_item_value(item, "from_name", "from", "source", "object1")
        relation = normalize_item_value(item, "relation", "rel").upper()
        to_name = normalize_item_value(item, "to_name", "to", "target", "object2")
        target_ok = True
        if strict_relation_targets and relation in constraints.relation_targets:
            target_ok = to_name in constraints.relation_targets[relation]
        if from_name not in object_set or to_name not in object_set or relation not in relation_set or not target_ok:
            invalid_edges += 1
            continue
        goals.edges.append((from_name, relation, to_name))

    action_items = list_for_key(decoded, {"actiongoals", "actions", "actiongoal"})
    for item in action_items:
        if isinstance(item, str):
            action = item.strip().upper()
            description = item.strip()
        elif isinstance(item, dict):
            action = normalize_item_value(item, "action", "name").upper()
            description = normalize_item_value(item, "description", "desc")
        else:
            invalid_actions += 1
            continue
        if action not in action_set:
            invalid_actions += 1
            continue
        goals.actions.append((action, description or action))

    goals.nodes = list(dict.fromkeys(goals.nodes))
    goals.edges = list(dict.fromkeys(goals.edges))
    action_map: dict[str, str] = {}
    for action, description in goals.actions:
        action_map.setdefault(action, description)
    goals.actions = [(action, description) for action, description in action_map.items()]
    return ParseResult(goals=goals, parsed_json=True, invalid_nodes=invalid_nodes, invalid_edges=invalid_edges, invalid_actions=invalid_actions)


def render_goals(goals: CandidateGoals) -> str:
    payload = {
        "node goals": [{"name": name, "state": state} for name, state in goals.nodes],
        "edge goals": [
            {"from_name": from_name, "relation": relation, "to_name": to_name}
            for from_name, relation, to_name in goals.edges
        ],
        "action goals": [
            {"action": action, "description": description}
            for action, description in goals.actions
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_output_map(path: Path) -> dict[str, str]:
    rows = load_json(path)
    return {row["identifier"]: row.get("llm_output", "") for row in rows}


def discover_candidate_files(candidate_values: list[list[str]], candidate_dirs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for group in candidate_values:
        for value in group:
            path = Path(value)
            check_allowed_path(path, "candidate")
            if path.is_file():
                paths.append(path)
            elif path.is_dir():
                direct = path / OUTPUT_FILE
                if direct.exists():
                    paths.append(direct)
    for raw_dir in candidate_dirs:
        root = Path(raw_dir)
        check_allowed_path(root, "candidate-dir")
        if not root.exists():
            continue
        paths.extend(sorted(root.glob("*/virtualhome_goal_interpretation*_outputs.json")))
        paths.extend(sorted(root.glob("virtualhome_goal_interpretation*_outputs.json")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def sort_nodes(nodes: list[tuple[str, str]], constraints: PromptConstraints) -> list[tuple[str, str]]:
    object_rank = {name: index for index, name in enumerate(constraints.objects)}
    return sorted(nodes, key=lambda item: (object_rank.get(item[0], 10_000), item[0], item[1]))


def sort_edges(edges: list[tuple[str, str, str]], constraints: PromptConstraints) -> list[tuple[str, str, str]]:
    object_rank = {name: index for index, name in enumerate(constraints.objects)}
    relation_rank = {name: index for index, name in enumerate(constraints.relations)}
    return sorted(
        edges,
        key=lambda item: (
            object_rank.get(item[0], 10_000),
            relation_rank.get(item[1], 10_000),
            object_rank.get(item[2], 10_000),
            item,
        ),
    )


def sort_actions(actions: list[tuple[str, str]], constraints: PromptConstraints) -> list[tuple[str, str]]:
    action_rank = {name: index for index, name in enumerate(constraints.actions)}
    return sorted(actions, key=lambda item: (action_rank.get(item[0], 10_000), item[0]))


def vote_goals(
    parsed_candidates: list[CandidateGoals],
    constraints: PromptConstraints,
    node_threshold: int,
    edge_threshold: int,
    action_threshold: int,
) -> CandidateGoals:
    node_counts: Counter[tuple[str, str]] = Counter()
    edge_counts: Counter[tuple[str, str, str]] = Counter()
    action_counts: Counter[str] = Counter()
    action_descriptions: dict[str, Counter[str]] = defaultdict(Counter)
    for goals in parsed_candidates:
        node_counts.update(set(goals.nodes))
        edge_counts.update(set(goals.edges))
        action_seen = {action for action, _ in goals.actions}
        action_counts.update(action_seen)
        for action, description in goals.actions:
            action_descriptions[action][description] += 1

    nodes = [node for node, count in node_counts.items() if count >= node_threshold]
    edges = [edge for edge, count in edge_counts.items() if count >= edge_threshold]
    actions: list[tuple[str, str]] = []
    for action, count in action_counts.items():
        if count < action_threshold:
            continue
        description = action_descriptions[action].most_common(1)[0][0] if action_descriptions[action] else action
        actions.append((action, description))
    return CandidateGoals(
        nodes=sort_nodes(nodes, constraints),
        edges=sort_edges(edges, constraints),
        actions=sort_actions(actions, constraints),
    )


def validate_rows(rows: list[dict[str, str]], expected_identifiers: list[str]) -> dict[str, int]:
    json_ok = schema_ok = think_count = fence_count = order_ok = 0
    for index, row in enumerate(rows):
        if index < len(expected_identifiers) and row.get("identifier") == expected_identifiers[index]:
            order_ok += 1
        text = row.get("llm_output", "")
        if "<think>" in text:
            think_count += 1
        if "```" in text:
            fence_count += 1
        decoded = parse_jsonish(text)
        if decoded is None:
            continue
        json_ok += 1
        keys = {canonical_key(key) for key in decoded.keys()}
        if {"nodegoals", "edgegoals", "actiongoals"}.issubset(keys):
            schema_ok += 1
    return {
        "rows": len(rows),
        "order_ok": order_ok,
        "json_ok": json_ok,
        "schema_ok": schema_ok,
        "think_count": think_count,
        "fence_count": fence_count,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zip_submission(output_dir: Path) -> Path:
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(output_dir / name, arcname=name)
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Vote VirtualHome GI candidates and build a submission.")
    parser.add_argument("--prompt-file", default="llm_prompts/virtualhome_goal_interpretation_prompts.json")
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--candidate", nargs="+", action="append", default=[])
    parser.add_argument("--candidate-dir", action="append", default=[])
    parser.add_argument("--fallback", required=True)
    parser.add_argument(
        "--secondary-fallback",
        default="outputs/qwen32_awq_base_score_44.6375/virtualhome_goal_interpretation_outputs.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-threshold", type=int, default=2)
    parser.add_argument("--edge-threshold", type=int, default=2)
    parser.add_argument("--action-threshold", type=int, default=1)
    parser.add_argument("--strict-relation-targets", action="store_true")
    args = parser.parse_args()

    prompt_file = Path(args.prompt_file)
    base_submission = Path(args.base_submission)
    output_dir = Path(args.output_dir)
    fallback_path = Path(args.fallback)
    secondary_fallback_path = Path(args.secondary_fallback)
    for path, role in [
        (prompt_file, "prompt-file"),
        (base_submission, "base-submission"),
        (fallback_path, "fallback"),
        (secondary_fallback_path, "secondary-fallback"),
        (output_dir, "output-dir"),
    ]:
        check_allowed_path(path, role)

    prompt_rows = load_json(prompt_file)
    expected_identifiers = [row["identifier"] for row in prompt_rows]
    constraints_by_id = {
        row["identifier"]: parse_prompt_constraints(row["llm_prompt"])
        for row in prompt_rows
    }

    candidate_files = discover_candidate_files(args.candidate, args.candidate_dir)
    if not candidate_files:
        raise SystemExit("No candidate files found.")

    candidate_maps = [(path, load_output_map(path)) for path in candidate_files]
    fallback = load_output_map(fallback_path)
    secondary_fallback = load_output_map(secondary_fallback_path) if secondary_fallback_path.exists() else {}

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(base_submission, output_dir)

    report: dict[str, Any] = {
        "prompt_file": str(prompt_file),
        "base_submission": str(base_submission),
        "output_dir": str(output_dir),
        "candidate_files": [str(path) for path in candidate_files],
        "thresholds": {
            "node": args.node_threshold,
            "edge": args.edge_threshold,
            "action": args.action_threshold,
            "strict_relation_targets": args.strict_relation_targets,
        },
        "candidate_stats": {},
        "fallback_count": 0,
        "secondary_fallback_count": 0,
        "invalid_nodes": 0,
        "invalid_edges": 0,
        "invalid_actions": 0,
        "parsed_candidates": 0,
        "unparsed_candidates": 0,
        "empty_vote_count": 0,
    }

    output_rows: list[dict[str, str]] = []
    for identifier in expected_identifiers:
        constraints = constraints_by_id[identifier]
        parsed_candidates: list[CandidateGoals] = []
        for path, candidate_map in candidate_maps:
            if identifier not in candidate_map:
                continue
            result = parse_candidate_output(candidate_map[identifier], constraints, args.strict_relation_targets)
            stats = report["candidate_stats"].setdefault(str(path), {"seen": 0, "parsed": 0, "unparsed": 0})
            stats["seen"] += 1
            if result.parsed_json:
                stats["parsed"] += 1
                report["parsed_candidates"] += 1
                parsed_candidates.append(result.goals)
            else:
                stats["unparsed"] += 1
                report["unparsed_candidates"] += 1
            report["invalid_nodes"] += result.invalid_nodes
            report["invalid_edges"] += result.invalid_edges
            report["invalid_actions"] += result.invalid_actions

        voted = vote_goals(
            parsed_candidates,
            constraints,
            args.node_threshold,
            args.edge_threshold,
            args.action_threshold,
        )
        if voted.is_empty():
            report["empty_vote_count"] += 1
            fallback_result = parse_candidate_output(fallback.get(identifier, ""), constraints, args.strict_relation_targets)
            if fallback_result.parsed_json and not fallback_result.goals.is_empty():
                voted = fallback_result.goals
                report["fallback_count"] += 1
            else:
                secondary_result = parse_candidate_output(secondary_fallback.get(identifier, ""), constraints, args.strict_relation_targets)
                voted = secondary_result.goals
                report["secondary_fallback_count"] += 1

        output_rows.append({"identifier": identifier, "llm_output": render_goals(voted)})

    output_path = output_dir / OUTPUT_FILE
    write_json(output_path, output_rows)

    report["output_validation"] = validate_rows(output_rows, expected_identifiers)
    report["submission_files"] = {}
    for name, expected_count in REQUIRED_FILES.items():
        path = output_dir / name
        rows = load_json(path)
        report["submission_files"][name] = {
            "rows": len(rows),
            "expected_rows": expected_count,
            "ok": len(rows) == expected_count,
        }
    base_hashes = {}
    for name in REQUIRED_FILES:
        base_file = base_submission / name
        output_file = output_dir / name
        if base_file.exists() and output_file.exists():
            base_hashes[name] = sha256(base_file) == sha256(output_file)
    report["same_as_base_submission"] = base_hashes
    report["zip_path"] = str(zip_submission(output_dir))

    report_path = output_dir.with_name(output_dir.name + "_vgi_vote_report.json")
    write_json(report_path, report)
    print(report_path)
    print(report["zip_path"])


if __name__ == "__main__":
    main()
