#!/usr/bin/env python3
"""Vote and postprocess learned-rules E4 VGI candidates from current-run outputs only."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from make_vgi_vote_submission import (
    CandidateGoals,
    check_allowed_path,
    discover_candidate_files,
    list_for_key,
    load_json,
    normalize_item_value,
    parse_jsonish,
    parse_prompt_constraints,
    render_goals,
    sort_actions,
    sort_edges,
    sort_nodes,
    validate_rows,
    write_json,
)


DISALLOWED_CANDIDATE_PARTS = {"outputs", "evalai_valid_submissions", "high_score_62.69", "sample_submission"}
ROOM_NAMES = {"bathroom", "bedroom", "dining_room", "home_office", "kitchen", "living_room"}
PROCESS_ACTIONS = {"FIND", "WALK", "OPEN", "CLOSE", "PUTIN", "SWITCHON", "PUTBACK", "PUTOBJBACK"}
STATE_PRUNE = {"GRABBED", "OCCUPIED", "FULL", "EMPTY", "FREE", "INSIDE"}

TASK_ACTIONS: dict[str, list[str]] = {
    "browse internet": ["LOOKAT"],
    "watch tv": ["LOOKAT", "WATCH"],
    "change tv channel": ["LOOKAT", "WATCH"],
    "write an email": ["TYPE", "SWITCHOFF"],
    "work": ["TYPE"],
    "wash hands": ["RINSE", "WASH"],
    "drink": ["DRINK"],
    "read book": ["READ"],
    "wash dishes by hand": ["GRAB", "WASH", "RINSE", "SCRUB", "WIPE"],
    "pet cat": ["TOUCH"],
    "go to sleep": ["SLEEP"],
    "make coffee": ["POUR"],
    "wash teeth": ["RINSE", "WASH"],
    "brush teeth": ["RINSE", "WASH"],
    "take shower": ["RINSE", "WASH"],
}

DEFAULT_ACTION_TASKS = {
    "browse internet",
    "watch tv",
    "change tv channel",
    "write an email",
    "wash hands",
    "drink",
    "read book",
    "wash dishes by hand",
    "pet cat",
    "go to sleep",
    "make coffee",
    "wash teeth",
    "brush teeth",
    "take shower",
}

PLACEMENT_TASKS = {
    "wash clothes",
    "wash dishes with dishwasher",
    "set up table",
    "put groceries in fridge",
    "wash dishes by hand",
    "make coffee",
}


def check_candidate_source(path: Path) -> None:
    blocked = sorted(set(path.parts) & DISALLOWED_CANDIDATE_PARTS)
    if blocked:
        raise SystemExit(f"Refusing non-current-run candidate path {path}: contains {blocked}")


def load_output_map(path: Path) -> dict[str, str]:
    rows = load_json(path)
    return {row["identifier"]: row.get("llm_output", "") for row in rows}


def extract_goal_name(prompt: str) -> str:
    match = re.search(r"Goal name:\s*(.*?)(?:\n|$)", prompt)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def task_key(goal_name: str) -> str:
    return re.sub(r"\s+", " ", goal_name.strip().lower())


def is_room(name: str) -> bool:
    return name in ROOM_NAMES or name.endswith("_room")


def state_allowed(constraints, name: str, state: str) -> bool:
    return name in set(constraints.objects) and state in constraints.object_states.get(name, set())


def relation_allowed(constraints, from_name: str, relation: str, to_name: str) -> bool:
    if from_name == to_name:
        return False
    if from_name not in set(constraints.objects) or to_name not in set(constraints.objects):
        return False
    if relation not in set(constraints.relations):
        return False
    if relation == "CLOSE":
        return True
    targets = constraints.relation_targets.get(relation)
    return not targets or to_name in targets


def action_allowed(constraints, action: str, task: str) -> bool:
    if action not in set(constraints.actions):
        return False
    whitelist = TASK_ACTIONS.get(task, [])
    if not whitelist:
        return False
    if action not in whitelist:
        return False
    if action in PROCESS_ACTIONS and not (task == "wash dishes by hand" and action == "GRAB"):
        return False
    return True


def should_prune_node(name: str, state: str, task: str) -> bool:
    if is_room(name):
        return True
    if state in STATE_PRUNE:
        return True
    if state == "CLEAN" and not (task in {"wash dishes with dishwasher", "wash clothes"} and name in {"dishwasher", "washing_machine"}):
        return True
    if name in {"novel", "book"} and state == "OPEN":
        return True
    if name == "faucet" and state == "OFF":
        return True
    if name == "character" and state == "SITTING" and task not in {"relax on sofa", "go to toilet"}:
        return True
    if name == "character" and state == "LYING" and task not in {"relax on sofa", "go to sleep"}:
        return True
    return False


def parse_candidate(text: str, constraints, task: str) -> tuple[CandidateGoals, bool, Counter[str]]:
    decoded = parse_jsonish(text)
    stats: Counter[str] = Counter()
    if decoded is None:
        return CandidateGoals(), False, stats

    goals = CandidateGoals()
    for item in list_for_key(decoded, {"nodegoals", "nodes", "nodegoal"}):
        if not isinstance(item, dict):
            stats["invalid_node"] += 1
            continue
        name = normalize_item_value(item, "name", "object", "object_name")
        state = normalize_item_value(item, "state").upper()
        if not state_allowed(constraints, name, state):
            stats["invalid_node"] += 1
            continue
        if should_prune_node(name, state, task):
            stats["pruned_node"] += 1
            continue
        goals.nodes.append((name, state))

    for item in list_for_key(decoded, {"edgegoals", "edges", "edgegoal"}):
        if not isinstance(item, dict):
            stats["invalid_edge"] += 1
            continue
        from_name = normalize_item_value(item, "from_name", "from", "source", "object1")
        relation = normalize_item_value(item, "relation", "rel").upper()
        to_name = normalize_item_value(item, "to_name", "to", "target", "object2")
        if not relation_allowed(constraints, from_name, relation, to_name):
            stats["invalid_edge"] += 1
            continue
        goals.edges.append((from_name, relation, to_name))

    for item in list_for_key(decoded, {"actiongoals", "actions", "actiongoal"}):
        if isinstance(item, str):
            action = item.strip().upper()
            description = item.strip()
        elif isinstance(item, dict):
            action = normalize_item_value(item, "action", "name").upper()
            description = normalize_item_value(item, "description", "desc")
        else:
            stats["invalid_action"] += 1
            continue
        if not action_allowed(constraints, action, task):
            stats["invalid_action"] += 1
            continue
        goals.actions.append((action, description or action))

    goals.nodes = list(dict.fromkeys(goals.nodes))
    goals.edges = list(dict.fromkeys(goals.edges))
    action_map: dict[str, str] = {}
    for action, description in goals.actions:
        action_map.setdefault(action, description)
    goals.actions = list(action_map.items())
    return goals, True, stats


def find_objects(constraints, *needles: str) -> list[str]:
    needles = tuple(needle.lower() for needle in needles)
    return [name for name in constraints.objects if any(needle in name.lower() for needle in needles)]


def first_object(constraints, *needles: str) -> str | None:
    objects = find_objects(constraints, *needles)
    return objects[0] if objects else None


def add_node(goals: CandidateGoals, constraints, name: str | None, state: str) -> None:
    if name and state_allowed(constraints, name, state):
        goals.nodes.append((name, state))


def add_edge(goals: CandidateGoals, constraints, from_name: str | None, relation: str, to_name: str | None) -> None:
    if from_name and to_name and relation_allowed(constraints, from_name, relation, to_name):
        goals.edges.append((from_name, relation, to_name))


def add_action(goals: CandidateGoals, constraints, task: str, action: str, description: str) -> None:
    if action_allowed(constraints, action, task):
        goals.actions.append((action, description))


def add_powered_states(goals: CandidateGoals, constraints, *names: str) -> None:
    for name in names:
        add_node(goals, constraints, name, "ON")
        add_node(goals, constraints, name, "PLUGGED_IN")


def template_goals(constraints, task: str) -> CandidateGoals:
    goals = CandidateGoals()
    character = first_object(constraints, "character")

    if task == "wash clothes":
        machine = first_object(constraints, "washing_machine")
        add_node(goals, constraints, machine, "CLOSED")
        add_powered_states(goals, constraints, *(filter(None, [machine])))
        for obj in find_objects(constraints, "clothes", "shirt", "jacket", "pants", "soap", "detergent"):
            if obj != machine:
                add_edge(goals, constraints, obj, "ON", machine)

    elif task == "wash dishes with dishwasher":
        dishwasher = first_object(constraints, "dishwasher")
        add_node(goals, constraints, dishwasher, "CLOSED")
        add_node(goals, constraints, dishwasher, "CLEAN")
        add_powered_states(goals, constraints, *(filter(None, [dishwasher])))
        for obj in find_objects(constraints, "dish_soap", "plate", "cup", "fork", "spoon", "knife", "bowl"):
            if obj != dishwasher:
                add_edge(goals, constraints, obj, "ON", dishwasher)

    elif task == "turn on light":
        for obj in find_objects(constraints, "light", "lamp"):
            add_powered_states(goals, constraints, obj)

    elif task in {"watch tv", "change tv channel"}:
        tv = first_object(constraints, "television", "tv")
        remote = first_object(constraints, "remote_control", "remote")
        add_powered_states(goals, constraints, *(filter(None, [tv])))
        add_edge(goals, constraints, character, "FACING", tv)
        add_edge(goals, constraints, character, "HOLDS_RH", remote)
        add_action(goals, constraints, task, "LOOKAT", "look at the television")

    elif task == "browse internet":
        device = first_object(constraints, "computer", "laptop")
        mouse = first_object(constraints, "mouse")
        keyboard = first_object(constraints, "keyboard")
        add_powered_states(goals, constraints, *(filter(None, [device])))
        add_edge(goals, constraints, character, "CLOSE", device)
        add_edge(goals, constraints, character, "FACING", device)
        add_edge(goals, constraints, character, "HOLDS_RH", mouse)
        add_edge(goals, constraints, character, "HOLDS_LH", keyboard)
        add_action(goals, constraints, task, "LOOKAT", "look at the computer")

    elif task in {"write an email", "work"}:
        device = first_object(constraints, "computer", "laptop")
        mouse = first_object(constraints, "mouse")
        add_node(goals, constraints, device, "ON")
        add_edge(goals, constraints, character, "CLOSE", device)
        add_edge(goals, constraints, character, "FACING", device)
        add_edge(goals, constraints, character, "HOLDS_RH", mouse)
        if task == "write an email":
            add_action(goals, constraints, task, "TYPE", "type the email")

    elif task == "pick up phone":
        phone = first_object(constraints, "phone", "cellphone")
        add_edge(goals, constraints, character, "HOLDS_RH", phone)

    elif task == "drink":
        cup = first_object(constraints, "water_glass", "drinking_glass", "glass", "cup", "mug")
        add_edge(goals, constraints, character, "HOLDS_RH", cup)
        add_action(goals, constraints, task, "DRINK", "drink")

    elif task == "read book":
        book = first_object(constraints, "novel", "book")
        add_edge(goals, constraints, character, "HOLDS_RH", book)
        add_action(goals, constraints, task, "READ", "read the book")

    elif task == "relax on sofa":
        couch = first_object(constraints, "couch", "sofa")
        add_node(goals, constraints, character, "SITTING")
        add_edge(goals, constraints, character, "ON", couch)

    elif task == "pet cat":
        cat = first_object(constraints, "cat")
        add_edge(goals, constraints, character, "CLOSE", cat)
        add_action(goals, constraints, task, "TOUCH", "touch the cat")

    elif task == "go to sleep":
        bed = first_object(constraints, "bed")
        add_node(goals, constraints, character, "LYING")
        add_edge(goals, constraints, character, "ON", bed)
        add_action(goals, constraints, task, "SLEEP", "sleep")

    elif task == "set up table":
        table = first_object(constraints, "table")
        for obj in find_objects(constraints, "plate", "cup", "napkin", "fork", "spoon", "knife", "bowl"):
            if obj != table:
                add_edge(goals, constraints, obj, "ON", table)

    elif task == "put groceries in fridge":
        target = first_object(constraints, "fridge", "freezer", "refrigerator")
        for obj in find_objects(constraints, "food", "grocery", "apple", "milk", "juice", "cheese", "salmon", "chicken"):
            if obj != target:
                add_edge(goals, constraints, obj, "INSIDE", target)

    elif task == "make coffee":
        maker = first_object(constraints, "coffee")
        cup = first_object(constraints, "mug", "cup")
        add_powered_states(goals, constraints, *(filter(None, [maker])))
        add_edge(goals, constraints, cup, "CLOSE", maker)
        add_action(goals, constraints, task, "POUR", "pour coffee")

    elif task == "wash dishes by hand":
        sink = first_object(constraints, "sink")
        for obj in find_objects(constraints, "plate", "cup", "fork", "spoon", "knife", "bowl", "dish_soap"):
            if obj != sink:
                add_edge(goals, constraints, obj, "CLOSE", sink)
        add_action(goals, constraints, task, "WASH", "wash the dishes")

    elif task in {"wash hands", "wash teeth", "brush teeth", "take shower"}:
        add_action(goals, constraints, task, "RINSE", "rinse")
        add_action(goals, constraints, task, "WASH", "wash")
        if task == "brush teeth":
            toothbrush = first_object(constraints, "toothbrush")
            add_edge(goals, constraints, character, "HOLDS_RH", toothbrush)
        if task == "take shower":
            shower = first_object(constraints, "shower")
            add_edge(goals, constraints, character, "CLOSE", shower)

    elif task == "go to toilet":
        toilet = first_object(constraints, "toilet")
        add_node(goals, constraints, character, "SITTING")
        add_edge(goals, constraints, character, "ON", toilet)

    elif task == "cook some food":
        food = find_objects(constraints, "food", "salmon", "chicken", "pie")
        for obj in food:
            add_node(goals, constraints, obj, "COOKED")
        appliance = first_object(constraints, "stove", "oven", "microwave")
        add_node(goals, constraints, appliance, "ON")

    elif task == "get some water":
        cup = first_object(constraints, "water_glass", "drinking_glass", "glass", "cup")
        add_edge(goals, constraints, character, "HOLDS_RH", cup)

    return dedupe_and_sort(goals, constraints, task)


def dedupe_and_sort(goals: CandidateGoals, constraints, task: str) -> CandidateGoals:
    nodes = []
    for name, state in goals.nodes:
        if state_allowed(constraints, name, state) and not should_prune_node(name, state, task):
            nodes.append((name, state))
    edges = []
    for from_name, relation, to_name in goals.edges:
        if relation_allowed(constraints, from_name, relation, to_name):
            edges.append((from_name, relation, to_name))
    action_map: dict[str, str] = {}
    for action, description in goals.actions:
        if action_allowed(constraints, action, task):
            action_map.setdefault(action, description or action)
    return CandidateGoals(
        nodes=sort_nodes(list(dict.fromkeys(nodes)), constraints),
        edges=sort_edges(list(dict.fromkeys(edges)), constraints),
        actions=sort_actions(list(action_map.items()), constraints),
    )


def edge_threshold(edge: tuple[str, str, str], task: str, template_edges: set[tuple[str, str, str]]) -> int:
    relation = edge[1]
    if edge in template_edges:
        return 1
    if task in PLACEMENT_TASKS and relation in {"ON", "INSIDE", "CLOSE"}:
        return 1
    if task in {"browse internet", "watch tv", "change tv channel", "pick up phone", "drink", "read book", "pet cat", "relax on sofa", "go to sleep"} and relation in {"CLOSE", "FACING", "HOLDS_RH", "HOLDS_LH", "ON"}:
        return 1
    return 2


def node_threshold(node: tuple[str, str], task: str, template_nodes: set[tuple[str, str]]) -> int:
    if node in template_nodes:
        return 1
    if task in {"turn on light", "wash clothes", "wash dishes with dishwasher", "watch tv", "browse internet", "change tv channel"}:
        return 1
    return 2


def action_threshold(task: str, action: str, template_actions: set[str]) -> int:
    if task == "work":
        return 3
    if action in template_actions and task in DEFAULT_ACTION_TASKS:
        return 1
    return 2


def max_actions(task: str) -> int:
    if task in {"wash clothes", "wash dishes with dishwasher", "pick up phone", "turn on light", "set up table", "put groceries in fridge", "relax on sofa", "work", "cook some food", "listen to music", "go to toilet", "get some water"}:
        return 0
    if task == "wash dishes by hand":
        return 2
    return 1


def edge_cap(task: str) -> int:
    if task in {"wash clothes", "wash dishes with dishwasher"}:
        return 7
    if task in {"set up table", "put groceries in fridge", "wash dishes by hand"}:
        return 6
    if task in {"browse internet", "watch tv", "change tv channel", "work", "write an email"}:
        return 4
    return 3


def edge_priority(edge: tuple[str, str, str], task: str, template_edges: set[tuple[str, str, str]], count: int) -> tuple[int, int, str]:
    relation = edge[1]
    template_bonus = 0 if edge in template_edges else 1
    relation_rank = {"INSIDE": 0, "ON": 1, "HOLDS_RH": 2, "HOLDS_LH": 3, "FACING": 4, "CLOSE": 5}.get(relation, 9)
    return (template_bonus, -count, f"{relation_rank}:{edge}")


def vote_goals(candidates: list[CandidateGoals], constraints, task: str, template: CandidateGoals) -> CandidateGoals:
    node_counts: Counter[tuple[str, str]] = Counter()
    edge_counts: Counter[tuple[str, str, str]] = Counter()
    action_counts: Counter[str] = Counter()
    action_descriptions: dict[str, Counter[str]] = defaultdict(Counter)

    for goals in candidates:
        node_counts.update(set(goals.nodes))
        edge_counts.update(set(goals.edges))
        seen_actions = {action for action, _ in goals.actions}
        action_counts.update(seen_actions)
        for action, description in goals.actions:
            action_descriptions[action][description or action] += 1

    template_nodes = set(template.nodes)
    template_edges = set(template.edges)
    template_actions = {action for action, _ in template.actions}

    nodes = [node for node, count in node_counts.items() if count >= node_threshold(node, task, template_nodes)]
    for node in template.nodes:
        if task in {"wash clothes", "wash dishes with dishwasher", "turn on light", "watch tv", "browse internet", "change tv channel", "relax on sofa", "go to sleep", "go to toilet", "cook some food"}:
            nodes.append(node)

    edges = [edge for edge, count in edge_counts.items() if count >= edge_threshold(edge, task, template_edges)]
    for edge in template.edges:
        edges.append(edge)
    edges = list(dict.fromkeys(edges))
    edges = sorted(edges, key=lambda edge: edge_priority(edge, task, template_edges, edge_counts[edge]))[: edge_cap(task)]

    actions = []
    allowed_action_count = max_actions(task)
    if allowed_action_count:
        for action, count in action_counts.items():
            if count >= action_threshold(task, action, template_actions):
                description = action_descriptions[action].most_common(1)[0][0] if action_descriptions[action] else action
                actions.append((action, description))
        for action, description in template.actions:
            if task in DEFAULT_ACTION_TASKS:
                actions.append((action, description))
        action_map: dict[str, str] = {}
        for action, description in actions:
            if action_allowed(constraints, action, task):
                action_map.setdefault(action, description or action)
        actions = sort_actions(list(action_map.items()), constraints)[:allowed_action_count]

    voted = CandidateGoals(nodes=nodes, edges=edges, actions=actions)
    voted = dedupe_and_sort(voted, constraints, task)
    voted.edges = sorted(voted.edges, key=lambda edge: edge_priority(edge, task, template_edges, edge_counts[edge]))[: edge_cap(task)]
    voted.actions = voted.actions[:allowed_action_count]
    return voted


def shape_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    totals = Counter()
    empty = action_gt_2 = self_loop = parse_ok = 0
    relations = Counter()
    actions = Counter()
    for row in rows:
        decoded = parse_jsonish(row.get("llm_output", ""))
        if decoded is None:
            continue
        parse_ok += 1
        nodes = decoded.get("node goals") or []
        edges = decoded.get("edge goals") or []
        action_items = decoded.get("action goals") or []
        totals["node"] += len(nodes)
        totals["edge"] += len(edges)
        totals["action"] += len(action_items)
        empty += int(not nodes and not edges and not action_items)
        action_gt_2 += int(len(action_items) > 2)
        for edge in edges:
            if edge.get("from_name") == edge.get("to_name"):
                self_loop += 1
            relations[edge.get("relation")] += 1
        for item in action_items:
            actions[item.get("action")] += 1
    count = len(rows) or 1
    return {
        "rows": len(rows),
        "parse_ok": parse_ok,
        "avg_node": totals["node"] / count,
        "avg_edge": totals["edge"] / count,
        "avg_action": totals["action"] / count,
        "total_node": totals["node"],
        "total_edge": totals["edge"],
        "total_action": totals["action"],
        "empty": empty,
        "action_gt_2": action_gt_2,
        "self_loop": self_loop,
        "relations": dict(relations.most_common()),
        "actions": dict(actions.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Vote learned-rules E4 VGI candidates from current-run outputs only.")
    parser.add_argument("--prompt-file", default="llm_prompts/virtualhome_goal_interpretation_prompts.json")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    prompt_file = Path(args.prompt_file)
    candidate_dir = Path(args.candidate_dir)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else None

    check_allowed_path(prompt_file, "prompt-file")
    check_candidate_source(candidate_dir)
    check_allowed_path(output_path, "output")
    if report_path:
        check_allowed_path(report_path, "report")

    prompt_rows = load_json(prompt_file)
    identifiers = [row["identifier"] for row in prompt_rows]
    constraints_by_id = {row["identifier"]: parse_prompt_constraints(row["llm_prompt"]) for row in prompt_rows}
    task_by_id = {row["identifier"]: task_key(extract_goal_name(row["llm_prompt"])) for row in prompt_rows}

    candidate_files = discover_candidate_files([], [str(candidate_dir)])
    if not candidate_files:
        raise SystemExit("No E4 candidate files found.")
    for path in candidate_files:
        check_candidate_source(path)
    candidate_maps = [(path, load_output_map(path)) for path in candidate_files]

    output_rows: list[dict[str, str]] = []
    report: dict[str, Any] = {
        "source_policy": "current E4 candidates plus prompt-derived learned-rule templates only; no best/history fallback",
        "candidate_files": [str(path) for path in candidate_files],
        "parsed_candidates": 0,
        "unparsed_candidates": 0,
        "empty_after_vote": 0,
        "invalid": Counter(),
    }

    for identifier in identifiers:
        constraints = constraints_by_id[identifier]
        task = task_by_id[identifier]
        template = template_goals(constraints, task)
        parsed_candidates: list[CandidateGoals] = []
        for _, output_map in candidate_maps:
            goals, parsed, stats = parse_candidate(output_map.get(identifier, ""), constraints, task)
            if parsed:
                report["parsed_candidates"] += 1
                if not goals.is_empty():
                    parsed_candidates.append(goals)
            else:
                report["unparsed_candidates"] += 1
            report["invalid"].update(stats)

        final_goals = vote_goals(parsed_candidates, constraints, task, template)
        if final_goals.is_empty():
            report["empty_after_vote"] += 1
            final_goals = template
        output_rows.append({"identifier": identifier, "llm_output": render_goals(final_goals)})

    report["invalid"] = dict(report["invalid"])
    report["output_validation"] = validate_rows(output_rows, identifiers)
    report["output_shape"] = shape_stats(output_rows)
    write_json(output_path, output_rows)
    if report_path:
        write_json(report_path, report)
    print(output_path)
    print(json.dumps(report["output_validation"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(report["output_shape"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
