#!/usr/bin/env python3
"""Utilities for VirtualHome subgoal decomposition cleanup and reranking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


STATE_NAMES = {
    "CLOSED",
    "OPEN",
    "ON",
    "OFF",
    "PLUGGED_IN",
    "PLUGGED_OUT",
    "SITTING",
    "LYING",
    "CLEAN",
    "DIRTY",
    "ONTOP",
    "INSIDE",
    "BETWEEN",
    "NEXT_TO",
    "FACING",
    "HOLDS_RH",
    "HOLDS_LH",
}

ACTION_NAMES = {
    "DRINK",
    "EAT",
    "CUT",
    "TOUCH",
    "LOOKAT",
    "WATCH",
    "READ",
    "TYPE",
    "PUSH",
    "PULL",
    "MOVE",
    "SQUEEZE",
    "SLEEP",
    "WAKEUP",
    "RINSE",
    "SCRUB",
    "WASH",
    "GRAB",
    "SWITCHOFF",
    "POUR",
}

ACTION_ARITY = {
    "SLEEP": 0,
    "WAKEUP": 0,
    "POUR": 2,
}

ACTION_PROP_HINTS = {
    "DRINK": {"DRINKABLE", "RECIPIENT"},
    "EAT": {"EATABLE"},
    "CUT": {"EATABLE", "CUTABLE"},
    "READ": {"READABLE"},
    "TYPE": {"HAS_SWITCH"},
    "PUSH": {"MOVABLE"},
    "PULL": {"MOVABLE"},
    "MOVE": {"MOVABLE"},
    "SQUEEZE": {"CLOTHES"},
    "GRAB": {"GRABBABLE"},
    "SWITCHOFF": {"HAS_SWITCH"},
}

ATOM_RE = re.compile(r"\b([A-Za-z_]+)\s*\(([^()]*)\)")
OBJ_RE = re.compile(r"\b[A-Za-z_]+(?:\.[0-9]+)\b")
JSON_FENCE_RE = re.compile(r"^```(?:json|python)?\s*(.*?)\s*```$", re.DOTALL)


@dataclass
class PromptInfo:
    identifier: str
    task_category: str
    task_block: str
    object_ids: set[str]
    object_props: dict[str, set[str]]
    initial_states: list[str]
    goal_states: list[str]
    action_groups: list[list[str]]
    necessity: str


def strip_wrappers(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    fence = JSON_FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    return text


def extract_json_value(text: str) -> Any | None:
    """Extract the first JSON object or array from model text."""
    text = strip_wrappers(text)
    try:
        return json.loads(text)
    except Exception:
        pass

    starts = [idx for idx, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        stack: list[str] = []
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
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    break
                opener = stack.pop()
                if (opener, ch) not in {("{", "}"), ("[", "]")}:
                    break
                if not stack:
                    candidate = text[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    return None


def parse_between(text: str, start_marker: str, end_markers: list[str]) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = len(text)
    for marker in end_markers:
        idx = text.find(marker, start)
        if idx >= 0:
            end = min(end, idx)
    return text[start:end].strip()


def clean_lines(block: str) -> list[str]:
    return [line.strip() for line in block.splitlines() if line.strip()]


def parse_prompt_item(item: dict[str, Any]) -> PromptInfo:
    prompt = item["llm_prompt"]
    marker = "Now, it is time for you to generate the subgoal plan for the following task."
    task_block = prompt.rsplit(marker, 1)[-1].strip()

    task_match = re.search(r"# Target Task:\s*Task category is\s*(.+)", task_block)
    task_category = task_match.group(1).strip() if task_match else ""

    objects_block = parse_between(task_block, "## Relevant Objects in the Scene", ["## Initial States"])
    object_ids: set[str] = set()
    object_props: dict[str, set[str]] = {}
    for line in clean_lines(objects_block):
        ids = OBJ_RE.findall(line)
        if not ids:
            continue
        obj_id = ids[0]
        object_ids.add(obj_id)
        props_match = re.search(r"\[(.*?)\]", line)
        props: set[str] = set()
        if props_match:
            props = {part.strip().strip("'\"") for part in props_match.group(1).split(",") if part.strip()}
        object_props[obj_id] = props

    initial_block = parse_between(task_block, "## Initial States", ["## Goal States"])
    initial_states = [line for line in clean_lines(initial_block) if "(" in line and ")" in line]
    for state in initial_states:
        object_ids.update(OBJ_RE.findall(state))

    goal_block = parse_between(task_block, "[States]", ["[Actions Must Include]"])
    goal_states = [line for line in clean_lines(goal_block) if "(" in line and ")" in line]
    for state in goal_states:
        object_ids.update(OBJ_RE.findall(state))

    action_block = parse_between(
        task_block,
        "[Actions Must Include]: Actions are listed in the execution order, each line is one action to satisfy. If \"A or B or ...\" is presented in one line, then only one of them needs to be satisfied.",
        ["## Necessity to Use Actions"],
    )
    action_groups: list[list[str]] = []
    for line in clean_lines(action_block):
        if line.lower() == "none":
            continue
        names = [name.upper() for name in re.findall(r"\b[A-Za-z_]+\b", line)]
        names = [name for name in names if name in ACTION_NAMES]
        if names:
            action_groups.append(names)

    necessity_block = parse_between(task_block, "## Necessity to Use Actions", ["## Output:"])
    necessity = "yes" if clean_lines(necessity_block)[:1] == ["Yes"] else "no"
    if action_groups:
        necessity = "yes"

    return PromptInfo(
        identifier=item["identifier"],
        task_category=task_category,
        task_block=task_block,
        object_ids=object_ids,
        object_props=object_props,
        initial_states=initial_states,
        goal_states=goal_states,
        action_groups=action_groups,
        necessity=necessity,
    )


def atom_key(name: str, args: list[str]) -> str:
    return f"{name.upper()}({', '.join(args)})"


def atoms_in_expr(expr: str) -> list[tuple[str, list[str], str]]:
    atoms: list[tuple[str, list[str], str]] = []
    for match in ATOM_RE.finditer(expr or ""):
        name = match.group(1).upper()
        args = [part.strip() for part in match.group(2).split(",") if part.strip()]
        atoms.append((name, args, atom_key(name, args)))
    return atoms


def atom_keys_in_output(output: list[str]) -> set[str]:
    keys: set[str] = set()
    for item in output:
        for _name, _args, key in atoms_in_expr(item):
            keys.add(key)
    return keys


def normalize_predicate_case(expr: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1).upper()
        return f"{name}("

    return re.sub(r"\b([A-Za-z_]+)\s*\(", repl, expr)


def split_safe_conjunction(expr: str) -> list[str]:
    """Split only pure conjunctions of atoms. Preserve expressions with OR."""
    if re.search(r"\bor\b", expr, flags=re.IGNORECASE):
        return [expr]
    parts = re.split(r"\band\b", expr, flags=re.IGNORECASE)
    cleaned = [part.strip() for part in parts if part.strip()]
    if len(cleaned) <= 1:
        return [expr]
    if all(len(atoms_in_expr(part)) == 1 and atoms_in_expr(part)[0][2] == normalize_predicate_case(part) for part in cleaned):
        return [normalize_predicate_case(part) for part in cleaned]
    return [expr]


def choose_object_with_props(info: PromptInfo, props: set[str]) -> str | None:
    for obj, obj_props in info.object_props.items():
        if obj != "character.65" and props.intersection(obj_props):
            return obj
    for obj in sorted(info.object_ids):
        if not obj.startswith("character."):
            return obj
    return None


def choose_action_atom(action: str, info: PromptInfo) -> str:
    action = action.upper()
    arity = ACTION_ARITY.get(action, 1)
    if arity == 0:
        return f"{action}()"

    target: str | None = None
    if action in {"LOOKAT", "WATCH"}:
        for goal in info.goal_states:
            for name, args, _key in atoms_in_expr(goal):
                if name == "FACING" and len(args) >= 2:
                    target = args[1]
                    break
            if target:
                break
        if not target:
            target = choose_object_with_props(info, {"LOOKABLE"})
    elif action == "POUR":
        source = choose_object_with_props(info, {"POURABLE", "DRINKABLE", "CREAM"})
        target = choose_object_with_props(info, {"RECIPIENT", "CONTAINERS", "CAN_OPEN"})
        if source and target and source != target:
            return f"{action}({source}, {target})"
        target = source or target
    elif action in ACTION_PROP_HINTS:
        target = choose_object_with_props(info, ACTION_PROP_HINTS[action])
    else:
        target = choose_object_with_props(info, set())

    if arity == 2:
        first = target or choose_object_with_props(info, set()) or ""
        second = ""
        for obj in sorted(info.object_ids):
            if obj != first and not obj.startswith("character."):
                second = obj
                break
        return f"{action}({first}, {second})"

    return f"{action}({target or ''})"


def insertion_index_for_action(output: list[str], action_atom: str) -> int:
    atoms = atoms_in_expr(action_atom)
    if not atoms:
        return len(output)
    action, args, _key = atoms[0]
    primary = args[0] if args else ""
    secondary = args[1] if len(args) > 1 else ""

    def item_has(index: int, names: set[str], obj: str | None = None) -> bool:
        for name, atom_args, _atom_key in atoms_in_expr(output[index]):
            if name not in names:
                continue
            if obj is None or obj in atom_args:
                return True
        return False

    if action in {"LOOKAT", "WATCH"} and primary:
        idx = -1
        for i in range(len(output)):
            if item_has(i, {"FACING"}, primary):
                idx = i
        return idx + 1 if idx >= 0 else len(output)

    if action == "POUR":
        after = 0
        if secondary:
            for i in range(len(output)):
                if item_has(i, {"OPEN"}, secondary):
                    after = i + 1
        before = len(output)
        if secondary:
            for i in range(after, len(output)):
                if item_has(i, {"CLOSED", "ON"}, secondary):
                    before = i
                    break
        return before

    if action in {"DRINK", "EAT", "READ"} and primary:
        idx = -1
        for i in range(len(output)):
            if item_has(i, {"HOLDS_RH", "HOLDS_LH"}, primary):
                idx = i
        return idx + 1 if idx >= 0 else len(output)

    if action == "SLEEP":
        idx = -1
        for i in range(len(output)):
            if item_has(i, {"SITTING", "LYING"}):
                idx = i
        return idx + 1 if idx >= 0 else len(output)

    if primary:
        idx = -1
        for i in range(len(output)):
            if item_has(i, {"FACING", "NEXT_TO", "HOLDS_RH", "HOLDS_LH"}, primary):
                idx = i
        if idx >= 0:
            return idx + 1

    return len(output)


def normalize_candidate(
    info: PromptInfo,
    raw_text: str,
    split_and: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = extract_json_value(raw_text)
    report: dict[str, Any] = {
        "parse_ok": isinstance(parsed, dict),
        "dropped_items": 0,
        "invalid_atoms": [],
        "extra_actions_removed": [],
        "missing_goals_added": [],
        "required_actions_added": [],
    }
    if not isinstance(parsed, dict):
        parsed = {}

    raw_output = parsed.get("output", [])
    if not isinstance(raw_output, list):
        raw_output = []

    normalized_items: list[str] = []
    seen_items: set[str] = set()
    allowed_required_actions = {name for group in info.action_groups for name in group}

    for raw_item in raw_output:
        if not isinstance(raw_item, str):
            report["dropped_items"] += 1
            continue
        item = normalize_predicate_case(raw_item.strip().strip("`").strip())
        item = re.sub(r"\s+", " ", item).strip()
        if not item:
            report["dropped_items"] += 1
            continue
        pieces = split_safe_conjunction(item) if split_and else [item]
        for piece in pieces:
            atoms = atoms_in_expr(piece)
            if not atoms:
                report["dropped_items"] += 1
                continue
            invalid = False
            extra_action = False
            for name, args, key in atoms:
                if name not in STATE_NAMES and name not in ACTION_NAMES:
                    report["invalid_atoms"].append(key)
                    invalid = True
                if any(arg and arg not in info.object_ids for arg in args):
                    report["invalid_atoms"].append(key)
                    invalid = True
                if name in ACTION_NAMES and name not in allowed_required_actions:
                    report["extra_actions_removed"].append(key)
                    extra_action = True
            if invalid or extra_action:
                report["dropped_items"] += 1
                continue
            if piece not in seen_items:
                seen_items.add(piece)
                normalized_items.append(piece)

    present_atoms = atom_keys_in_output(normalized_items)
    for goal in info.goal_states:
        goal_norm = normalize_predicate_case(goal)
        goal_atoms = atoms_in_expr(goal_norm)
        if goal_atoms and any(key not in present_atoms for _name, _args, key in goal_atoms):
            normalized_items.append(goal_norm)
            report["missing_goals_added"].append(goal_norm)
            for _name, _args, key in goal_atoms:
                present_atoms.add(key)

    used_required_actions: list[str] = []
    present_atoms = atom_keys_in_output(normalized_items)
    for group in info.action_groups:
        group_present = None
        for action in group:
            if any(name == action for item in normalized_items for name, _args, _key in atoms_in_expr(item)):
                group_present = action
                break
        if group_present:
            used_required_actions.append(group_present)
            continue
        chosen = next((action for action in ("LOOKAT", "WATCH") if action in group), group[0])
        action_atom = choose_action_atom(chosen, info)
        insert_at = insertion_index_for_action(normalized_items, action_atom)
        normalized_items.insert(insert_at, action_atom)
        used_required_actions.append(chosen)
        report["required_actions_added"].append(action_atom)
        for _name, _args, key in atoms_in_expr(action_atom):
            present_atoms.add(key)

    if not normalized_items:
        normalized_items = [normalize_predicate_case(goal) for goal in info.goal_states]

    result = {
        "necessity_to_use_action": "yes" if info.action_groups or info.necessity == "yes" else "no",
        "actions_to_include": used_required_actions if info.action_groups else [],
        "output": normalized_items,
    }
    return result, report


def hard_issue_count(info: PromptInfo, candidate: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    issues = {
        "bad_shape": 0,
        "missing_goals": [],
        "missing_action_groups": [],
        "invalid_atoms": [],
        "extra_actions": [],
    }
    if not isinstance(candidate, dict) or not isinstance(candidate.get("output"), list):
        issues["bad_shape"] = 1
        return 9999, issues

    output = [item for item in candidate.get("output", []) if isinstance(item, str)]
    present = atom_keys_in_output(output)
    for goal in info.goal_states:
        for _name, _args, key in atoms_in_expr(normalize_predicate_case(goal)):
            if key not in present:
                issues["missing_goals"].append(key)

    for group in info.action_groups:
        if not any(name == action for action in group for item in output for name, _args, _key in atoms_in_expr(item)):
            issues["missing_action_groups"].append(group)

    allowed_required_actions = {name for group in info.action_groups for name in group}
    for item in output:
        for name, args, key in atoms_in_expr(item):
            if name not in STATE_NAMES and name not in ACTION_NAMES:
                issues["invalid_atoms"].append(key)
            if any(arg and arg not in info.object_ids for arg in args):
                issues["invalid_atoms"].append(key)
            if name in ACTION_NAMES and name not in allowed_required_actions:
                issues["extra_actions"].append(key)

    count = (
        issues["bad_shape"] * 1000
        + len(issues["missing_goals"]) * 10
        + len(issues["missing_action_groups"]) * 10
        + len(issues["invalid_atoms"]) * 5
        + len(issues["extra_actions"]) * 3
    )
    return count, issues


def stable_json_output(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ": "))
