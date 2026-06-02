#!/usr/bin/env python3
"""Evaluator-guided VAS v6 repairs.

This is a narrow post-repair layer on top of the v5 optimizer output.  It keeps
the v5 candidate selection intact, then replaces a few recurring VirtualHome
patterns with shorter sequences that satisfy the official executor's immediate
preconditions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from optimize_virtualhome_action_sequencing import (  # noqa: E402
    ActionStep,
    MiniSimulator,
    action_goals_satisfied,
    parse_action_goals,
    parse_ordered_steps,
    render_steps,
    validate_candidate,
    Candidate,
)
from virtualhome_two_stage_planner import VHObject, parse_prompt  # noqa: E402


ROOM_NAMES = {
    "bathroom",
    "bedroom",
    "kitchen",
    "living_room",
    "dining_room",
    "home_office",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def obj(parsed: dict[str, Any], *names: str) -> VHObject | None:
    wanted = set(names)
    for item in parsed["objects"]:
        if item.name in wanted:
            return item
    return None


def first_with_prop(parsed: dict[str, Any], prop: str) -> VHObject | None:
    for item in parsed["objects"]:
        if item.name != "character" and prop in item.properties:
            return item
    return None


def object_states(parsed: dict[str, Any], item: VHObject | None) -> set[str]:
    if item is None:
        return set()
    states: set[str] = set()
    for name, raw_states, _props in parsed["nodes"]:
        if name == item.name:
            states.update(raw_states)
    return states


def character_states(parsed: dict[str, Any]) -> set[str]:
    states: set[str] = set()
    for name, raw_states, _props in parsed["nodes"]:
        if name == "character":
            states.update(raw_states)
    return states


def current_room(parsed: dict[str, Any]) -> str | None:
    for src_name, _src_id, rel, dst_name, _dst_id in parsed["edges"]:
        if src_name == "character" and rel == "INSIDE":
            return dst_name
    return None


def room_objects(parsed: dict[str, Any]) -> list[VHObject]:
    return [item for item in parsed["objects"] if item.name in ROOM_NAMES or item.name.endswith("_room")]


def likely_target_room(parsed: dict[str, Any]) -> VHObject | None:
    current = current_room(parsed)
    for item in room_objects(parsed):
        if item.name != current:
            return item
    return obj(parsed, current) if current else None


def object_by_key(parsed: dict[str, Any], key: tuple[str, str]) -> VHObject | None:
    for item in parsed["objects"]:
        if item.name == key[0] and str(item.object_id) == str(key[1]):
            return item
    return None


def related_holder(parsed: dict[str, Any], item: VHObject | None) -> VHObject | None:
    if item is None:
        return None
    for src_name, src_id, rel, dst_name, dst_id in parsed["edges"]:
        if src_name == item.name and str(src_id) == str(item.object_id) and rel in {"INSIDE", "ON"}:
            return object_by_key(parsed, (dst_name, str(dst_id)))
        if dst_name == item.name and str(dst_id) == str(item.object_id) and rel == "NEAR":
            holder = object_by_key(parsed, (src_name, str(src_id)))
            if holder and ("SURFACES" in holder.properties or "CAN_OPEN" in holder.properties):
                return holder
    return None


def add_standup_if_needed(parsed: dict[str, Any], steps: list[ActionStep]) -> None:
    if character_states(parsed) & {"SITTING", "LYING"}:
        steps.append(ActionStep("STANDUP", ()))


def walk_to(item: VHObject | None, steps: list[ActionStep]) -> None:
    if item is not None and item.name != "character":
        steps.append(ActionStep("WALK", (item.name, str(item.object_id))))


def open_if_needed(parsed: dict[str, Any], item: VHObject | None, steps: list[ActionStep]) -> None:
    if item is None or "CAN_OPEN" not in item.properties:
        return
    if "OPEN" not in object_states(parsed, item):
        walk_to(item, steps)
        steps.append(ActionStep("OPEN", (item.name, str(item.object_id))))


def plugin_if_needed(parsed: dict[str, Any], item: VHObject | None, steps: list[ActionStep]) -> None:
    # The prompt action list mentions plug actions, but the official VAS
    # evaluator's valid action table does not accept PLUGIN/PLUGOUT.  Emitting
    # them improves our proxy state but causes hallucination errors officially.
    return


def grab_with_access(parsed: dict[str, Any], item: VHObject | None, steps: list[ActionStep]) -> None:
    if item is None:
        return
    holder = related_holder(parsed, item)
    if holder and "CAN_OPEN" in holder.properties:
        open_if_needed(parsed, holder, steps)
    walk_to(holder or item, steps)
    steps.append(ActionStep("GRAB", (item.name, str(item.object_id))))


def template_watch(parsed: dict[str, Any]) -> list[ActionStep] | None:
    target = obj(parsed, "television")
    if target is None:
        return None
    wants_tv = any(name == target.name for name, _state in parsed["node_goals"]) or any(
        dst == target.name and src == "character" and rel == "FACING"
        for src, rel, dst in parsed["edge_goals"]
    )
    wants_action = any(any(action.upper() in {"LOOKAT", "WATCH"} for action in group) for group in parsed["action_goals"])
    if not (wants_tv or wants_action):
        return None
    steps: list[ActionStep] = []
    add_standup_if_needed(parsed, steps)
    walk_to(target, steps)
    plugin_if_needed(parsed, target, steps)
    if ("ON" not in object_states(parsed, target)) and "HAS_SWITCH" in target.properties:
        walk_to(target, steps)
        steps.append(ActionStep("SWITCHON", (target.name, str(target.object_id))))
    steps.append(ActionStep("TURNTO", (target.name, str(target.object_id))))
    if wants_action:
        steps.append(ActionStep("WATCH", (target.name, str(target.object_id))))
    return steps


def template_drink(parsed: dict[str, Any]) -> list[ActionStep] | None:
    wants_drink = any(any(action.upper() == "DRINK" for action in group) for group in parsed["action_goals"])
    if not wants_drink:
        return None
    recipient = None
    for src, rel, dst in parsed["edge_goals"]:
        if src == "character" and rel.startswith("HOLDS"):
            recipient = obj(parsed, dst)
            break
    if recipient is None:
        recipient = first_with_prop(parsed, "RECIPIENT")
    if recipient is None:
        return None
    water = obj(parsed, "water")
    steps: list[ActionStep] = []
    add_standup_if_needed(parsed, steps)
    holder = related_holder(parsed, recipient)
    if water is not None and holder is not None and "CAN_OPEN" in holder.properties:
        grab_with_access(parsed, water, steps)
        open_if_needed(parsed, holder, steps)
        walk_to(holder or recipient, steps)
        steps.append(
            ActionStep(
                "POUR",
                (water.name, str(water.object_id), recipient.name, str(recipient.object_id)),
            )
        )
    grab_with_access(parsed, recipient, steps)
    steps.append(ActionStep("DRINK", (recipient.name, str(recipient.object_id))))
    return steps


def template_food_inside_container(parsed: dict[str, Any]) -> list[ActionStep] | None:
    food = obj(parsed, "food_food")
    target = None
    for src, rel, dst in parsed["edge_goals"]:
        if src == "food_food" and rel == "INSIDE":
            target = obj(parsed, dst)
            break
    if food is None or target is None or "CAN_OPEN" not in target.properties:
        return None
    steps: list[ActionStep] = []
    add_standup_if_needed(parsed, steps)
    open_if_needed(parsed, target, steps)
    grab_with_access(parsed, food, steps)
    walk_to(target, steps)
    steps.append(ActionStep("PUTIN", (food.name, str(food.object_id), target.name, str(target.object_id))))
    if ("ON" in [state for name, state in parsed["node_goals"] if name == target.name]) and "HAS_SWITCH" in target.properties:
        steps.append(ActionStep("SWITCHON", (target.name, str(target.object_id))))
    return steps


def template_light(parsed: dict[str, Any]) -> list[ActionStep] | None:
    target = None
    for name, state in parsed["node_goals"]:
        if state == "ON":
            candidate = obj(parsed, name)
            if candidate and "HAS_SWITCH" in candidate.properties:
                target = candidate
                break
    if target is None or target.name not in {"light", "floor_lamp", "lamp"}:
        return None
    steps: list[ActionStep] = []
    add_standup_if_needed(parsed, steps)
    room = likely_target_room(parsed)
    if room is not None:
        walk_to(room, steps)
    walk_to(target, steps)
    plugin_if_needed(parsed, target, steps)
    if "OFF" in object_states(parsed, target) or (target.name, "ON") in parsed["node_goals"]:
        walk_to(target, steps)
        steps.append(ActionStep("SWITCHON", (target.name, str(target.object_id))))
    return steps


def template_read(parsed: dict[str, Any]) -> list[ActionStep] | None:
    wants_read = any(any(action.upper() == "READ" for action in group) for group in parsed["action_goals"])
    target = first_with_prop(parsed, "READABLE")
    if not wants_read or target is None:
        return None
    steps: list[ActionStep] = []
    add_standup_if_needed(parsed, steps)
    grab_with_access(parsed, target, steps)
    steps.append(ActionStep("READ", (target.name, str(target.object_id))))
    return steps


def template_phone(parsed: dict[str, Any]) -> list[ActionStep] | None:
    phone = None
    for src, rel, dst in parsed["edge_goals"]:
        if src == "character" and rel.startswith("HOLDS") and dst == "phone":
            phone = obj(parsed, "phone")
            break
    if phone is None:
        return None
    steps: list[ActionStep] = []
    add_standup_if_needed(parsed, steps)
    holder = related_holder(parsed, phone)
    walk_to(holder or phone, steps)
    steps.append(ActionStep("GRAB", (phone.name, str(phone.object_id))))
    return steps


def collapse_repeats(steps: list[ActionStep]) -> list[ActionStep]:
    repaired: list[ActionStep] = []
    for step in steps:
        if repaired and repaired[-1] == step:
            continue
        if step.action in {"TURNTO", "LOOKAT"} and any(prev == step for prev in repaired[-3:]):
            continue
        repaired.append(step)
    return repaired


def prefix_goal_trim(steps: list[ActionStep], prompt: str) -> list[ActionStep]:
    parsed = parse_prompt(prompt)
    goals = parse_action_goals(prompt)
    simulator = MiniSimulator(parsed)
    best = steps
    for index, step in enumerate(steps):
        simulator.apply(step)
        passed, total = simulator.score_goals()
        prefix = steps[: index + 1]
        if not any(step.args for step in prefix):
            continue
        if (total == 0 or passed == total) and action_goals_satisfied(prefix, goals):
            best = prefix
            break
    return best


def candidate_ok(steps: list[ActionStep], prompt: str) -> bool:
    candidate = Candidate(source="v6_template", raw_text="", steps=list(steps))
    validate_candidate(candidate, prompt, property_policy="warn")
    return bool(
        candidate.steps
        and candidate.checks.get("action_goals_ok", True)
        and candidate.checks.get("node_edge_goals_ok", True)
        and not candidate.parse_error
        and not candidate.violations.get("invalid_action")
        and not candidate.violations.get("unknown_object")
        and not candidate.violations.get("too_few_args")
    )


def repair_steps(prompt: str, base_steps: list[ActionStep]) -> tuple[list[ActionStep], str]:
    parsed = parse_prompt(prompt)
    for name, builder in [
        ("watch_tv_template", template_watch),
        ("drink_template", template_drink),
        ("food_inside_container_template", template_food_inside_container),
        ("light_template", template_light),
        ("read_template", template_read),
        ("phone_template", template_phone),
    ]:
        steps = builder(parsed)
        if steps and candidate_ok(steps, prompt):
            return collapse_repeats(steps), name

    trimmed = prefix_goal_trim(collapse_repeats(base_steps), prompt)
    if len(trimmed) < len(base_steps) and candidate_ok(trimmed, prompt):
        return trimmed, "prefix_goal_trim"
    collapsed = collapse_repeats(base_steps)
    collapsed = [step for step in collapsed if step.action not in {"PLUGIN", "PLUGOUT"}]
    if len(collapsed) < len(base_steps):
        return collapsed, "collapse_repeats"
    return base_steps, "unchanged"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("llm_prompts/virtualhome_action_sequencing_prompts.json"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    prompts = {str(row["identifier"]): str(row["llm_prompt"]) for row in read_json(args.prompts)}
    input_rows = read_json(args.input)
    output_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for row in input_rows:
        identifier = str(row["identifier"])
        prompt = prompts[identifier]
        parsed = parse_ordered_steps(str(row.get("llm_output", "")), "v5")
        repaired_steps, repair = repair_steps(prompt, parsed.steps)
        counts[repair] += 1
        output_rows.append({"identifier": identifier, "llm_output": render_steps(repaired_steps)})
        report_rows.append(
            {
                "identifier": identifier,
                "repair": repair,
                "before_steps": len(parsed.steps),
                "after_steps": len(repaired_steps),
            }
        )

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "summary": {"rows": len(output_rows), "repairs": dict(counts)},
        "rows": report_rows,
    }
    write_json(args.output, output_rows)
    write_json(args.report or args.output.with_name("v6_repair_report.json"), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
