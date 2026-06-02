#!/usr/bin/env python3
"""Validate, repair, and rerank VirtualHome Action Sequencing outputs.

The EAI VAS prompt asks for a JSON dictionary, but repeated actions such as
WALK are common and are represented by repeated keys in practice.  This script
therefore parses outputs as ordered action pairs, performs structural repair,
and renders the final answer back to the evaluator-compatible dictionary text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from virtualhome_two_stage_planner import VirtualHomeCompiler, parse_prompt  # noqa: E402


ACTION_SPECS: dict[str, tuple[int, list[list[str]]]] = {
    "CLOSE": (1, [["CAN_OPEN"]]),
    "DRINK": (1, [["DRINKABLE", "RECIPIENT"]]),
    "FIND": (1, [[]]),
    "WALK": (1, [[]]),
    "GRAB": (1, [["GRABBABLE"]]),
    "LOOKAT": (1, [[]]),
    "OPEN": (1, [["CAN_OPEN"]]),
    "POINTAT": (1, [[]]),
    "PUTBACK": (2, [["GRABBABLE"], []]),
    "PUTIN": (2, [["GRABBABLE"], ["CAN_OPEN"]]),
    "RUN": (1, [[]]),
    "SIT": (1, [["SITTABLE"]]),
    "STANDUP": (0, []),
    "SWITCHOFF": (1, [["HAS_SWITCH"]]),
    "SWITCHON": (1, [["HAS_SWITCH"]]),
    "TOUCH": (1, [[]]),
    "TURNTO": (1, [[]]),
    "WATCH": (1, [[]]),
    "WIPE": (1, [[]]),
    "PUTON": (1, [["CLOTHES"]]),
    "PUTOFF": (1, [["CLOTHES"]]),
    "GREET": (1, [["PERSON"]]),
    "DROP": (1, [[]]),
    "READ": (1, [["READABLE"]]),
    "LIE": (1, [["LIEABLE"]]),
    "POUR": (2, [["POURABLE", "DRINKABLE"], ["RECIPIENT"]]),
    "PUSH": (1, [["MOVABLE"]]),
    "PULL": (1, [["MOVABLE"]]),
    "MOVE": (1, [["MOVABLE"]]),
    "WASH": (1, [[]]),
    "RINSE": (1, [[]]),
    "SCRUB": (1, [[]]),
    "SQUEEZE": (1, [["CLOTHES"]]),
    "PLUGIN": (1, [["HAS_PLUG"]]),
    "PLUGOUT": (1, [["HAS_PLUG"]]),
    "CUT": (1, [["EATABLE", "CUTABLE"]]),
    "EAT": (1, [["EATABLE"]]),
    "RELEASE": (1, [[]]),
    "TYPE": (1, [["HAS_SWITCH"]]),
}

ACTION_ALIASES = {
    "PUT_BACK": "PUTBACK",
    "PUT_INSIDE": "PUTIN",
    "PUT_IN": "PUTIN",
    "PUT_ON": "PUTBACK",
    "SWITCH_ON": "SWITCHON",
    "SWITCH_OFF": "SWITCHOFF",
    "PLUG_IN": "PLUGIN",
    "PLUG_OUT": "PLUGOUT",
    "TURN_TO": "TURNTO",
    "TURN": "TURNTO",
    "LOOK_AT": "LOOKAT",
    "POINT_AT": "POINTAT",
    "STAND_UP": "STANDUP",
    "WAKEUP": "STANDUP",
}

ACTION_GOAL_ALIASES = {
    "SLEEP": "LIE",
    "WAKEUP": "STANDUP",
}


@dataclass(frozen=True)
class ActionStep:
    action: str
    args: tuple[str, ...]


@dataclass
class Candidate:
    source: str
    raw_text: str
    steps: list[ActionStep] = field(default_factory=list)
    parse_error: str | None = None
    repairs: Counter[str] = field(default_factory=Counter)
    violations: Counter[str] = field(default_factory=Counter)
    checks: dict[str, Any] = field(default_factory=dict)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def parse_candidate_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"--candidate must be NAME=PATH, got: {raw}")
    name, path_text = raw.split("=", 1)
    if not name.strip():
        raise SystemExit(f"Candidate name is empty in: {raw}")
    return name.strip(), Path(path_text)


def load_candidate_rows(path: Path) -> dict[str, str]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise SystemExit(f"Candidate is not a JSON list: {path}")
    loaded: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or "identifier" not in row:
            continue
        loaded[str(row["identifier"])] = str(row.get("llm_output", ""))
    return loaded


def normalize_action(action: Any) -> str | None:
    text = str(action).strip().upper().replace(" ", "_")
    text = re.sub(r"[^A-Z0-9_\\-]", "", text)
    if text in ACTION_SPECS:
        return text
    text = ACTION_ALIASES.get(text, text)
    if text in ACTION_SPECS:
        return text
    stripped = re.sub(r"[-_]?\\d+$", "", text)
    stripped = ACTION_ALIASES.get(stripped, stripped)
    if stripped in ACTION_SPECS:
        return stripped
    compact = text.replace("_", "").replace("-", "")
    if compact in ACTION_SPECS:
        return compact
    return None


def compact_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"```(?:json|python)?", "", text, flags=re.I)
    return text.replace("```", "").strip()


def balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    stack: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append(index)
        elif char == "}" and stack:
            start = stack.pop()
            if not stack:
                objects.append(text[start : index + 1])
    return objects


def decode_pairs_from_json(raw: str) -> list[tuple[Any, Any]] | None:
    try:
        decoded = json.loads(raw, object_pairs_hook=lambda pairs: pairs)
    except Exception:
        return None
    if isinstance(decoded, list) and all(
        isinstance(item, tuple) and len(item) == 2 for item in decoded
    ):
        return list(decoded)
    if isinstance(decoded, list):
        pairs: list[tuple[Any, Any]] = []
        for item in decoded:
            if isinstance(item, list) and len(item) == 1 and isinstance(item[0], tuple):
                pairs.append(item[0])
            elif isinstance(item, list) and all(
                isinstance(part, tuple) and len(part) == 2 for part in item
            ):
                pairs.extend(item)
            elif isinstance(item, dict):
                pairs.extend(item.items())
            else:
                return None
        return pairs
    return None


PAIR_RE = re.compile(r'"([^"]+)"\s*:\s*(\[[^\]]*\])', flags=re.S)


def regex_pairs(text: str) -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    for match in PAIR_RE.finditer(text):
        try:
            value = json.loads(match.group(2))
        except Exception:
            continue
        pairs.append((match.group(1), value))
    return pairs


def coerce_step_pairs(raw_pairs: list[tuple[Any, Any]], candidate: Candidate) -> list[ActionStep]:
    steps: list[ActionStep] = []
    for raw_action, raw_args in raw_pairs:
        action = normalize_action(raw_action)
        if action is None:
            candidate.violations["invalid_action"] += 1
            continue
        if not isinstance(raw_args, list):
            candidate.violations["args_not_list"] += 1
            continue
        args = tuple(str(arg) for arg in raw_args)
        arity = ACTION_SPECS[action][0]
        needed = arity * 2
        if len(args) < needed:
            candidate.violations["too_few_args"] += 1
            continue
        if len(args) > needed:
            candidate.repairs["truncated_extra_args"] += 1
            args = args[:needed]
        steps.append(ActionStep(action, args))
    return steps


def parse_ordered_steps(text: str, source: str) -> Candidate:
    candidate = Candidate(source=source, raw_text=text)
    snippets = [compact_text(text), text]
    snippets.extend(balanced_json_objects(text))
    snippets.extend(balanced_json_objects(compact_text(text)))

    parsed_options: list[list[tuple[Any, Any]]] = []
    for snippet in snippets:
        if not snippet:
            continue
        decoded = decode_pairs_from_json(snippet)
        if decoded:
            parsed_options.append(decoded)
    regex_option = regex_pairs(text)
    if regex_option:
        parsed_options.append(regex_option)

    best_steps: list[ActionStep] = []
    best_repairs = Counter()
    best_violations = Counter()
    for option in parsed_options:
        trial = Candidate(source=source, raw_text=text)
        steps = coerce_step_pairs(option, trial)
        if len(steps) > len(best_steps):
            best_steps = steps
            best_repairs = trial.repairs
            best_violations = trial.violations

    if not best_steps:
        candidate.parse_error = "no_action_pairs"
    candidate.steps = best_steps
    candidate.repairs.update(best_repairs)
    candidate.violations.update(best_violations)
    return candidate


def object_maps(parsed: dict[str, Any]) -> tuple[dict[tuple[str, str], Any], dict[str, list[Any]], dict[str, Any]]:
    by_key = {(obj.name, str(obj.object_id)): obj for obj in parsed["objects"]}
    by_name: dict[str, list[Any]] = defaultdict(list)
    by_id: dict[str, Any] = {}
    for obj in parsed["objects"]:
        by_name[obj.name].append(obj)
        by_id[str(obj.object_id)] = obj
    return by_key, by_name, by_id


def repair_step_args(candidate: Candidate, parsed: dict[str, Any]) -> None:
    by_key, by_name, by_id = object_maps(parsed)
    repaired: list[ActionStep] = []
    for step in candidate.steps:
        arity = ACTION_SPECS[step.action][0]
        fixed: list[str] = []
        keep = True
        for index in range(arity):
            name = str(step.args[index * 2])
            object_id = str(step.args[index * 2 + 1])
            if name == "character":
                candidate.violations["character_argument"] += 1
                keep = False
                break
            obj = by_key.get((name, object_id))
            if obj is None and object_id in by_id and by_id[object_id].name != "character":
                obj = by_id[object_id]
                candidate.repairs["object_name_from_id"] += 1
            if obj is None and len(by_name.get(name, [])) == 1 and name != "character":
                obj = by_name[name][0]
                candidate.repairs["object_id_from_name"] += 1
            if obj is None:
                candidate.violations["unknown_object"] += 1
                keep = False
                break
            fixed.extend([obj.name, str(obj.object_id)])
        if keep:
            repaired.append(ActionStep(step.action, tuple(fixed)))
    candidate.steps = repaired


def drop_property_violating_steps(candidate: Candidate, parsed: dict[str, Any]) -> None:
    by_key, _, _ = object_maps(parsed)
    repaired: list[ActionStep] = []
    dropped = 0
    for step in candidate.steps:
        _arity, required = ACTION_SPECS[step.action]
        keep = True
        for index, props in enumerate(required):
            if not props:
                continue
            obj = by_key.get((step.args[index * 2], step.args[index * 2 + 1]))
            if obj is None:
                continue
            if not all(prop in obj.properties for prop in props):
                keep = False
                dropped += 1
                break
        if keep:
            repaired.append(step)
    if dropped:
        candidate.repairs["dropped_property_violating_steps"] += dropped
    candidate.steps = repaired


def property_violations(candidate: Candidate, parsed: dict[str, Any]) -> int:
    by_key, _, _ = object_maps(parsed)
    count = 0
    for step in candidate.steps:
        _arity, required = ACTION_SPECS[step.action]
        for index, props in enumerate(required):
            if not props:
                continue
            obj = by_key.get((step.args[index * 2], step.args[index * 2 + 1]))
            if obj is None:
                continue
            if not all(prop in obj.properties for prop in props):
                count += 1
    return count


def parse_action_goals(prompt: str) -> list[list[str]]:
    if "Action goals are:" not in prompt:
        return []
    block = prompt.split("Action goals are:", 1)[1]
    block = block.split("Please output", 1)[0]
    goals: list[list[str]] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("The following action"):
            continue
        if "no action requirement" in line.lower():
            continue
        if set(line) <= {"-"}:
            continue
        alternatives = []
        for part in re.split(r"\\s+or\\s+", line, flags=re.I):
            raw_action = str(part).strip().upper().replace(" ", "_")
            alternatives.append(normalize_action(ACTION_GOAL_ALIASES.get(raw_action, raw_action)))
        cleaned = [action for action in alternatives if action]
        if cleaned:
            goals.append(cleaned)
    return goals


def action_goals_satisfied(steps: list[ActionStep], goals: list[list[str]]) -> bool:
    position = -1
    actions = [step.action for step in steps]
    for alternatives in goals:
        found = None
        for index in range(position + 1, len(actions)):
            if actions[index] in alternatives:
                found = index
                break
        if found is None:
            return False
        position = found
    return True


def action_goal_repair_steps(fallback_steps: list[ActionStep], goals: list[list[str]]) -> list[ActionStep]:
    repair: list[ActionStep] = []
    start = 0
    for alternatives in goals:
        found = None
        for index in range(start, len(fallback_steps)):
            if fallback_steps[index].action in alternatives:
                found = index
                break
        if found is None:
            for index, step in enumerate(fallback_steps):
                if step.action in alternatives:
                    found = index
                    break
        if found is not None:
            repair.append(fallback_steps[found])
            start = found + 1
    return repair


def first_object_with_props(parsed: dict[str, Any], props: set[str]) -> Any | None:
    for obj in parsed["objects"]:
        if obj.name == "character":
            continue
        if props.issubset(obj.properties):
            return obj
    return None


def goal_object_for_action(parsed: dict[str, Any], action: str) -> Any | None:
    wanted_state = {
        "SWITCHON": "ON",
        "SWITCHOFF": "OFF",
        "OPEN": "OPEN",
        "CLOSE": "CLOSED",
    }.get(action)
    if wanted_state:
        for name, state in parsed["node_goals"]:
            if state == wanted_state:
                obj = first_object_by_name(parsed, name)
                if obj:
                    return obj
    if action == "GRAB":
        for src, rel, dst in parsed["edge_goals"]:
            if src == "character" and rel in {"HOLDS_RH", "HOLDS_LH"}:
                obj = first_object_by_name(parsed, dst)
                if obj:
                    return obj
            if src != "character" and rel in {"ON", "INSIDE"}:
                obj = first_object_by_name(parsed, src)
                if obj:
                    return obj
    if action in {"LOOKAT", "WATCH", "TOUCH", "TURNTO", "POINTAT"}:
        for src, rel, dst in parsed["edge_goals"]:
            if src == "character" and rel in {"FACING", "CLOSE", "NEAR"}:
                obj = first_object_by_name(parsed, dst)
                if obj:
                    return obj
    return None


def first_object_by_name(parsed: dict[str, Any], name: str) -> Any | None:
    for obj in parsed["objects"]:
        if obj.name == name and obj.name != "character":
            return obj
    return None


def choose_action_goal_step(parsed: dict[str, Any], alternatives: list[str]) -> list[ActionStep]:
    for action in alternatives:
        if action == "STANDUP":
            return [ActionStep("STANDUP", ())]
        if action == "LIE":
            obj = goal_object_for_action(parsed, action) or first_object_with_props(parsed, {"LIEABLE"})
            if obj:
                return [
                    ActionStep("WALK", (obj.name, str(obj.object_id))),
                    ActionStep("LIE", (obj.name, str(obj.object_id))),
                ]
        if action in {"SIT"}:
            obj = goal_object_for_action(parsed, action) or first_object_with_props(parsed, {"SITTABLE"})
            if obj:
                return [
                    ActionStep("WALK", (obj.name, str(obj.object_id))),
                    ActionStep("SIT", (obj.name, str(obj.object_id))),
                ]
        if action in {"SWITCHON", "SWITCHOFF", "TYPE"}:
            obj = goal_object_for_action(parsed, action) or first_object_with_props(parsed, {"HAS_SWITCH"})
            if obj:
                return [
                    ActionStep("WALK", (obj.name, str(obj.object_id))),
                    ActionStep(action, (obj.name, str(obj.object_id))),
                ]
        if action in {"OPEN", "CLOSE"}:
            obj = goal_object_for_action(parsed, action) or first_object_with_props(parsed, {"CAN_OPEN"})
            if obj:
                return [
                    ActionStep("WALK", (obj.name, str(obj.object_id))),
                    ActionStep(action, (obj.name, str(obj.object_id))),
                ]
        if action in {"READ"}:
            obj = first_object_with_props(parsed, {"READABLE"})
            if obj:
                return [
                    ActionStep("WALK", (obj.name, str(obj.object_id))),
                    ActionStep(action, (obj.name, str(obj.object_id))),
                ]
        if action in {"DRINK"}:
            obj = first_object_with_props(parsed, {"DRINKABLE", "RECIPIENT"})
            if obj:
                return [
                    ActionStep("WALK", (obj.name, str(obj.object_id))),
                    ActionStep(action, (obj.name, str(obj.object_id))),
                ]
        if action in {"WASH", "RINSE", "SCRUB", "WIPE"}:
            obj = goal_object_for_action(parsed, action)
            if obj is None:
                obj = next((item for item in parsed["objects"] if item.name != "character"), None)
            if obj:
                return [
                    ActionStep("WALK", (obj.name, str(obj.object_id))),
                    ActionStep(action, (obj.name, str(obj.object_id))),
                ]
        if action in {"PUSH", "PULL", "MOVE"}:
            obj = first_object_with_props(parsed, {"MOVABLE"})
            if obj:
                return [
                    ActionStep("WALK", (obj.name, str(obj.object_id))),
                    ActionStep(action, (obj.name, str(obj.object_id))),
                ]
        if action == "GRAB":
            obj = goal_object_for_action(parsed, action) or first_object_with_props(parsed, {"GRABBABLE"})
            if obj:
                return [
                    ActionStep("WALK", (obj.name, str(obj.object_id))),
                    ActionStep("GRAB", (obj.name, str(obj.object_id))),
                ]
        if action == "POUR":
            src = first_object_with_props(parsed, {"POURABLE", "DRINKABLE"})
            dst = first_object_with_props(parsed, {"RECIPIENT"})
            if src is None:
                src = first_object_with_props(parsed, {"GRABBABLE"}) or next(
                    (item for item in parsed["objects"] if item.name != "character"),
                    None,
                )
            if dst is None:
                dst = goal_object_for_action(parsed, action) or next(
                    (item for item in parsed["objects"] if item.name != "character" and item != src),
                    None,
                )
            if src and dst:
                return [
                    ActionStep("WALK", (src.name, str(src.object_id))),
                    ActionStep("GRAB", (src.name, str(src.object_id))),
                    ActionStep("WALK", (dst.name, str(dst.object_id))),
                    ActionStep("POUR", (src.name, str(src.object_id), dst.name, str(dst.object_id))),
                ]
    return []


def synthesize_action_goal_prefix(prompt: str) -> list[ActionStep]:
    parsed = parse_prompt(prompt)
    steps: list[ActionStep] = []
    for alternatives in parse_action_goals(prompt):
        current = steps
        if current and action_goals_satisfied(current, [alternatives]):
            continue
        steps.extend(choose_action_goal_step(parsed, alternatives))
    return steps


def build_fallback_steps(prompt: str) -> list[ActionStep]:
    parsed = parse_prompt(prompt)
    raw_actions = VirtualHomeCompiler(parsed).compile()
    steps: list[ActionStep] = []
    for action, params in raw_actions:
        normalized = normalize_action(action)
        if normalized is None:
            continue
        if len(params) < ACTION_SPECS[normalized][0] * 2:
            continue
        steps.append(ActionStep(normalized, tuple(params)))
    return steps


def render_steps(steps: list[ActionStep]) -> str:
    lines = ["{"]
    for index, step in enumerate(steps):
        suffix = "," if index < len(steps) - 1 else ""
        lines.append(
            f'  "{step.action}": {json.dumps(list(step.args), ensure_ascii=False)}{suffix}'
        )
    lines.append("}")
    return "\n".join(lines)


class MiniSimulator:
    def __init__(self, parsed: dict[str, Any]):
        self.parsed = parsed
        self.by_key, self.by_name, _by_id = object_maps(parsed)
        self.states: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.locations: dict[tuple[str, str], tuple[str, tuple[str, str]]] = {}
        self.near: set[tuple[str, str]] = set()
        self.held: set[tuple[str, str]] = set()
        self.facing: set[tuple[str, str]] = set()
        self.char_states: set[str] = set()
        self.char_inside: tuple[str, str] | None = None
        self.char_on: tuple[str, str] | None = None

        for obj_name, states, _props in parsed["nodes"]:
            if obj_name == "character":
                self.char_states.update(states)
                continue
            for obj in self.by_name.get(obj_name, []):
                self.states[obj.key].update(states)

        for src_name, src_id, rel, dst_name, dst_id in parsed["edges"]:
            src = self.by_key.get((src_name, str(src_id)))
            dst = self.by_key.get((dst_name, str(dst_id)))
            if not src or not dst:
                continue
            if src.name == "character":
                if rel in {"NEAR", "CLOSE"}:
                    self.near.add(dst.key)
                elif rel == "FACING":
                    self.facing.add(dst.key)
                elif rel in {"HOLDS_RH", "HOLDS_LH"}:
                    self.held.add(dst.key)
                elif rel == "INSIDE":
                    self.char_inside = dst.key
                elif rel == "ON":
                    self.char_on = dst.key
            elif rel in {"INSIDE", "ON"}:
                self.locations[src.key] = (rel, dst.key)

    def first(self, name: str) -> Any | None:
        values = self.by_name.get(name, [])
        return values[0] if values else None

    def key_for_args(self, args: tuple[str, ...], index: int = 0) -> tuple[str, str] | None:
        if len(args) <= index * 2 + 1:
            return None
        key = (args[index * 2], args[index * 2 + 1])
        return key if key in self.by_key else None

    def apply(self, step: ActionStep) -> None:
        key0 = self.key_for_args(step.args, 0)
        key1 = self.key_for_args(step.args, 1)
        action = step.action
        if action in {"WALK", "RUN", "FIND"} and key0:
            self.char_states.discard("SITTING")
            self.char_states.discard("LYING")
            self.near.add(key0)
            if self.by_key[key0].name.endswith("_room") or self.by_key[key0].name in {
                "bathroom",
                "bedroom",
                "kitchen",
                "living_room",
                "dining_room",
                "home_office",
            }:
                self.char_inside = key0
            return
        if action == "STANDUP":
            self.char_states.discard("SITTING")
            self.char_states.discard("LYING")
            self.char_on = None
            return
        if action == "OPEN" and key0:
            self.states[key0].discard("CLOSED")
            self.states[key0].add("OPEN")
        elif action == "CLOSE" and key0:
            self.states[key0].discard("OPEN")
            self.states[key0].add("CLOSED")
        elif action == "SWITCHON" and key0:
            self.states[key0].discard("OFF")
            self.states[key0].add("ON")
        elif action == "SWITCHOFF" and key0:
            self.states[key0].discard("ON")
            self.states[key0].add("OFF")
        elif action == "PLUGIN" and key0:
            self.states[key0].discard("PLUGGED_OUT")
            self.states[key0].add("PLUGGED_IN")
        elif action == "PLUGOUT" and key0:
            self.states[key0].discard("PLUGGED_IN")
            self.states[key0].add("PLUGGED_OUT")
        elif action == "GRAB" and key0:
            self.held.add(key0)
            self.locations.pop(key0, None)
        elif action in {"PUTBACK", "POUR"} and key0 and key1:
            self.held.discard(key0)
            self.locations[key0] = ("ON", key1)
        elif action == "PUTIN" and key0 and key1:
            self.held.discard(key0)
            self.locations[key0] = ("INSIDE", key1)
        elif action == "DROP" and key0:
            self.held.discard(key0)
        elif action == "PUTON" and key0:
            self.held.discard(key0)
            self.locations[key0] = ("ON", ("character", "character"))
        elif action == "SIT" and key0:
            self.char_states.discard("LYING")
            self.char_states.add("SITTING")
            self.char_on = key0
        elif action == "LIE" and key0:
            self.char_states.discard("SITTING")
            self.char_states.add("LYING")
            self.char_on = key0
        elif action in {"TURNTO", "LOOKAT", "WATCH", "POINTAT", "TOUCH"} and key0:
            self.facing.add(key0)
            self.near.add(key0)
        elif action in {"WIPE", "WASH", "RINSE", "SCRUB"} and key0:
            self.states[key0].add("CLEAN")

    def object_goal_satisfied(self, name: str, state: str) -> bool:
        if name == "character":
            return state in self.char_states
        return any(state in self.states.get(obj.key, set()) for obj in self.by_name.get(name, []))

    def edge_goal_satisfied(self, src_name: str, rel: str, dst_name: str) -> bool:
        dst_values = self.by_name.get(dst_name, [])
        if not dst_values:
            return False
        dst_keys = {obj.key for obj in dst_values}
        if src_name == "character":
            if rel in {"NEAR", "CLOSE"}:
                return bool(self.near & dst_keys)
            if rel == "FACING":
                return bool(self.facing & dst_keys)
            if rel in {"HOLDS_RH", "HOLDS_LH"}:
                return bool(self.held & dst_keys)
            if rel == "INSIDE":
                return self.char_inside in dst_keys
            if rel == "ON":
                return self.char_on in dst_keys
            return False
        src_values = self.by_name.get(src_name, [])
        for src in src_values:
            loc = self.locations.get(src.key)
            if loc and loc[0] == rel and loc[1] in dst_keys:
                return True
        return False

    def score_goals(self) -> tuple[int, int]:
        total = 0
        passed = 0
        for name, state in self.parsed["node_goals"]:
            total += 1
            if self.object_goal_satisfied(name, state):
                passed += 1
        for src, rel, dst in self.parsed["edge_goals"]:
            total += 1
            if self.edge_goal_satisfied(src, rel, dst):
                passed += 1
        return passed, total


def validate_candidate(candidate: Candidate, prompt: str, property_policy: str = "drop") -> None:
    parsed = parse_prompt(prompt)
    repair_step_args(candidate, parsed)
    if property_policy == "drop":
        drop_property_violating_steps(candidate, parsed)
    prop_bad = property_violations(candidate, parsed)
    if prop_bad:
        candidate.violations["property_precondition"] += prop_bad
    goals = parse_action_goals(prompt)
    candidate.checks["action_goals_total"] = len(goals)
    candidate.checks["action_goals_ok"] = action_goals_satisfied(candidate.steps, goals)
    simulator = MiniSimulator(parsed)
    for step in candidate.steps:
        simulator.apply(step)
    goal_passed, goal_total = simulator.score_goals()
    candidate.checks["goal_passed"] = goal_passed
    candidate.checks["goal_total"] = goal_total
    candidate.checks["node_edge_goals_ok"] = goal_total == 0 or goal_passed == goal_total
    candidate.checks["nonempty"] = bool(candidate.steps)


def structurally_ok(candidate: Candidate, allow_property_violations: bool = False) -> bool:
    severe = (
        candidate.parse_error
        or not candidate.steps
        or candidate.violations.get("invalid_action", 0)
        or candidate.violations.get("args_not_list", 0)
        or candidate.violations.get("too_few_args", 0)
        or candidate.violations.get("character_argument", 0)
        or candidate.violations.get("unknown_object", 0)
        or (
            not allow_property_violations
            and candidate.violations.get("property_precondition", 0)
        )
    )
    return not severe


def candidate_score(
    candidate: Candidate,
    source_rank: int,
    allow_property_violations: bool = False,
) -> float:
    score = 1000.0 if structurally_ok(candidate, allow_property_violations) else 0.0
    if candidate.checks.get("action_goals_ok"):
        score += 500.0
    if candidate.checks.get("node_edge_goals_ok"):
        score += 120.0
    score += 25.0 * candidate.checks.get("goal_passed", 0)
    score -= 15.0 * sum(candidate.violations.values())
    score -= 0.03 * len(candidate.steps)
    score -= min(source_rank, 10)
    return score


def optimize_row(
    identifier: str,
    prompt: str,
    candidate_texts: list[tuple[str, str]],
    source_ranks: dict[str, int],
    selection_mode: str = "source_priority",
    property_policy: str = "drop",
) -> tuple[dict[str, str], dict[str, Any]]:
    parsed_candidates: list[Candidate] = []
    fallback_steps = build_fallback_steps(prompt)
    synthesized_goal_steps = synthesize_action_goal_prefix(prompt)
    goals = parse_action_goals(prompt)
    allow_property_violations = property_policy == "warn"

    for source, text in candidate_texts:
        candidate = parse_ordered_steps(text, source)
        validate_candidate(candidate, prompt, property_policy)
        if (
            structurally_ok(candidate, allow_property_violations)
            and goals
            and not candidate.checks.get("action_goals_ok")
        ):
            repair_steps = action_goal_repair_steps(fallback_steps, goals)
            if not repair_steps:
                repair_steps = synthesized_goal_steps
            if repair_steps:
                repaired = Candidate(
                    source=f"{source}+action_goal_repair",
                    raw_text=text,
                    steps=list(candidate.steps) + repair_steps,
                )
                repaired.repairs.update(candidate.repairs)
                repaired.repairs["appended_action_goal_steps"] += len(repair_steps)
                validate_candidate(repaired, prompt, property_policy)
                if repaired.checks.get("action_goals_ok"):
                    candidate = repaired
        if (
            structurally_ok(candidate, allow_property_violations)
            and not candidate.checks.get("node_edge_goals_ok")
        ):
            repaired = Candidate(
                source=f"{candidate.source}+goal_suffix",
                raw_text=text,
                steps=list(candidate.steps) + list(fallback_steps),
            )
            repaired.repairs.update(candidate.repairs)
            repaired.repairs["appended_goal_suffix_steps"] += len(fallback_steps)
            validate_candidate(repaired, prompt, property_policy)
            if (
                repaired.checks.get("action_goals_ok", True)
                and repaired.checks.get("node_edge_goals_ok")
            ):
                candidate = repaired
        parsed_candidates.append(candidate)

    fallback = Candidate(source="compiler_fallback", raw_text="", steps=fallback_steps)
    fallback.repairs["compiler_fallback"] += 1
    validate_candidate(fallback, prompt, property_policy)
    if goals and not fallback.checks.get("action_goals_ok") and synthesized_goal_steps:
        repaired_fallback = Candidate(
            source="compiler_fallback+action_goal_prefix",
            raw_text="",
            steps=list(synthesized_goal_steps) + list(fallback_steps),
        )
        repaired_fallback.repairs["compiler_fallback"] += 1
        repaired_fallback.repairs["synthesized_action_goal_steps"] += len(synthesized_goal_steps)
        validate_candidate(repaired_fallback, prompt, property_policy)
        if repaired_fallback.checks.get("action_goals_ok"):
            fallback = repaired_fallback
    parsed_candidates.append(fallback)

    preferred = [
        candidate
        for candidate in parsed_candidates
        if candidate.source != "compiler_fallback"
        and structurally_ok(candidate, allow_property_violations)
        and candidate.checks.get("action_goals_ok")
        and candidate.checks.get("node_edge_goals_ok")
    ]
    if preferred:
        if selection_mode == "score":
            best = max(
                preferred,
                key=lambda item: candidate_score(
                    item,
                    source_ranks.get(item.source.split("+", 1)[0], 999),
                    allow_property_violations,
                ),
            )
        else:
            best = min(
                preferred,
                key=lambda item: source_ranks.get(item.source.split("+", 1)[0], 999),
            )
    else:
        best = max(
            parsed_candidates,
            key=lambda item: candidate_score(
                item,
                source_ranks.get(item.source.split("+", 1)[0], 999),
                allow_property_violations,
            ),
        )
    if (
        not best.checks.get("action_goals_ok", True)
        or not best.checks.get("node_edge_goals_ok", True)
    ) and fallback.checks.get("action_goals_ok", True):
        best = fallback

    return (
        {"identifier": identifier, "llm_output": render_steps(best.steps)},
        {
            "identifier": identifier,
            "selected_source": best.source,
            "steps": len(best.steps),
            "parse_error": best.parse_error,
            "repairs": dict(best.repairs),
            "violations": dict(best.violations),
            "checks": best.checks,
            "candidate_summaries": [
                {
                    "source": candidate.source,
                    "steps": len(candidate.steps),
                    "parse_error": candidate.parse_error,
                    "repairs": dict(candidate.repairs),
                    "violations": dict(candidate.violations),
                    "checks": candidate.checks,
                    "score": candidate_score(
                        candidate,
                        source_ranks.get(candidate.source.split("+", 1)[0], 999),
                        allow_property_violations,
                    ),
                }
                for candidate in parsed_candidates
            ],
        },
    )


def summarize(report_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(report_rows),
        "selected_sources": Counter(),
        "rows_with_parse_error": 0,
        "rows_with_violations": 0,
        "rows_with_repairs": 0,
        "rows_action_goals_failed": 0,
        "rows_node_edge_goals_failed": 0,
        "total_repairs": Counter(),
        "total_violations": Counter(),
        "examples": defaultdict(list),
    }
    for row in report_rows:
        source = row["selected_source"]
        summary["selected_sources"][source] += 1
        repairs = Counter(row.get("repairs", {}))
        violations = Counter(row.get("violations", {}))
        summary["total_repairs"].update(repairs)
        summary["total_violations"].update(violations)
        if row.get("parse_error"):
            summary["rows_with_parse_error"] += 1
            if len(summary["examples"]["parse_error"]) < 10:
                summary["examples"]["parse_error"].append(row["identifier"])
        if repairs:
            summary["rows_with_repairs"] += 1
        if violations:
            summary["rows_with_violations"] += 1
            if len(summary["examples"]["violations"]) < 10:
                summary["examples"]["violations"].append(row["identifier"])
        checks = row.get("checks", {})
        if not checks.get("action_goals_ok", True):
            summary["rows_action_goals_failed"] += 1
            if len(summary["examples"]["action_goals_failed"]) < 10:
                summary["examples"]["action_goals_failed"].append(row["identifier"])
        if not checks.get("node_edge_goals_ok", True):
            summary["rows_node_edge_goals_failed"] += 1
            if len(summary["examples"]["node_edge_goals_failed"]) < 10:
                summary["examples"]["node_edge_goals_failed"].append(row["identifier"])

    return json.loads(
        json.dumps(
            {
                **summary,
                "selected_sources": dict(summary["selected_sources"]),
                "total_repairs": dict(summary["total_repairs"]),
                "total_violations": dict(summary["total_violations"]),
                "examples": dict(summary["examples"]),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("llm_prompts/virtualhome_action_sequencing_prompts.json"),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate output in NAME=PATH form. Earlier candidates have priority when valid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vas_optimized/virtualhome_action_sequencing_outputs.json"),
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["source_priority", "score"],
        default="source_priority",
        help="How to choose between candidates that pass local validation.",
    )
    parser.add_argument(
        "--property-policy",
        choices=["drop", "warn"],
        default="drop",
        help="Whether to drop property-precondition-violating steps or keep them as warnings.",
    )
    args = parser.parse_args()

    prompts = read_json(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    if not args.candidate:
        raise SystemExit("At least one --candidate NAME=PATH is required.")

    candidate_specs = [parse_candidate_arg(raw) for raw in args.candidate]
    candidate_rows = [(name, load_candidate_rows(path)) for name, path in candidate_specs]
    source_ranks = {name: index for index, (name, _path) in enumerate(candidate_specs)}

    output_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, Any]] = []
    for prompt_row in prompts:
        identifier = str(prompt_row["identifier"])
        prompt = str(prompt_row["llm_prompt"])
        texts = [
            (name, rows[identifier])
            for name, rows in candidate_rows
            if identifier in rows and rows[identifier].strip()
        ]
        output_row, report_row = optimize_row(
            identifier,
            prompt,
            texts,
            source_ranks,
            selection_mode=args.selection_mode,
            property_policy=args.property_policy,
        )
        output_rows.append(output_row)
        report_rows.append(report_row)

    report = {
        "prompt_path": str(args.prompts),
        "output_path": str(args.output),
        "selection_mode": args.selection_mode,
        "property_policy": args.property_policy,
        "candidates": [{"name": name, "path": str(path)} for name, path in candidate_specs],
        "summary": summarize(report_rows),
        "rows": report_rows,
    }
    write_json(args.output, output_rows)
    write_json(args.report or args.output.with_name("validation_report.json"), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
