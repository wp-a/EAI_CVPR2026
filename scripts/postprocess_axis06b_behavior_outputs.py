#!/usr/bin/env python3
"""Postprocess and validate compact AxisTilted2 BEHAVIOR GI/SD/AS outputs."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any


TASK_FILES = {
    "gi": "behavior_goal_interpretation_outputs.json",
    "sd": "behavior_subgoal_decomposition_outputs.json",
    "as": "behavior_action_sequencing_outputs.json",
}
PROMPT_FILES = {task: filename.replace("_outputs.json", "_prompts.json") for task, filename in TASK_FILES.items()}

NODE_CANON = {
    "cooked": "Cooked",
    "open": "Open",
    "frozen": "Frozen",
    "dusty": "Dusty",
    "stained": "Stained",
    "sliced": "Sliced",
    "soaked": "Soaked",
    "toggled_on": "Toggled_On",
    "toggledon": "Toggled_On",
    "burnt": "Burnt",
}
EDGE_CANON = {
    "inside": "inside",
    "ontop": "ontop",
    "on": "ontop",
    "onfloor": "onfloor",
    "nextto": "nextto",
    "next_to": "nextto",
    "touching": "touching",
    "under": "under",
}
SD_PREDICATES = {
    "inside",
    "ontop",
    "nextto",
    "under",
    "onfloor",
    "touching",
    "cooked",
    "burnt",
    "dusty",
    "frozen",
    "open",
    "sliced",
    "soaked",
    "stained",
    "toggledon",
    "toggled_on",
    "holds_rh",
    "holds_lh",
    "holding",
}
VALID_ACTIONS = {
    "LEFT_GRASP",
    "RIGHT_GRASP",
    "LEFT_PLACE_ONTOP",
    "RIGHT_PLACE_ONTOP",
    "LEFT_PLACE_INSIDE",
    "RIGHT_PLACE_INSIDE",
    "RIGHT_RELEASE",
    "LEFT_RELEASE",
    "OPEN",
    "CLOSE",
    "COOK",
    "CLEAN",
    "FREEZE",
    "UNFREEZE",
    "SLICE",
    "SOAK",
    "DRY",
    "TOGGLE_ON",
    "TOGGLE_OFF",
    "LEFT_PLACE_NEXTTO",
    "RIGHT_PLACE_NEXTTO",
    "LEFT_TRANSFER_CONTENTS_INSIDE",
    "RIGHT_TRANSFER_CONTENTS_INSIDE",
    "LEFT_TRANSFER_CONTENTS_ONTOP",
    "RIGHT_TRANSFER_CONTENTS_ONTOP",
    "LEFT_PLACE_NEXTTO_ONTOP",
    "RIGHT_PLACE_NEXTTO_ONTOP",
    "LEFT_PLACE_UNDER",
    "RIGHT_PLACE_UNDER",
}
AS_EXAMPLE_OBJECTS = {"sink_7", "rag_0", "bucket_0", "bathtub_4", "cabinet_1"}
OBJECT_REF_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*(?:[._][0-9]+(?:_part_[0-9]+)?)\b")


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_id(path: Path) -> dict[str, str]:
    return {row["identifier"]: row.get("llm_output", "") for row in load_rows(path)}


def atomic_write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=4), encoding="utf-8")
    os.replace(tmp, path)


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.match(r"^```(?:json|python)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text


def strip_hash_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
        out.append(line)
    return "\n".join(out)


def balanced_slice(text: str, start_char: str, end_char: str) -> str | None:
    start = text.find(start_char)
    if start < 0:
        return None
    depth = 0
    in_string = False
    quote = ""
    escape = False
    for index, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch in {'"', "'"}:
            if not in_string:
                in_string = True
                quote = ch
            elif quote == ch:
                in_string = False
            continue
        if in_string:
            continue
        if ch == start_char:
            depth += 1
        elif ch == end_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_any_json(text: str | None) -> Any:
    text = clean_text(text)
    if not text:
        return None
    candidates = [text, strip_hash_comments(text)]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        sliced = balanced_slice(text, start_char, end_char)
        if sliced:
            candidates.extend([sliced, strip_hash_comments(sliced)])
        first = text.find(start_char)
        last = text.rfind(end_char)
        if first >= 0 and last > first:
            raw = text[first : last + 1]
            candidates.extend([raw, strip_hash_comments(raw)])

    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass
    return None


def between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def split_compact(prompt: str, expected_parts: int) -> list[str]:
    parts = [part.strip() for part in prompt.split("--")]
    if len(parts) < expected_parts:
        raise ValueError("compact prompt has too few sections")
    return parts[:expected_parts]


def parse_gi_meta(prompt: str) -> dict:
    if "Relevant objects in the scene are:\n" in prompt:
        block = between(prompt, "Relevant objects in the scene are:\n", "\n\nAll initial states in the scene are:")
    else:
        block = split_compact(prompt, 3)[0]
    objects = set()
    for line in block.splitlines():
        if ":" in line:
            objects.add(line.split(":", 1)[0].strip())
    return {"objects": objects}


def parse_sd_meta(prompt: str) -> dict:
    if "Now, it is time for you to generate the subgoal plan for the following task." in prompt:
        marker = "Now, it is time for you to generate the subgoal plan for the following task."
        target = prompt.split(marker, 1)[1]
        relevant = between(target, "## Relevant objects in this scene\n", "\n\n## Initial States")
    else:
        target = prompt
        _, relevant, _, _ = split_compact(prompt, 4)
    object_names = set()
    for line in relevant.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = ast.literal_eval(line)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            object_names.add(str(obj["name"]))
    for ref in OBJECT_REF_RE.findall(target):
        if ref.startswith("agent"):
            object_names.add(ref)
    return {"objects": object_names}


def parse_as_meta(prompt: str) -> dict:
    if "Your task:" in prompt:
        target = prompt.split("Your task:", 1)[1]
        block = between(target, "interactable objects:\n", "\n\n\nPlease output")
    else:
        _, _, block = split_compact(prompt, 3)
    object_names = set()
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = ast.literal_eval(line)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            object_names.add(str(obj["name"]))
    return {"objects": object_names}


def normalize_pred(value: Any) -> str:
    return str(value).strip().replace("-", "_").replace(" ", "_").lower()


def gi_goal_from_dict(goal: dict, kind: str) -> list | None:
    if kind == "node":
        name = goal.get("name") or goal.get("object") or goal.get("object_name")
        state = goal.get("state") or goal.get("predicate")
        negated = bool(goal.get("not") or goal.get("negated"))
        if name and state:
            inner = [state, name]
            return ["not", inner] if negated else inner
    if kind == "edge":
        relation = goal.get("relation") or goal.get("predicate")
        source = goal.get("from_name") or goal.get("from") or goal.get("source") or goal.get("object1")
        target = goal.get("to_name") or goal.get("to") or goal.get("target") or goal.get("object2")
        negated = bool(goal.get("not") or goal.get("negated"))
        if relation and source and target:
            inner = [relation, source, target]
            return ["not", inner] if negated else inner
    return None


def normalize_gi_goal(goal: Any, kind: str, object_names: set[str]) -> list | None:
    if isinstance(goal, dict):
        goal = gi_goal_from_dict(goal, kind)
    if not isinstance(goal, list) or not goal:
        return None

    negated = False
    inner = goal
    if len(goal) == 2 and normalize_pred(goal[0]) == "not" and isinstance(goal[1], list):
        negated = True
        inner = goal[1]
    if not isinstance(inner, list) or not inner:
        return None

    pred_key = normalize_pred(inner[0])
    if kind == "node" and len(inner) == 2 and pred_key in NODE_CANON:
        obj = str(inner[1]).strip()
        if obj not in object_names:
            return None
        clean = [NODE_CANON[pred_key], obj]
        return ["not", clean] if negated else clean
    if kind == "edge" and len(inner) == 3 and pred_key in EDGE_CANON:
        obj1 = str(inner[1]).strip()
        obj2 = str(inner[2]).strip()
        if obj1 not in object_names or obj2 not in object_names:
            return None
        clean = [EDGE_CANON[pred_key], obj1, obj2]
        return ["not", clean] if negated else clean
    return None


def process_gi(raw: str, meta: dict, stats: dict) -> str:
    parsed = parse_any_json(raw)
    if not isinstance(parsed, dict):
        stats["parse_failed"] += 1
        stats["empty_schema_rows"] += 1
        return json.dumps({"node goals": [], "edge goals": []}, ensure_ascii=False, indent=2)

    object_names = meta["objects"]
    node_values = parsed.get("node goals") or parsed.get("node_goals") or parsed.get("nodes") or []
    edge_values = parsed.get("edge goals") or parsed.get("edge_goals") or parsed.get("edges") or []
    clean_nodes = []
    clean_edges = []
    dropped = 0
    for goal in node_values if isinstance(node_values, list) else []:
        normalized = normalize_gi_goal(goal, "node", object_names)
        if normalized is None:
            dropped += 1
        else:
            clean_nodes.append(normalized)
    for goal in edge_values if isinstance(edge_values, list) else []:
        normalized = normalize_gi_goal(goal, "edge", object_names)
        if normalized is None:
            dropped += 1
        else:
            clean_edges.append(normalized)
    stats["parsed_rows"] += 1
    stats["dropped_items"] += dropped
    return json.dumps({"node goals": clean_nodes, "edge goals": clean_edges}, ensure_ascii=False, indent=2)


def split_subgoal(text: str) -> list[str]:
    text = text.strip().strip(",;")
    if not text:
        return []
    parts = re.split(r"\s+\b(?:and|or)\b\s+", text)
    return [part.strip().strip(",;") for part in parts if part.strip().strip(",;")]


def normalize_sd_item(item: Any, object_names: set[str]) -> list[str]:
    if not isinstance(item, str):
        return []
    out = []
    for part in split_subgoal(item):
        lowered = part.lower()
        if any(token in lowered for token in ("forall", "exists", "forpairs", "forn(")):
            continue
        match = re.match(r"^(not\s+)?([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)$", part)
        if not match:
            continue
        pred = normalize_pred(match.group(2))
        if pred not in SD_PREDICATES:
            continue
        refs = OBJECT_REF_RE.findall(part)
        if not refs or any(ref not in object_names for ref in refs):
            continue
        out.append(part)
    return out


def process_sd(raw: str, meta: dict, stats: dict) -> str:
    parsed = parse_any_json(raw)
    if isinstance(parsed, dict):
        values = parsed.get("output", [])
    elif isinstance(parsed, list):
        values = parsed
    else:
        stats["parse_failed"] += 1
        stats["empty_schema_rows"] += 1
        return json.dumps({"output": []}, ensure_ascii=False)

    object_names = meta["objects"]
    clean = []
    seen = set()
    source_count = 0
    for item in values if isinstance(values, list) else []:
        source_count += 1
        for normalized in normalize_sd_item(item, object_names):
            if normalized not in seen:
                clean.append(normalized)
                seen.add(normalized)

    stats["parsed_rows"] += 1
    stats["dropped_items"] += max(source_count - len(clean), 0)
    if not clean:
        stats["empty_schema_rows"] += 1
    return json.dumps({"output": clean}, ensure_ascii=False)


def normalize_as_actions(raw: str, object_names: set[str]) -> tuple[list[dict], int, bool]:
    parsed = parse_any_json(raw)
    if isinstance(parsed, dict) and "output" in parsed:
        parsed = parse_any_json(parsed.get("output")) if isinstance(parsed.get("output"), str) else parsed.get("output")
    if not isinstance(parsed, list):
        return [], 1, False

    clean = []
    invalid = 0
    leaked = False
    for item in parsed:
        if not isinstance(item, dict):
            invalid += 1
            continue
        action = str(item.get("action", "")).strip().upper()
        obj = item.get("object", "")
        if isinstance(obj, list):
            obj = ", ".join(str(part).strip() for part in obj)
        obj = str(obj).strip()
        parts = [part.strip() for part in obj.split(",") if part.strip()]
        if action not in VALID_ACTIONS or not parts:
            invalid += 1
            continue
        if any(part in AS_EXAMPLE_OBJECTS and part not in object_names for part in parts):
            leaked = True
        if any(part not in object_names for part in parts):
            invalid += 1
            continue
        clean.append({"action": action, "object": ", ".join(parts)})
    return clean, invalid, leaked


def process_as(raw: str, meta: dict, stats: dict) -> str:
    actions, invalid, leaked = normalize_as_actions(raw, meta["objects"])
    if leaked:
        stats["example_leak_rows"] += 1
    if invalid:
        stats["invalid_action_rows"] += 1
    if not actions or invalid or leaked:
        stats["parse_failed"] += int(not actions)
        stats["empty_schema_rows"] += int(not actions)
    else:
        stats["parsed_rows"] += 1
    return json.dumps(actions, ensure_ascii=False, indent=4)


def process_task(task: str, args: argparse.Namespace) -> dict:
    filename = TASK_FILES[task]
    prompt_filename = PROMPT_FILES[task]
    prompt_rows = load_rows(Path(args.official_prompt_dir) / prompt_filename)
    if args.limit is not None:
        prompt_rows = prompt_rows[: args.limit]
    raw_by_id = rows_by_id(Path(args.raw_output_dir) / filename)

    meta_parser = {"gi": parse_gi_meta, "sd": parse_sd_meta, "as": parse_as_meta}[task]
    processor = {"gi": process_gi, "sd": process_sd, "as": process_as}[task]
    stats = {
        "task": task,
        "total": len(prompt_rows),
        "raw_missing": 0,
        "parsed_rows": 0,
        "parse_failed": 0,
        "empty_schema_rows": 0,
        "dropped_items": 0,
        "invalid_action_rows": 0,
        "example_leak_rows": 0,
    }

    out_rows = []
    for prompt in prompt_rows:
        identifier = prompt["identifier"]
        raw = raw_by_id.get(identifier, "")
        if not raw:
            stats["raw_missing"] += 1
        meta = meta_parser(prompt["llm_prompt"])
        output = processor(raw, meta, stats)
        out_rows.append({"identifier": identifier, "llm_output": output})

    out_path = Path(args.output_dir) / filename
    atomic_write_json(out_path, out_rows)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output-dir", required=True)
    parser.add_argument("--official-prompt-dir", default="llm_prompts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="*", choices=sorted(TASK_FILES), default=sorted(TASK_FILES))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    report = [process_task(task, args) for task in args.tasks]
    report_path = Path(args.output_dir) / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in report:
        print(json.dumps(item, ensure_ascii=False))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
