#!/usr/bin/env python3
"""Build AxisTilted2 compact BEHAVIOR prompts for Qwen3-0.6B task models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_FILES = {
    "gi": "behavior_goal_interpretation_prompts.json",
    "sd": "behavior_subgoal_decomposition_prompts.json",
    "as": "behavior_action_sequencing_prompts.json",
}


def between(text: str, start: str, end: str) -> str:
    if start not in text:
        raise ValueError(f"missing start marker: {start!r}")
    rest = text.split(start, 1)[1]
    if end not in rest:
        raise ValueError(f"missing end marker: {end!r}")
    return rest.split(end, 1)[0].strip()


def extract_json_object_after(text: str, marker: str) -> dict:
    if marker not in text:
        raise ValueError(f"missing marker: {marker!r}")
    tail = text.split(marker, 1)[1]
    start = tail.find("{")
    if start < 0:
        raise ValueError("missing JSON object")

    depth = 0
    in_string = False
    escape = False
    for offset, ch in enumerate(tail[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(tail[start : offset + 1])
    raise ValueError("unterminated JSON object")


def compact_gi(prompt: str) -> str:
    relevant_objects = between(
        prompt,
        "Relevant objects in the scene are:\n",
        "\n\nAll initial states in the scene are:",
    )
    initial_states = between(
        prompt,
        "All initial states in the scene are:\n",
        "\n\nSymbolic goals format:",
    )
    task_info = extract_json_object_after(prompt, "Task Name and Goal Instructions:")
    task_name = str(task_info.get("Task Name", "")).strip()
    goal_instructions = str(task_info.get("Goal Instructions", "")).strip()
    return f"{relevant_objects}\n--\n{initial_states}\n--\n{task_name}\n{goal_instructions}\n"


def compact_sd(prompt: str) -> str:
    marker = "Now, it is time for you to generate the subgoal plan for the following task."
    if marker not in prompt:
        raise ValueError("missing SD target task marker")
    target = prompt.split(marker, 1)[1]
    task_name = between(target, "# Target Task:", "\n## Relevant objects in this scene").strip()
    relevant_objects = between(
        target,
        "## Relevant objects in this scene\n",
        "\n\n## Initial States",
    )
    initial_states = between(target, "## Initial States\n", "\n\n## Goal States")
    goal_states = between(target, "## Goal States\n", "\n\n## Output:")
    return f"{task_name}\n--\n{relevant_objects}\n--\n{initial_states}\n--\n{goal_states}\n"


def compact_as(prompt: str) -> str:
    if "Your task:" not in prompt:
        raise ValueError("missing AS target task marker")
    target = prompt.split("Your task:", 1)[1]
    initial_states = between(
        target,
        "initial environment state:\n",
        "\n\n\ntarget environment state:",
    )
    target_states = between(
        target,
        "target environment state:\n",
        "\n\n\ninteractable objects:",
    )
    interactable_objects = between(
        target,
        "interactable objects:\n",
        "\n\n\nPlease output",
    )
    return f"{initial_states}\n--\n{target_states}\n--\n{interactable_objects}\n"


def rewrite_rows(rows: list[dict], task: str) -> list[dict]:
    builders = {"gi": compact_gi, "sd": compact_sd, "as": compact_as}
    builder = builders[task]
    out = []
    for row in rows:
        out.append(
            {
                "identifier": row["identifier"],
                "llm_prompt": builder(row["llm_prompt"]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="llm_prompts")
    parser.add_argument("--output-dir", default="llm_prompts_axis06b_behavior_compact")
    parser.add_argument("--tasks", nargs="*", choices=sorted(TASK_FILES), default=sorted(TASK_FILES))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        filename = TASK_FILES[task]
        rows = json.loads((input_dir / filename).read_text(encoding="utf-8"))
        rewritten = rewrite_rows(rows, task)
        out_path = output_dir / filename
        out_path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        lengths = [len(row["llm_prompt"]) for row in rewritten]
        print(f"{filename}: rows={len(rewritten)} min/avg/max={min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}")


if __name__ == "__main__":
    main()
