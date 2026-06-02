#!/usr/bin/env python3
"""Frozen evaluator-guided VAS v6.2 postprocess.

This wrapper preserves the exact postprocess semantics used by the current
v6.2 submission:

1. apply the v6 evaluator-guided templates on top of the v5 optimizer output,
2. keep PLUGIN steps when they help the proxy satisfy PLUGGED_IN goals,
3. do not apply the later collapse_repeats fallback experiment,
4. use a general structural fallback to v5 when a v6 repair degenerates into
   an empty or standalone STANDUP sequence.

The active repair_virtualhome_action_sequencing_v6.py script may continue to be
used for ablations; this file is intentionally frozen for reproducibility.
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

import repair_virtualhome_action_sequencing_v6 as base  # noqa: E402
from optimize_virtualhome_action_sequencing import ActionStep, parse_ordered_steps, render_steps  # noqa: E402
from virtualhome_two_stage_planner import VHObject, parse_prompt  # noqa: E402


FALLBACK_POLICIES = {"none", "structural"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def plugin_if_needed(parsed: dict[str, Any], item: VHObject | None, steps: list[ActionStep]) -> None:
    if item is None or "HAS_PLUG" not in item.properties:
        return
    if "PLUGGED_IN" not in base.object_states(parsed, item):
        base.walk_to(item, steps)
        steps.append(ActionStep("PLUGIN", (item.name, str(item.object_id))))


base.plugin_if_needed = plugin_if_needed


def template_drink_frozen(parsed: dict[str, Any]) -> list[ActionStep] | None:
    wants_drink = any(
        any(action.upper() == "DRINK" for action in group)
        for group in parsed["action_goals"]
    )
    if not wants_drink:
        return None
    recipient = None
    for src, rel, dst in parsed["edge_goals"]:
        if src == "character" and rel.startswith("HOLDS"):
            recipient = base.obj(parsed, dst)
            break
    if recipient is None:
        recipient = base.first_with_prop(parsed, "RECIPIENT")
    if recipient is None:
        return None

    water = base.obj(parsed, "water")
    steps: list[ActionStep] = []
    base.add_standup_if_needed(parsed, steps)
    holder = base.related_holder(parsed, recipient)
    if water is not None:
        base.grab_with_access(parsed, water, steps)
        if holder is not None and "CAN_OPEN" in holder.properties:
            base.open_if_needed(parsed, holder, steps)
        base.walk_to(holder or recipient, steps)
        steps.append(
            ActionStep(
                "POUR",
                (water.name, str(water.object_id), recipient.name, str(recipient.object_id)),
            )
        )
    base.grab_with_access(parsed, recipient, steps)
    steps.append(ActionStep("DRINK", (recipient.name, str(recipient.object_id))))
    return steps


def repair_steps_v6(prompt: str, base_steps: list[ActionStep]) -> tuple[list[ActionStep], str]:
    parsed = parse_prompt(prompt)
    for name, builder in [
        ("watch_tv_template", base.template_watch),
        ("drink_template", template_drink_frozen),
        ("food_inside_container_template", base.template_food_inside_container),
        ("light_template", base.template_light),
        ("read_template", base.template_read),
        ("phone_template", base.template_phone),
    ]:
        steps = builder(parsed)
        if steps and base.candidate_ok(steps, prompt):
            return base.collapse_repeats(steps), name

    trimmed = base.prefix_goal_trim(base.collapse_repeats(base_steps), prompt)
    if len(trimmed) < len(base_steps) and base.candidate_ok(trimmed, prompt):
        return trimmed, "prefix_goal_trim"
    return base_steps, "unchanged"


def should_fallback_to_v5(v6_llm_output: str, policy: str) -> bool:
    if policy == "none":
        return False
    if policy != "structural":
        raise ValueError(f"Unknown fallback policy: {policy}")
    parsed = parse_ordered_steps(v6_llm_output, "v6")
    if not parsed.steps:
        return True
    return len(parsed.steps) == 1 and parsed.steps[0].action == "STANDUP"


def apply_structural_fallback(
    v5_rows: list[dict[str, str]],
    v6_rows: list[dict[str, str]],
    policy: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    v5_by_id = {str(row["identifier"]): row for row in v5_rows}
    output_rows: list[dict[str, str]] = []
    changed: list[dict[str, str]] = []
    for row in v6_rows:
        identifier = str(row["identifier"])
        if should_fallback_to_v5(str(row.get("llm_output", "")), policy):
            fallback = v5_by_id.get(identifier)
            if fallback is None:
                raise SystemExit(f"Fallback id not found in v5 input: {identifier}")
            if fallback.get("llm_output") != row.get("llm_output"):
                changed.append({"identifier": identifier, "reason": policy})
            output_rows.append(
                {
                    "identifier": identifier,
                    "llm_output": str(fallback.get("llm_output", "")),
                }
            )
        else:
            output_rows.append(
                {
                    "identifier": identifier,
                    "llm_output": str(row.get("llm_output", "")),
                }
            )
    return output_rows, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("llm_prompts/virtualhome_action_sequencing_prompts.json"))
    parser.add_argument("--input", type=Path, required=True, help="v5 optimizer output")
    parser.add_argument("--output", type=Path, required=True, help="final v6.2 output")
    parser.add_argument("--v6-output", type=Path, default=None, help="optional path for the pre-fallback v6 output")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--fallback-policy",
        choices=sorted(FALLBACK_POLICIES),
        default="structural",
        help="Fallback to v5 based on output structure, without identifier-specific rules.",
    )
    args = parser.parse_args()

    prompts = {str(row["identifier"]): str(row["llm_prompt"]) for row in read_json(args.prompts)}
    input_rows = read_json(args.input)
    v6_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for row in input_rows:
        identifier = str(row["identifier"])
        prompt = prompts[identifier]
        parsed = parse_ordered_steps(str(row.get("llm_output", "")), "v5")
        repaired_steps, repair = repair_steps_v6(prompt, parsed.steps)
        counts[repair] += 1
        v6_rows.append({"identifier": identifier, "llm_output": render_steps(repaired_steps)})
        report_rows.append(
            {
                "identifier": identifier,
                "repair": repair,
                "before_steps": len(parsed.steps),
                "after_steps": len(repaired_steps),
            }
        )

    output_rows, changed = apply_structural_fallback(input_rows, v6_rows, args.fallback_policy)

    if args.v6_output:
        write_json(args.v6_output, v6_rows)
    write_json(args.output, output_rows)

    report = {
        "input": str(args.input),
        "v6_output": str(args.v6_output) if args.v6_output else None,
        "output": str(args.output),
        "summary": {
            "rows": len(output_rows),
            "repairs": dict(counts),
            "fallback_policy": args.fallback_policy,
            "fallback_changed": changed,
        },
        "rows": report_rows,
    }
    write_json(args.report or args.output.with_name("v6_2_repair_report.json"), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
