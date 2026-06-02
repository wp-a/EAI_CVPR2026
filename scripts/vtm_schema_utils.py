#!/usr/bin/env python3
"""Runtime helpers for VirtualHome transition-modeling PDDL outputs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"\(:action\s+([A-Za-z_][A-Za-z0-9_]*)")

SYSTEM = (
    "You are a VirtualHome Transition Modeling action-schema model. "
    "Given action stubs, output only the completed PDDL action block(s). "
    "Do not output markdown, JSON, explanations, or thinking text."
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def decode_llm_output(value: str) -> str:
    """Decode an EvalAI-style stringified JSON output for local format checks."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(decoded, dict) and isinstance(decoded.get("output"), str):
        return decoded["output"]
    if isinstance(decoded, str):
        return decoded
    raise ValueError(f"Unsupported llm_output payload: {type(decoded).__name__}")


def balanced_block_at(text: str, start: int) -> str:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()
    raise ValueError(f"Unbalanced PDDL block starting at {start}")


def extract_action_blocks(text: str) -> list[str]:
    return [balanced_block_at(text, match.start()) for match in ACTION_RE.finditer(text)]


def action_name(block: str) -> str:
    match = ACTION_RE.search(block)
    if not match:
        raise ValueError(f"No action name found in block: {block[:80]}")
    return match.group(1)


def first_balanced_expr_after(text: str, marker: str) -> str:
    marker_index = text.find(marker)
    if marker_index < 0:
        raise ValueError(f"Missing section {marker!r}")
    start = text.find("(", marker_index)
    if start < 0:
        raise ValueError(f"Missing expression after {marker!r}")
    return balanced_block_at(text, start)


def action_parameters(block: str) -> str:
    return first_balanced_expr_after(block, ":parameters")


def normalize_schema(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def final_input_segment(prompt: str) -> str:
    segment = prompt.rsplit("Input:", 1)[-1]
    return segment.split("Output:", 1)[0]


def prompt_action_stubs(prompt: str) -> list[str]:
    return extract_action_blocks(final_input_segment(prompt))


def prompt_action_names(prompt: str) -> list[str]:
    return [action_name(block) for block in prompt_action_stubs(prompt)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
