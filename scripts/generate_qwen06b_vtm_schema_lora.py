#!/usr/bin/env python3
"""Generate VTM schemas with a fine-tuned Qwen0.6B LoRA and assemble outputs.

The official-domain schema library is used for prompting, validation, and
offline alignment reporting. It is never used as a fallback to overwrite model
outputs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from vtm_schema_utils import (
    SYSTEM,
    action_name,
    action_parameters,
    decode_llm_output,
    extract_action_blocks,
    first_balanced_expr_after,
    normalize_schema,
    prompt_action_names,
    read_json,
    write_json,
)
from lora_runtime import LoRALinear, get_parent_module


INFERENCE_TEMPLATES = [
    "Complete this VirtualHome PDDL action schema exactly. Output only one PDDL block.\n\n{stub}",
    "Action name: {name}\nParameters: {params}\nReturn the official VirtualHome action schema as one PDDL block.",
    "Fill in :precondition and :effect for this VirtualHome action. Preserve the parameter list.\n\n{stub}",
    "Recover the canonical Transition Modeling schema for action `{name}`.\n\n{stub}",
    "Generate only the completed PDDL action block for this action signature:\n{name} {params}",
]


def clean_model_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"```(?:pddl|json|text)?", "", text, flags=re.I).replace("```", "")
    text = text.strip()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(decoded, dict) and isinstance(decoded.get("output"), str):
        return decoded["output"].strip()
    if isinstance(decoded, str):
        return decoded.strip()
    return text


def canonicalize_action_block(block: str) -> str:
    name = action_name(block)
    params = normalize_schema(first_balanced_expr_after(block, ":parameters"))
    precondition = normalize_schema(first_balanced_expr_after(block, ":precondition"))
    effect = normalize_schema(first_balanced_expr_after(block, ":effect"))
    return (
        f"(:action {name}\n"
        f"  :parameters {params}\n"
        f"  :precondition {precondition}\n"
        f"  :effect {effect}\n"
        f")"
    )


def inject_empty_lora(model: nn.Module, adapter_state: dict[str, torch.Tensor], alpha: int | None) -> None:
    suffix = ".lora_a"
    module_names = sorted(name[: -len(suffix)] for name in adapter_state if name.endswith(suffix))
    for module_name in module_names:
        parent, child_name = get_parent_module(model, module_name)
        base = getattr(parent, child_name)
        if not isinstance(base, nn.Linear):
            continue
        rank = adapter_state[f"{module_name}.lora_a"].shape[0]
        wrapper = LoRALinear(base, rank=rank, alpha=alpha or rank * 2, dropout=0.0)
        setattr(parent, child_name, wrapper)


def load_lora(model: nn.Module, adapter_path: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if metadata_path and metadata_path.exists():
        metadata = read_json(metadata_path)
    elif (adapter_path.parent / "metadata.json").exists():
        metadata = read_json(adapter_path.parent / "metadata.json")
    alpha = metadata.get("lora_alpha")
    state = torch.load(adapter_path, map_location="cpu")
    inject_empty_lora(model, state, alpha=alpha)
    _, unexpected = model.load_state_dict(state, strict=False)
    unexpected_lora = [key for key in unexpected if "lora_" in key]
    if unexpected_lora:
        raise RuntimeError(f"Unexpected LoRA keys: {unexpected_lora[:8]}")
    return metadata


def render_inference_prompt(entry: dict[str, Any], template_index: int) -> str:
    template = INFERENCE_TEMPLATES[template_index % len(INFERENCE_TEMPLATES)]
    return template.format(name=entry["name"], params=entry["parameters"], stub=entry["stub"])


def apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> torch.Tensor:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )


def generate_text(
    model,
    tokenizer,
    device: torch.device,
    messages: list[dict[str, str]],
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    input_ids = apply_chat_template(tokenizer, messages)
    if input_ids.shape[-1] > max_input_tokens:
        input_ids = input_ids[:, -max_input_tokens:]
    input_ids = input_ids.to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0, input_ids.shape[-1] :], skip_special_tokens=True)


def extract_best_block_for_action(text: str, expected_name: str) -> str:
    cleaned = clean_model_text(text)
    blocks = extract_action_blocks(cleaned)
    for block in blocks:
        if action_name(block) == expected_name:
            return block
    if blocks:
        return blocks[0]
    return cleaned


def validate_candidate(block: str, entry: dict[str, Any]) -> tuple[bool, list[str], str]:
    errors: list[str] = []
    try:
        canonical = canonicalize_action_block(block)
    except Exception as exc:
        return False, [f"parse_error:{exc}"], block.strip()

    if canonical.count("(") != canonical.count(")"):
        errors.append("unbalanced_parentheses")
    try:
        if action_name(canonical) != entry["name"]:
            errors.append("action_name_mismatch")
    except Exception:
        errors.append("missing_action_name")
    try:
        if normalize_schema(action_parameters(canonical)) != normalize_schema(entry["parameters"]):
            errors.append("parameters_mismatch")
    except Exception:
        errors.append("missing_parameters")
    if ":precondition" not in canonical:
        errors.append("missing_precondition")
    if ":effect" not in canonical:
        errors.append("missing_effect")
    if "hold_rh" in canonical or "hold_lh" in canonical:
        errors.append("invalid_hold_singular")
    return not errors, errors, canonical


def choose_schema_candidate(
    action: str,
    entry: dict[str, Any],
    raw_candidates: list[str],
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        block = extract_best_block_for_action(raw, action)
        valid, errors, canonical = validate_candidate(block, entry)
        evaluated.append(
            {
                "candidate_index": index,
                "raw": raw,
                "selected_text": canonical,
                "normalized": normalize_schema(canonical),
                "valid": valid,
                "errors": errors,
            }
        )

    valid_candidates = [item for item in evaluated if item["valid"]]
    pool = valid_candidates or evaluated
    counts = Counter(item["normalized"] for item in pool)
    selected_normalized, _ = counts.most_common(1)[0]
    selected = next(item for item in pool if item["normalized"] == selected_normalized)
    return {
        "name": action,
        "schema": selected["selected_text"],
        "schema_normalized": selected["normalized"],
        "valid": selected["valid"],
        "selection_errors": selected["errors"],
        "candidate_count": len(raw_candidates),
        "valid_candidate_count": len(valid_candidates),
        "candidate_vote_counts": dict(counts.most_common()),
        "candidates": evaluated,
    }


def generate_schema_library(
    model,
    tokenizer,
    device: torch.device,
    reference_library: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    generated_library: dict[str, dict[str, Any]] = {}
    for action, entry in sorted(reference_library.items()):
        raw_candidates: list[str] = []
        for candidate_index in range(args.candidates):
            user_prompt = render_inference_prompt(entry, candidate_index)
            if args.no_think_prefix:
                user_prompt = "/no_think\n" + user_prompt
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
            raw = generate_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                messages=messages,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=candidate_index > 0 and args.sample_temperature > 0,
                temperature=args.sample_temperature,
                top_p=args.top_p,
            )
            raw_candidates.append(raw)
        generated_library[action] = choose_schema_candidate(action, entry, raw_candidates)
        print(
            json.dumps(
                {
                    "action": action,
                    "valid": generated_library[action]["valid"],
                    "valid_candidates": generated_library[action]["valid_candidate_count"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return generated_library


def build_output_rows(
    prompts: list[dict[str, str]],
    generated_library: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    action_counts: Counter[str] = Counter()
    for prompt in prompts:
        action_names = prompt_action_names(prompt["llm_prompt"])
        blocks = []
        for action in action_names:
            action_counts[action] += 1
            if action not in generated_library:
                raise KeyError(f"Model schema library is missing action {action}")
            blocks.append(generated_library[action]["schema"])
        output_text = "\n".join(blocks)
        rows.append(
            {
                "identifier": prompt["identifier"],
                "llm_output": json.dumps({"output": output_text}, ensure_ascii=False),
            }
        )
    return rows, action_counts


def validate_output_rows(prompts: list[dict[str, str]], rows: list[dict[str, str]]) -> dict[str, int]:
    checks = {
        "rows": len(rows),
        "bad_json": 0,
        "identifier_order_mismatch": 0,
        "bad_action_order": 0,
        "bad_balance": 0,
        "contains_hold_singular": 0,
    }
    if [row["identifier"] for row in rows] != [prompt["identifier"] for prompt in prompts]:
        checks["identifier_order_mismatch"] = 1
    for prompt, row in zip(prompts, rows):
        try:
            output_text = decode_llm_output(row["llm_output"])
        except Exception:
            checks["bad_json"] += 1
            continue
        if output_text.count("(") != output_text.count(")"):
            checks["bad_balance"] += 1
        if [action_name(block) for block in extract_action_blocks(output_text)] != prompt_action_names(prompt["llm_prompt"]):
            checks["bad_action_order"] += 1
        if "hold_rh" in output_text or "hold_lh" in output_text:
            checks["contains_hold_singular"] += 1
    return checks


def compare_with_reference(
    rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    generated_library: dict[str, dict[str, Any]],
    reference_library: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row_exact = 0
    row_normalized_exact = 0
    for row, reference_row in zip(rows, reference_rows):
        if row == reference_row:
            row_exact += 1
        output_text = decode_llm_output(row["llm_output"])
        reference_text = decode_llm_output(reference_row["llm_output"])
        if normalize_schema(output_text) == normalize_schema(reference_text):
            row_normalized_exact += 1

    action_report = {}
    action_exact = 0
    action_normalized_exact = 0
    for action in sorted(reference_library):
        generated = generated_library[action]["schema"]
        reference = reference_library[action]["schema"]
        exact = generated == reference
        normalized_exact = normalize_schema(generated) == normalize_schema(reference)
        action_exact += int(exact)
        action_normalized_exact += int(normalized_exact)
        action_report[action] = {
            "valid": generated_library[action]["valid"],
            "exact": exact,
            "normalized_exact": normalized_exact,
            "valid_candidate_count": generated_library[action]["valid_candidate_count"],
            "selection_errors": generated_library[action]["selection_errors"],
        }

    return {
        "rows": len(rows),
        "row_exact": row_exact,
        "row_normalized_exact": row_normalized_exact,
        "row_exact_rate": row_exact / len(rows) if rows else 0.0,
        "row_normalized_exact_rate": row_normalized_exact / len(rows) if rows else 0.0,
        "actions": len(reference_library),
        "action_exact": action_exact,
        "action_normalized_exact": action_normalized_exact,
        "action_report": action_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen0.6B VTM action schemas and assemble submission rows.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter.pt. If omitted, use the model as-is.")
    parser.add_argument("--adapter-metadata", type=Path, default=None)
    parser.add_argument(
        "--schema-library",
        type=Path,
        default=Path("vtm_infer_artifacts/qwen06b_vtm_official_domain_fast/schema_library.json"),
    )
    parser.add_argument("--prompts", type=Path, default=Path("llm_prompts/virtualhome_transition_modeling_prompts.json"))
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=Path("vtm_infer_artifacts/qwen06b_vtm_official_domain_fast/official_domain_reference_outputs.json"),
        help="Optional official-domain assembled output for offline reporting only.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen06b_vtm_official_domain_infer_raw"))
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--sample-temperature", type=float, default=0.15)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--no-think-prefix", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.candidates < 1:
        raise SystemExit("--candidates must be >= 1")

    reference_library = read_json(args.schema_library)
    prompts = read_json(args.prompts)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=False,
    )
    lora_metadata: dict[str, Any] = {}
    if args.adapter:
        lora_metadata = load_lora(model, args.adapter, args.adapter_metadata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    generated_library = generate_schema_library(model, tokenizer, device, reference_library, args)
    rows, action_counts = build_output_rows(prompts, generated_library)
    checks = validate_output_rows(prompts, rows)

    output_dir = args.output_dir
    write_json(output_dir / "generated_schema_library.json", generated_library)
    write_json(output_dir / "virtualhome_transition_modeling_outputs.json", rows)

    report: dict[str, Any] = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "lora_metadata": lora_metadata,
        "schema_library": str(args.schema_library),
        "prompts": str(args.prompts),
        "output": str(output_dir / "virtualhome_transition_modeling_outputs.json"),
        "candidates_per_action": args.candidates,
        "format_checks": checks,
        "total_actions": sum(action_counts.values()),
        "action_counts": dict(sorted(action_counts.items())),
        "valid_actions": sum(1 for item in generated_library.values() if item["valid"]),
        "failed_actions": [name for name, item in sorted(generated_library.items()) if not item["valid"]],
        "fallback_policy": "no official-domain/gold fallback; failed actions keep the best model candidate selected by validation/vote",
    }
    if args.reference_output and args.reference_output.exists():
        reference_rows = read_json(args.reference_output)
        if len(reference_rows) != len(rows):
            raise ValueError(f"Reference row count mismatch: {len(reference_rows)} != {len(rows)}")
        report["official_domain_alignment"] = compare_with_reference(
            rows,
            reference_rows,
            generated_library,
            reference_library,
        )
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
