#!/usr/bin/env python3
"""Generate VirtualHome GI outputs with a lightweight Qwen LoRA adapter."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_runtime import LoRALinear, get_parent_module


SYSTEM = "You are a VirtualHome Goal Interpretation model. Output exactly one JSON object with keys: node goals, edge goals, action goals. Do not output markdown, explanations, or <think>."


def inject_empty_lora(model: nn.Module, adapter_state: dict[str, torch.Tensor]) -> None:
    suffix = ".lora_a"
    modules = [name[: -len(suffix)] for name in adapter_state if name.endswith(suffix)]
    for module_name in modules:
        parent, child_name = get_parent_module(model, module_name)
        base = getattr(parent, child_name)
        if not isinstance(base, nn.Linear):
            continue
        rank = adapter_state[f"{module_name}.lora_a"].shape[0]
        wrapper = LoRALinear(base, rank=rank, alpha=rank * 2, dropout=0.0)
        setattr(parent, child_name, wrapper)


def load_lora(model: nn.Module, adapter_path: Path) -> None:
    state = torch.load(adapter_path, map_location="cpu")
    inject_empty_lora(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [item for item in unexpected if "lora_" in item]
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {unexpected[:5]}")


def load_rows(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "").strip()
    start = text.find("{")
    if start >= 0:
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
            elif char in {"'", '"'}:
                in_string = True
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--max-input-tokens", type=int, default=12000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=False,
    )
    load_lora(model, Path(args.adapter))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rows = load_rows(Path(args.prompt_file))
    if args.limit:
        rows = rows[: args.limit]
    output_rows = []
    for index, row in enumerate(rows):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["llm_prompt"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        if input_ids.shape[-1] > args.max_input_tokens:
            input_ids = input_ids[:, -args.max_input_tokens :]
        input_ids = input_ids.to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p if args.temperature > 0 else None,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0, input_ids.shape[-1] :], skip_special_tokens=True)
        output_rows.append({"identifier": row["identifier"], "llm_output": strip_output(text)})
        if (index + 1) % 50 == 0:
            print(json.dumps({"generated": index + 1}, ensure_ascii=False), flush=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
