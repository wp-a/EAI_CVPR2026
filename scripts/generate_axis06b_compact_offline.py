#!/usr/bin/env python3
"""Generate compact-prompt BEHAVIOR outputs with a full AxisTilted2 Qwen3-0.6B model."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams


TASK_MAX_TOKENS = {
    "behavior_goal_interpretation": 1024,
    "behavior_subgoal_decomposition": 2048,
    "behavior_action_sequencing": 4096,
}


def clean_qwen_output(text: str | None) -> str:
    if text is None:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    fence = re.match(r"^```(?:json|python)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text


def load_existing(path: Path) -> dict[str, str]:
    candidates = [path, path.with_suffix(path.suffix + ".tmp")]
    best_rows = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            rows = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if len(rows) > len(best_rows):
            best_rows = rows
    return {row["identifier"]: row.get("llm_output", "") for row in best_rows}


def atomic_write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=4), encoding="utf-8")
    os.replace(tmp, path)


def output_name_for_prompt_file(prompt_file: Path) -> str:
    return prompt_file.name.replace("_prompts.json", "_outputs.json")


def sharded_output_name_for_prompt_file(prompt_file: Path, args: argparse.Namespace) -> str:
    name = output_name_for_prompt_file(prompt_file)
    if args.num_shards <= 1 and not args.shard_output:
        return name
    stem = name[:-5] if name.endswith(".json") else name
    return f"{stem}.shard{args.shard_index}-of-{args.num_shards}.json"


def iter_prompt_files(prompt_dir: Path, selected: list[str] | None) -> list[Path]:
    files = sorted(prompt_dir.glob("*_prompts.json"))
    if selected:
        selected_set = set(selected)
        files = [
            file
            for file in files
            if file.name in selected_set
            or file.stem in selected_set
            or file.name.replace("_prompts.json", "") in selected_set
        ]
    return files


def task_key(prompt_file: Path) -> str | None:
    name = prompt_file.name.replace("_prompts.json", "")
    for key in TASK_MAX_TOKENS:
        if key in name:
            return key
    return None


def max_tokens_for_file(args: argparse.Namespace, prompt_file: Path) -> int:
    if args.max_tokens is not None:
        return args.max_tokens
    return TASK_MAX_TOKENS.get(task_key(prompt_file) or "", args.default_max_tokens)


def build_chat_prompt(tokenizer, raw_prompt: str, no_think: bool) -> str:
    content = raw_prompt
    if no_think:
        content = "/no_think\n" + content
    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not no_think,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


def make_sampling_params(args: argparse.Namespace, max_tokens: int) -> SamplingParams:
    kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens,
    }
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("top_k", None)
        try:
            return SamplingParams(**kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            return SamplingParams(**kwargs)


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-dir", default="llm_prompts_axis06b_behavior_compact")
    parser.add_argument("--output-dir", default="outputs/axis06b_behavior_compact_raw")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--default-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-output", action="store_true")
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--think", dest="no_think", action="store_false", help="Do not prepend /no_think.")
    parser.set_defaults(no_think=True)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < num_shards")

    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    prompt_files = iter_prompt_files(prompt_dir, args.files)
    if not prompt_files:
        raise SystemExit(f"No prompt files found in {prompt_dir}")

    print(f"model={args.model}")
    print(f"prompt_dir={prompt_dir}")
    print(f"prompt_files={len(prompt_files)}")
    print(f"output_dir={output_dir}")
    print(f"temperature={args.temperature} top_p={args.top_p} top_k={args.top_k} seed={args.seed}")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        trust_remote_code=False,
    )
    tokenizer = llm.get_tokenizer()

    started_at = time.time()
    for prompt_file in prompt_files:
        prompts = json.loads(prompt_file.read_text(encoding="utf-8"))
        if args.limit is not None:
            prompts = prompts[: args.limit]
        if args.num_shards > 1:
            prompts = [prompt for index, prompt in enumerate(prompts) if index % args.num_shards == args.shard_index]

        output_path = output_dir / sharded_output_name_for_prompt_file(prompt_file, args)
        existing = {} if args.no_resume else load_existing(output_path)
        rows = [
            {"identifier": prompt["identifier"], "llm_output": existing[prompt["identifier"]]}
            for prompt in prompts
            if prompt["identifier"] in existing and existing[prompt["identifier"]]
        ]
        pending = [prompt for prompt in prompts if prompt["identifier"] not in existing or not existing[prompt["identifier"]]]
        max_tokens = max_tokens_for_file(args, prompt_file)
        sampling_params = make_sampling_params(args, max_tokens)

        print(f"\n== {prompt_file.name}: total={len(prompts)} resume={len(rows)} pending={len(pending)}")
        print(f"output={output_path} max_tokens={max_tokens}")
        atomic_write_json(output_path, rows)

        for start, batch in batched(pending, args.batch_size):
            batch_prompts = [build_chat_prompt(tokenizer, prompt["llm_prompt"], args.no_think) for prompt in batch]
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            for prompt, output in zip(batch, outputs):
                text = output.outputs[0].text if output.outputs else ""
                existing[prompt["identifier"]] = clean_qwen_output(text)

            rows = [
                {"identifier": prompt["identifier"], "llm_output": existing.get(prompt["identifier"], "")}
                for prompt in prompts
                if prompt["identifier"] in existing
            ]
            atomic_write_json(output_path, rows)
            elapsed = time.time() - started_at
            print(f"[{len(rows)}/{len(prompts)}] batch_start={start + 1} batch={len(batch)} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
