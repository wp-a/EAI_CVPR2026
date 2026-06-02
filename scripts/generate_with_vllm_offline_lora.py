import argparse
import json
import os
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


TASK_MAX_TOKENS = {
    "goal_interpretation": 2048,
    "transition_modeling": 4096,
    "action_sequencing": 4096,
    "subgoal_decomposition": 4096,
}


def clean_qwen_output(text):
    if text is None:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    fence = re.match(r"^```(?:json|python)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text


def load_existing(path):
    candidates = [path]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        candidates.append(tmp_path)

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

    if not best_rows:
        return {}
    return {row["identifier"]: row.get("llm_output", "") for row in best_rows}


def atomic_write_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=4), encoding="utf-8")
    os.replace(tmp, path)


def output_name_for_prompt_file(prompt_file):
    return prompt_file.name.replace("_prompts.json", "_outputs.json")


def sharded_output_name_for_prompt_file(prompt_file, args):
    name = output_name_for_prompt_file(prompt_file)
    if args.num_shards <= 1 and not args.shard_output:
        return name
    stem = name[:-5] if name.endswith(".json") else name
    return f"{stem}.shard{args.shard_index}-of-{args.num_shards}.json"


def iter_prompt_files(prompt_dir, selected):
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


def task_key(prompt_file):
    name = prompt_file.name.replace("_prompts.json", "")
    for key in TASK_MAX_TOKENS:
        if key in name:
            return key
    return None


def max_tokens_for_file(args, prompt_file):
    if args.max_tokens is not None:
        return args.max_tokens
    return TASK_MAX_TOKENS.get(task_key(prompt_file), args.default_max_tokens)


def build_chat_prompt(tokenizer, raw_prompt, no_think):
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
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return content


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def main():
    parser = argparse.ArgumentParser(
        description="Generate EAI submission JSON files with vLLM offline batch inference and a LoRA adapter."
    )
    parser.add_argument("--prompt-dir", default="llm_prompts")
    parser.add_argument("--output-dir", default="sample_submission_qwen3_32b_awq_vllm_lora_offline")
    parser.add_argument("--model", default="/workspace/models/Qwen3-32B-AWQ")
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--lora-name", default="axistilted2-eai")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--default-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N prompts per file.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split each prompt file into N round-robin shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for this worker.")
    parser.add_argument("--shard-output", action="store_true", help="Write shard-suffixed output JSON files.")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional prompt files/tasks, e.g. virtualhome_action_sequencing_prompts.json.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output files.")
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
    print(f"lora={args.lora_name}:{args.lora_path}")
    print(f"prompt_files={len(prompt_files)}")
    print(f"output_dir={output_dir}")
    print(f"batch_size={args.batch_size}")
    print(f"max_model_len={args.max_model_len}")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=False,
        enable_lora=True,
        max_loras=1,
    )
    tokenizer = llm.get_tokenizer()
    lora_request = LoRARequest(args.lora_name, 1, args.lora_path)

    started_at = time.time()
    for prompt_file in prompt_files:
        prompts = json.loads(prompt_file.read_text(encoding="utf-8"))
        if args.limit is not None:
            prompts = prompts[: args.limit]
        if args.num_shards > 1:
            prompts = [
                prompt
                for index, prompt in enumerate(prompts)
                if index % args.num_shards == args.shard_index
            ]

        output_path = output_dir / sharded_output_name_for_prompt_file(prompt_file, args)
        existing = {} if args.no_resume else load_existing(output_path)
        rows = [
            {"identifier": prompt["identifier"], "llm_output": existing[prompt["identifier"]]}
            for prompt in prompts
            if prompt["identifier"] in existing and existing[prompt["identifier"]]
        ]
        pending = [prompt for prompt in prompts if prompt["identifier"] not in existing or not existing[prompt["identifier"]]]
        max_tokens = max_tokens_for_file(args, prompt_file)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max_tokens,
        )

        print(f"\n== {prompt_file.name}: total={len(prompts)} resume={len(rows)} pending={len(pending)}")
        print(f"output={output_path} max_tokens={max_tokens}")
        atomic_write_json(output_path, rows)

        for start, batch in batched(pending, args.batch_size):
            batch_prompts = [
                build_chat_prompt(tokenizer, prompt["llm_prompt"], args.no_think)
                for prompt in batch
            ]
            outputs = llm.generate(
                batch_prompts,
                sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
            for prompt, output in zip(batch, outputs):
                text = output.outputs[0].text if output.outputs else ""
                answer = clean_qwen_output(text)
                existing[prompt["identifier"]] = answer
                rows.append({"identifier": prompt["identifier"], "llm_output": answer})

            atomic_write_json(output_path, rows)
            rows = [
                {"identifier": prompt["identifier"], "llm_output": existing.get(prompt["identifier"], "")}
                for prompt in prompts
                if prompt["identifier"] in existing
            ]
            done = len(rows)
            elapsed = time.time() - started_at
            print(
                f"[{done}/{len(prompts)}] batch_start={start + 1} "
                f"batch={len(batch)} elapsed={elapsed:.1f}s"
            )


if __name__ == "__main__":
    main()
