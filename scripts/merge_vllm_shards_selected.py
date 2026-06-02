import argparse
import json
from pathlib import Path


def output_name_for_prompt_file(prompt_file):
    return prompt_file.name.replace("_prompts.json", "_outputs.json")


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


def main():
    parser = argparse.ArgumentParser(description="Merge selected round-robin vLLM shard outputs into EAI JSON files.")
    parser.add_argument("--prompt-dir", default="llm_prompts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--files", nargs="*", default=None)
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    prompt_files = iter_prompt_files(prompt_dir, args.files)
    if not prompt_files:
        raise SystemExit(f"No prompt files selected in {prompt_dir}")

    for prompt_file in prompt_files:
        prompts = json.loads(prompt_file.read_text(encoding="utf-8"))
        if args.limit is not None:
            prompts = prompts[: args.limit]

        merged = {}
        output_name = output_name_for_prompt_file(prompt_file)
        stem = output_name[:-5] if output_name.endswith(".json") else output_name
        for shard_index in range(args.num_shards):
            shard_path = output_dir / f"{stem}.shard{shard_index}-of-{args.num_shards}.json"
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            rows = json.loads(shard_path.read_text(encoding="utf-8"))
            for row in rows:
                merged[row["identifier"]] = row.get("llm_output", "")

        rows = []
        missing = []
        for prompt in prompts:
            identifier = prompt["identifier"]
            if identifier not in merged:
                missing.append(identifier)
            else:
                rows.append({"identifier": identifier, "llm_output": merged[identifier]})

        if missing:
            raise RuntimeError(f"{output_name}: missing {len(missing)} identifiers, first={missing[0]}")

        output_path = output_dir / output_name
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"{output_name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
