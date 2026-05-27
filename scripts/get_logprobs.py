import json
import re
import argparse
from pathlib import Path
from leakaware.logprobs import LogProbsPrompt
from leakaware.config import PREFILL_SUFFIX

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def _sanitize_name(name: str) -> str:
    return re.sub(r'[\\/*?":<>|]', "_", name)


def count_done_records(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0

    done = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                done += 1
            except json.JSONDecodeError:
                continue
    return done


def build_save_path(args, model, msg_path, prefill_type) -> Path:
    model_name = _sanitize_name(Path(model).name)
    msg_stem = _sanitize_name(Path(msg_path).stem)
    if prefill_type == "universe":
        task = prefill_type + "_" + (args.msg_path.strip("/").split("/")[-1] if args.msg_path else args.msg_dir.strip("/").split("/")[-1])
    else:
        task = prefill_type

    file_name = f"{msg_stem}-prefill-{model_name}.jsonl"

    sub_resu_path = "intent" if args.intent else "specific"
    final_dir = Path(args.output_root) / sub_resu_path / model_name / task
    final_dir.mkdir(parents=True, exist_ok=True)

    return final_dir / file_name


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_messages(msg_path, lpt, args):
    messages = load_json(msg_path)
    print("The length of messages: ", len(messages))

    save_path = build_save_path(args, args.model_dir, msg_path, args.prefill_type)
    print("The save path is: ", save_path)

    total = len(messages)
    done = count_done_records(save_path)
    start = done
    print(f"Total: {total}, Done: {done}, Start from: {start}")

    if start >= total:
        print("All messages already processed. Exit.")
        return

    for idx in range(start, total):
        raw_msg = messages[idx]["messages"]
        msg = lpt.apply_prefill(raw_msg, prefill_type=args.prefill_type, intent=args.intent)

        out = lpt.get_logprobs(msg)
        record = {
            "input": msg,
            "output": out["text"],
            "prefill_logprobs_dict": out["prefill_logprobs_dict"],
            "logprobs": out["all_logprobs"],
            "rank": out["all_rank"],
        }

        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[{idx + 1}/{total}] done")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract prefill logprobs via vLLM (server or offline)")

    # model
    parser.add_argument("--model_dir", type=str, default=None, help="model path (required for Offline mode; auto-detected in server mode)")

    # server mode args
    parser.add_argument("--base_url", type=str, default=None, help="vLLM server URL (e.g. http://127.0.0.1:22991/v1). If set, use server mode; otherwise use Offline mode")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for server mode")

    # Offline mode args
    parser.add_argument("--dtype", type=str, default="auto", help="[Offline] vLLM dtype: auto / float16 / bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="[Offline] vLLM tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="[Offline] vLLM GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=None, help="[Offline] vLLM max_model_len")

    # generation params
    parser.add_argument("--temperature", type=float, default=0, help="temperature (does not affect prompt_logprobs)")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p")
    parser.add_argument("--reasoning_parser", type=str, choices=["deepseek_r1", "qwen3", "none"], default="none",
                        help="reasoning parser, decides the prefill think-prefix")

    # task params
    parser.add_argument("--prefill_type", type=str, choices=["sys_prompt", "rag_chunks", "universe"], default="universe",
                        help="prefill type: sys_prompt, rag_chunks or universe")
    parser.add_argument("--intent", action="store_true", help="use intent-based suffix")
    parser.add_argument("--msg_path", type=str, default=None, help="path to input messages json file")
    parser.add_argument("--msg_dir", type=str, default="data_input/", help="directory of input messages; used if --msg_path is not set")
    parser.add_argument("--output_root", type=str, default="logprobs", help="output root dir")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.base_url:
        # server mode
        lpt = LogProbsPrompt(
            reasoning_parser=args.reasoning_parser,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if not args.model_dir:
            args.model_dir = lpt.model_root
    else:
        # offline mode
        if not args.model_dir:
            raise ValueError("--model_dir is required in offline mode (no --base_url provided)")
        from vllm import LLM
        llm = LLM(
            model=args.model_dir,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            reasoning_parser=None if args.reasoning_parser == "none" else args.reasoning_parser,
        )
        lpt = LogProbsPrompt(reasoning_parser=args.reasoning_parser, llm=llm)

    # save meta
    print("===== Build Meta Information =====")
    meta_save_path = build_save_path(args, args.model_dir, "meta", args.prefill_type)
    suffix = PREFILL_SUFFIX.get(args.prefill_type, "") if args.intent else ""
    meta_info = {
        "model": args.model_dir,
        "reasoning_parser": args.reasoning_parser,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prefill_type": args.prefill_type,
        "intent_based": args.intent,
        "suffix": suffix
    }
    with open(meta_save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta_info, ensure_ascii=False) + "\n")

    # process
    print("===== Start Processing Messages =====")
    if args.msg_path:
        process_messages(args.msg_path, lpt, args)
    elif args.msg_dir:
        msg_dir = Path(args.msg_dir)
        json_files = list(msg_dir.glob("*.json"))
        print(f"Found {len(json_files)} json files in {msg_dir}")

        for json_file in json_files:
            print(f"Processing file: {json_file}")
            process_messages(json_file, lpt, args)
    else:
        print("Please provide either --msg_path or --msg_dir")
        return


if __name__ == "__main__":
    main()
