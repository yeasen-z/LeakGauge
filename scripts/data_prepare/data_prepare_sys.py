from pathlib import Path
import random
import itertools
import argparse

BASE_DIR = Path(__file__).resolve().parent

random.seed(42)

import json
import pandas as pd

SMALL_LIMIT = 2500


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true",
                        help="large mode: use the full dataset without capping")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_list_of_list(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def csv_to_json(csv_path, json_path, encoding="utf-8"):
    df = pd.read_csv(csv_path, encoding=encoding)

    df.to_json(
        json_path,
        orient="records",
        force_ascii=False,  
        indent=2
    )

def load_jsonl(path, *, skip_invalid=True, encoding="utf-8"):
    data = []
    with open(path, "r", encoding=encoding) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                if skip_invalid:
                    continue
                else:
                    raise RuntimeError(f"JSON parse error at line {lineno}") from e
    return data

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def cap_data(data, small):
    if small and len(data) > SMALL_LIMIT:
        return random.sample(data, SMALL_LIMIT)
    return data


sys_prompt = load_jsonl(BASE_DIR / "../../demo/content/prompts.chat.jsonl")

raccoon_attacks = load_jsonl(BASE_DIR / "../../demo/attacks/sys.raccoon.jsonl")
lma_attacks = load_jsonl(BASE_DIR / "../../demo/attacks/sys.lma.jsonl")


all_attacks = raccoon_attacks + lma_attacks
random.shuffle(all_attacks)
split_v = int(0.7 * len(all_attacks))
seen_inst = all_attacks[:split_v]
unseen_inst = all_attacks[split_v:]


random.shuffle(sys_prompt)
split_v = int(0.7 * len(sys_prompt))
seen_sys_prompt = sys_prompt[:split_v]
unseen_sys_prompt = sys_prompt[split_v:]


data_query = load_jsonl(BASE_DIR / "../../demo/content/rag_samples.wikitext.jsonl")


def data_prepare_sys(sys_prompt, adv_inst):
    attack_system = []

    for sys_prompt_item in sys_prompt:
        for adv_inst_item in adv_inst:
            attack_system.append({
                "messages": [
                    {"role": "system", "content": sys_prompt_item["prompt"] },
                    {"role": "user", "content": adv_inst_item["prompt"] }
                ],
                "privacy": [sys_prompt_item["prompt"]]
            })
    
    benign_system = []
    for sys_prompt_item in sys_prompt:
        for query_item in random.sample(data_query, len(adv_inst)-1):            
            benign_system.append({
                "messages": [
                    {"role": "system", "content": sys_prompt_item["prompt"] },
                    {"role": "user", "content": query_item["question"] }
                ],
                "privacy": [sys_prompt_item["prompt"]]
            })

    return attack_system, benign_system


def main():
    args = parse_args()

    train_val_attack, train_val_benign = data_prepare_sys(seen_sys_prompt, seen_inst)
    unseen_sys_attack, unseen_sys_benign = data_prepare_sys(unseen_sys_prompt, seen_inst)
    unseen_inst_attack, unseen_inst_benign = data_prepare_sys(seen_sys_prompt, unseen_inst)
    unseen_all_attack, unseen_all_benign = data_prepare_sys(unseen_sys_prompt, unseen_inst)

    output_dir = BASE_DIR / "../../data_input/sys_mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train_val_attack.json": train_val_attack,
        "train_val_benign.json": train_val_benign,
        "unseen_content_attack.json": unseen_sys_attack,
        "unseen_content_benign.json": unseen_sys_benign,
        "unseen_inst_attack.json": unseen_inst_attack,
        "unseen_inst_benign.json": unseen_inst_benign,
        "unseen_all_attack.json": unseen_all_attack,
        "unseen_all_benign.json": unseen_all_benign,
    }

    for file_name, data in datasets.items():
        capped = cap_data(data, args.large == False)
        save_list_of_list(capped, output_dir / file_name)
        print(f"{file_name}: {len(data)} -> {len(capped)}")

    print("Sys Done." + (" (large mode)" if args.large else ""))


if __name__ == "__main__":
    main()