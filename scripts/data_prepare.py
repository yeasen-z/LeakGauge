from pathlib import Path
import random
import itertools
import argparse
import json

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DEMO_DIR = BASE_DIR / "../demo"

SMALL_LIMIT = 2500
SMALL_LIMIT_LEAK = 3000



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


def save_list_of_list(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cap_data(data, small, limit):
    if small and len(data) > limit:
        return random.sample(data, limit)
    return data



def data_prepare_sys(sys_prompt, adv_inst, benign_queries):
    attack_system = []
    for sys_prompt_item in sys_prompt:
        for adv_inst_item in adv_inst:
            attack_system.append({
                "messages": [
                    {"role": "system", "content": sys_prompt_item["prompt"]},
                    {"role": "user", "content": adv_inst_item["prompt"]}
                ],
                "privacy": [sys_prompt_item["prompt"]]
            })

    benign_system = []
    for sys_prompt_item in sys_prompt:
        for query_item in random.sample(benign_queries, len(adv_inst) - 1):
            benign_system.append({
                "messages": [
                    {"role": "system", "content": sys_prompt_item["prompt"]},
                    {"role": "user", "content": query_item["question"]}
                ],
                "privacy": [sys_prompt_item["prompt"]]
            })

    return attack_system, benign_system


def data_prepare_rag(rag_samples, adv_inst, rag_sys_prompt):
    attack_rag = []
    benign_rag = []

    elements = [0, 1, 2, 3, 4]
    chunk_groups = [list(p) for p in itertools.combinations(elements, 3)]

    for user_query_item in rag_samples:
        chunks = user_query_item["chunks"]
        for group in random.sample(chunk_groups, k=min(8, len(chunk_groups))):
            selected_chunks = "\n".join([chunks[i] for i in group])
            benign_rag.append({
                "messages": [
                    {"role": "system", "content": rag_sys_prompt[0]["sys_prompt"]},
                    {"role": "user", "content": rag_sys_prompt[0]["usr_format"].format(
                        question=user_query_item["question"],
                        context=selected_chunks
                    )}
                ],
                "privacy": chunks,
                "question": user_query_item["question"]
            })

    len_benign = len(benign_rag)
    sampled_rag_samples = random.sample(
        rag_samples,
        k=min(int(len_benign * 2 / len(adv_inst)), len(rag_samples))
    )

    for user_query_item in sampled_rag_samples:
        chunks = [user_query_item["chunks"][i] for i in chunk_groups[0]]
        for adv_inst_item in adv_inst:
            adv_prompt = adv_inst_item["prompt"].replace("{text}", user_query_item["question"])
            attack_rag.append({
                "messages": [
                    {"role": "system", "content": rag_sys_prompt[0]["sys_prompt"]},
                    {"role": "user", "content": rag_sys_prompt[0]["usr_format"].format(
                        question=adv_prompt,
                        context="\n".join(chunks)
                    )}
                ],
                "privacy": chunks,
                "question": user_query_item["question"]
            })

    return attack_rag, benign_rag



def load_sys_data(only_raccoon=False):
    sys_prompt = load_jsonl(DEMO_DIR / "content/prompts.chat.jsonl")
    raccoon = load_jsonl(DEMO_DIR / "attacks/sys.raccoon.jsonl")
    if only_raccoon:
        lma = []
    else:
        lma = load_jsonl(DEMO_DIR / "attacks/sys.lma.jsonl")
    all_attacks = raccoon + lma
    random.shuffle(all_attacks)

    split_v = int(0.7 * len(all_attacks))
    seen_inst = all_attacks[:split_v]
    unseen_inst = all_attacks[split_v:]

    random.shuffle(sys_prompt)
    split_v = int(0.7 * len(sys_prompt))
    seen_content = sys_prompt[:split_v]
    unseen_content = sys_prompt[split_v:]

    benign_queries = load_jsonl(DEMO_DIR / "content/rag_samples.wikitext.jsonl")

    return seen_content, unseen_content, seen_inst, unseen_inst, benign_queries


def load_rag_data(only_raccoon=False):
    rag_sys_prompt = load_jsonl(DEMO_DIR / "content/rag_prompt.jsonl")
    raccoon = load_jsonl(DEMO_DIR / "attacks/rag.raccoon.jsonl")
    if only_raccoon:
        lma = []
    else:
        lma = load_jsonl(DEMO_DIR / "attacks/rag.lma.jsonl")
    all_attacks = raccoon + lma
    random.shuffle(all_attacks)

    split_v = int(0.7 * len(all_attacks))
    seen_inst = all_attacks[:split_v]
    unseen_inst = all_attacks[split_v:]

    wikitext = load_jsonl(DEMO_DIR / "content/rag_samples.wikitext.jsonl")
    fiqa = load_jsonl(DEMO_DIR / "content/rag_samples.fiqa.jsonl")
    enronmail = load_jsonl(DEMO_DIR / "content/rag_samples.enronmail.jsonl")
    nfcorpus = load_jsonl(DEMO_DIR / "content/rag_samples.nfcorpus.jsonl")
    scifact = load_jsonl(DEMO_DIR / "content/rag_samples.scifact.jsonl")

    seen_content = wikitext + fiqa
    unseen_content = enronmail + nfcorpus + scifact

    return seen_content, unseen_content, seen_inst, unseen_inst, rag_sys_prompt



def run_sys(large, only_raccoon=False):
    seen_content, unseen_content, seen_inst, unseen_inst, benign_queries = load_sys_data(only_raccoon)

    train_val_attack, train_val_benign = data_prepare_sys(seen_content, seen_inst, benign_queries)
    unseen_content_attack, unseen_content_benign = data_prepare_sys(unseen_content, seen_inst, benign_queries)
    unseen_inst_attack, unseen_inst_benign = data_prepare_sys(seen_content, unseen_inst, benign_queries)
    unseen_all_attack, unseen_all_benign = data_prepare_sys(unseen_content, unseen_inst, benign_queries)

    output_dir = BASE_DIR / "../data_input/sys_mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train_val_attack.json": train_val_attack,
        "train_val_benign.json": train_val_benign,
        "unseen_content_attack.json": unseen_content_attack,
        "unseen_content_benign.json": unseen_content_benign,
        "unseen_inst_attack.json": unseen_inst_attack,
        "unseen_inst_benign.json": unseen_inst_benign,
        "unseen_all_attack.json": unseen_all_attack,
        "unseen_all_benign.json": unseen_all_benign,
    }

    for file_name, data in datasets.items():
        capped = cap_data(data, not large, SMALL_LIMIT)
        save_list_of_list(capped, output_dir / file_name)
        print(f"{file_name}: {len(data)} -> {len(capped)}")

    print("Sys Done." + (" (large mode)" if large else ""))


def run_rag(large, only_raccoon=False):
    seen_content, unseen_content, seen_inst, unseen_inst, rag_sys_prompt = load_rag_data(only_raccoon)

    train_val_attack, train_val_benign = data_prepare_rag(seen_content, seen_inst, rag_sys_prompt)
    unseen_content_attack, unseen_content_benign = data_prepare_rag(unseen_content, seen_inst, rag_sys_prompt)
    unseen_inst_attack, unseen_inst_benign = data_prepare_rag(seen_content, unseen_inst, rag_sys_prompt)
    unseen_all_attack, unseen_all_benign = data_prepare_rag(unseen_content, unseen_inst, rag_sys_prompt)

    output_dir = BASE_DIR / "../data_input/rag_mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train_val_attack.json": train_val_attack,
        "train_val_benign.json": train_val_benign,
        "unseen_content_attack.json": unseen_content_attack,
        "unseen_content_benign.json": unseen_content_benign,
        "unseen_inst_attack.json": unseen_inst_attack,
        "unseen_inst_benign.json": unseen_inst_benign,
        "unseen_all_attack.json": unseen_all_attack,
        "unseen_all_benign.json": unseen_all_benign,
    }

    for file_name, data in datasets.items():
        capped = cap_data(data, not large, SMALL_LIMIT)
        save_list_of_list(capped, output_dir / file_name)
        print(f"{file_name}: {len(data)} -> {len(capped)}")

    print("RAG Done." + (" (large mode)" if large else ""))


def run_leak(large, only_raccoon=False):
    sys_seen_content, sys_unseen_content, sys_seen_inst, sys_unseen_inst, benign_queries = load_sys_data(only_raccoon)
    rag_seen_content, rag_unseen_content, rag_seen_inst, rag_unseen_inst, rag_sys_prompt = load_rag_data(only_raccoon)

    sys_tv_atk, sys_tv_ben = data_prepare_sys(sys_seen_content, sys_seen_inst, benign_queries)
    sys_uc_atk, sys_uc_ben = data_prepare_sys(sys_unseen_content, sys_seen_inst, benign_queries)
    sys_ui_atk, sys_ui_ben = data_prepare_sys(sys_seen_content, sys_unseen_inst, benign_queries)
    sys_ua_atk, sys_ua_ben = data_prepare_sys(sys_unseen_content, sys_unseen_inst, benign_queries)

    rag_tv_atk, rag_tv_ben = data_prepare_rag(rag_seen_content, rag_seen_inst, rag_sys_prompt)
    rag_uc_atk, rag_uc_ben = data_prepare_rag(rag_unseen_content, rag_seen_inst, rag_sys_prompt)
    rag_ui_atk, rag_ui_ben = data_prepare_rag(rag_seen_content, rag_unseen_inst, rag_sys_prompt)
    rag_ua_atk, rag_ua_ben = data_prepare_rag(rag_unseen_content, rag_unseen_inst, rag_sys_prompt)

    output_dir = BASE_DIR / "../data_input/leak_mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train_val_attack.json": sys_tv_atk + rag_tv_atk,
        "train_val_benign.json": sys_tv_ben + rag_tv_ben,
        "unseen_content_attack.json": sys_uc_atk + rag_uc_atk,
        "unseen_content_benign.json": sys_uc_ben + rag_uc_ben,
        "unseen_inst_attack.json": sys_ui_atk + rag_ui_atk,
        "unseen_inst_benign.json": sys_ui_ben + rag_ui_ben,
        "unseen_all_attack.json": sys_ua_atk + rag_ua_atk,
        "unseen_all_benign.json": sys_ua_ben + rag_ua_ben,
    }

    for file_name, data in datasets.items():
        capped = cap_data(data, not large, SMALL_LIMIT_LEAK)
        save_list_of_list(capped, output_dir / file_name)
        print(f"{file_name}: {len(data)} -> {len(capped)}")

    print("Leak Done." + (" (large mode)" if large else ""))



def parse_args():
    parser = argparse.ArgumentParser(description="Data preparation for LeakGauge")
    parser.add_argument("--mode", type=str, required=True, choices=["sys", "rag", "leak"],
                        help="data mode: sys (system prompt), rag (RAG chunks), or leak (mixed)")
    parser.add_argument("--large", action="store_true",
                        help="large mode: use the full dataset without capping")
    parser.add_argument("--only_raccoon", action="store_true",
                        help="only use Raccoon attack samples")
    return parser.parse_args()


def main():
    args = parse_args()

    runners = {
        "sys": run_sys,
        "rag": run_rag,
        "leak": run_leak,
    }

    runners[args.mode](args.large, args.only_raccoon)


if __name__ == "__main__":
    main()
