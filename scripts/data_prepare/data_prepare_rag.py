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


sys_prompt = load_jsonl( BASE_DIR / "../../demo/content/rag_prompt.jsonl" )
raccoon_attacks = load_jsonl( BASE_DIR / "../../demo/attacks/rag.raccoon.jsonl" )
lma_attacks = load_jsonl( BASE_DIR / "../../demo/attacks/rag.lma.jsonl" )
all_attacks = raccoon_attacks + lma_attacks
random.shuffle(all_attacks)
wikitext = load_jsonl( BASE_DIR / "../../demo/content/rag_samples.wikitext.jsonl" )
fiqa = load_jsonl( BASE_DIR / "../../demo/content/rag_samples.fiqa.jsonl" )
enronmail = load_jsonl( BASE_DIR / "../../demo/content/rag_samples.enronmail.jsonl" )
nfcorpus = load_jsonl( BASE_DIR / "../../demo/content/rag_samples.nfcorpus.jsonl" )
scifact = load_jsonl( BASE_DIR / "../../demo/content/rag_samples.scifact.jsonl" )

seen_content = wikitext + fiqa
unseen_content = enronmail + nfcorpus + scifact

split_v = int(0.7 * len(all_attacks))

seen_inst = all_attacks[:split_v]
unseen_inst = all_attacks[split_v:]

def data_prepare_rag(rag_samples, adv_inst, rag_sys_prompt):

    attack_rag = []
    benign_rag = []

    elements = [0, 1, 2, 3, 4]
    chunk_groups = [
        list(p)
        for p in itertools.combinations(elements, 3)
    ]

    for user_query_item in rag_samples:

        chunks = user_query_item["chunks"]

        for group in random.sample(
            chunk_groups,
            k=min(8, len(chunk_groups))
        ):

            selected_chunks = "\n".join(
                [chunks[i] for i in group]
            )

            benign_rag.append({
                "messages": [
                    {
                        "role": "system",
                        "content": rag_sys_prompt[0]["sys_prompt"]
                    },
                    {
                        "role": "user",
                        "content": rag_sys_prompt[0]["usr_format"].format(
                            question=user_query_item["question"],
                            context=selected_chunks
                        )
                    }
                ],
                "privacy": chunks,
                "question": user_query_item["question"]
            })

    len_benign = len(benign_rag)

    sampled_rag_samples = random.sample(
        rag_samples,
        k=min(
            int(len_benign * 2 / len(adv_inst)),
            len(rag_samples)
        )
    )

    for user_query_item in sampled_rag_samples:

        chunks = [
            user_query_item["chunks"][i]
            for i in chunk_groups[0]
        ]

        for adv_inst_item in adv_inst:

            adv_prompt = adv_inst_item["prompt"].replace(
                "{text}",
                user_query_item["question"]
            )

            attack_rag.append({
                "messages": [
                    {
                        "role": "system",
                        "content": rag_sys_prompt[0]["sys_prompt"]
                    },
                    {
                        "role": "user",
                        "content": rag_sys_prompt[0]["usr_format"].format(
                            question=adv_prompt,
                            context="\n".join(chunks)
                        )
                    }
                ],
                "privacy": chunks,
                "question": user_query_item["question"]
            })

    return attack_rag, benign_rag


def main():
    args = parse_args()

    train_val_attack, train_val_benign = data_prepare_rag(
        seen_content,
        seen_inst,
        sys_prompt
    )

    unseen_rag_attack, unseen_rag_benign = data_prepare_rag(
        unseen_content,
        seen_inst,
        sys_prompt
    )

    unseen_inst_attack, unseen_inst_benign = data_prepare_rag(
        seen_content,
        unseen_inst,
        sys_prompt
    )

    unseen_all_attack, unseen_all_benign = data_prepare_rag(
        unseen_content,
        unseen_inst,
        sys_prompt
    )

    output_dir = BASE_DIR / "../../data_input/rag_mixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train_val_attack.json": train_val_attack,
        "train_val_benign.json": train_val_benign,
        "unseen_content_attack.json": unseen_rag_attack,
        "unseen_content_benign.json": unseen_rag_benign,
        "unseen_inst_attack.json": unseen_inst_attack,
        "unseen_inst_benign.json": unseen_inst_benign,
        "unseen_all_attack.json": unseen_all_attack,
        "unseen_all_benign.json": unseen_all_benign,
    }

    for file_name, data in datasets.items():
        capped = cap_data(data, args.large == False)
        save_list_of_list(capped, output_dir / file_name)
        print(f"{file_name}: {len(data)} -> {len(capped)}")

    print("RAG Done." + (" (large mode)" if args.large else ""))


if __name__ == "__main__":
    main()