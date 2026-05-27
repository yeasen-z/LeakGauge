import os
import json
import copy
import argparse
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from leakaware.mlp import BinaryMlp, MlpTrainer
from leakaware.dataset import DatasetManager

def evaluate_model(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_labels.extend(y.cpu().numpy().flatten().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    if len(np.unique(all_labels)) < 2:
        return None

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    target_fpr = 0.05

    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) > 0:
        tpr_at_fpr5 = float(np.max(tpr[valid_idx]))
    else:
        tpr_at_fpr5 = 0.0

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)

    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])


    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(roc_auc),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "pr_auc": float(pr_auc),
        "probs": all_probs.tolist(),
        "labels": all_labels.tolist(),
        "tpr_at_fpr5": tpr_at_fpr5,
        "best_f1": best_f1,
        "best_threshold": float(_[best_idx])
    }

def load_jsonl(path, skip_invalid=True):
    data = []
    if not os.path.exists(path):
        print(f"Warning: File not found -> {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except:
                if skip_invalid: continue
                else: raise
    return data

def get_label_data(attack_datas, benign_datas, logits_dim=20):
    if not attack_datas or not benign_datas:
        return None, None, None
    all_datas = attack_datas + benign_datas
    labels = [1] * len(attack_datas) + [0] * len(benign_datas)
    prefill_logits = [np.array(item["logprobs"][:logits_dim]) for item in all_datas]
    return all_datas, labels, prefill_logits


def save_best_model(model, save_path, meta=None):
    model.save(save_path, meta)
    print(f">>> Best model saved to: {save_path}")


def train_test(data_dir, device="cuda:0", epochs=10, training_batch=64, train_lr=1e-3,
               model_save_dir=None):
    path_parts = data_dir.strip("/").split("/")
    if len(path_parts) < 4:
        print("Warning: Invalid path format (expected: prefill_type/model_name/task)")
        return

    prefill_type = path_parts[1]   # e.g., 'general'
    model_name = path_parts[2]     # e.g., 'Qwen2.5-14B-Instruct'
    task = path_parts[3]           # e.g., 'sys_prompt'

    print(f">>> Task detected: {task} | Model: {model_name} | Type: {prefill_type}")

    meta_info = load_jsonl(os.path.join(data_dir, f"meta-prefill-{model_name}.jsonl"), skip_invalid=False)
    
    data_types = ["train_val", "unseen_content", "unseen_inst", "unseen_all"]
    all_dataset_packages = {}


    for dtype in data_types:
        atk_file = f"{dtype}_attack-prefill-{model_name}.jsonl"
        ben_file = f"{dtype}_benign-prefill-{model_name}.jsonl"

        atk_data = load_jsonl(os.path.join(data_dir, atk_file))
        ben_data = load_jsonl(os.path.join(data_dir, ben_file))

        _, labels, logits = get_label_data(atk_data, ben_data)
        if labels:
            all_dataset_packages[dtype] = (logits, labels)

    if "train_val" not in all_dataset_packages:
        print("Warning: Unable to find training data train_val, cannot start training.")
        return

    input_dim = 20
    model = BinaryMlp(input_dim).to(device)

    train_logits, train_labels = all_dataset_packages["train_val"]

    dm = DatasetManager(train_logits, train_labels, batch_size=training_batch,
                        target_data_len=input_dim, test_ratio=0.2, device=device)

    trainer = MlpTrainer(model, dm.train_loader, dm.test_loader, lr=train_lr, device=device)

    print("\n--- Training ---")
    best_auc = -1.0
    best_state = None
    best_epoch = -1
    best_val_metrics = None
    for ep in range(epochs):
        trainer.train(1, fix_threshold=None)
        val_metrics = evaluate_model(model, dm.test_loader, device)
        if val_metrics is None:
            continue
        val_auc = val_metrics["auc"]
        print(f"[Epoch {ep + 1}/{epochs}] val ROC AUC = {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep + 1
            best_val_metrics = val_metrics

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f">>> Best epoch: {best_epoch} | best val ROC AUC = {best_auc:.4f}")

    if model_save_dir is not None and best_state is not None:
        save_path = os.path.join(model_save_dir, prefill_type, model_name, task, "best_model.pt")
        meta = {
            "prefill_type": prefill_type,
            "model_name": model_name,
            "task": task,
            "input_dim": input_dim,
            "best_epoch": best_epoch,
            "best_val_auc": best_auc,
            "best_val_pr_auc": best_val_metrics["pr_auc"] if best_val_metrics else None,
            "best_val_f1": best_val_metrics["best_f1"] if best_val_metrics else None,
            "best_threshold": best_val_metrics["best_threshold"] if best_val_metrics else None,
            "train_lr": train_lr,
            "training_batch": training_batch,
            "epochs": epochs,
            "reasoning_parser": meta_info[0]["reasoning_parser"], 
            "prefill_type": meta_info[0]["prefill_type"], 
            "intent_based": meta_info[0]["intent_based"], 
            "suffix": meta_info[0]["suffix"]
        }
        save_best_model(model, save_path, meta)

    print("\n--- Batch Evaluation Results ---")

    for dtype in data_types:
        if dtype not in all_dataset_packages:
            continue

        print(f"\n Evaluating: {dtype}")

        if dtype == "train_val":
            current_test_loader = dm.test_loader
            print("Note: For the train_val dataset, only 20 percent of the data is extracted as an isolated validation set for evaluation.")
        else:
            test_logits, test_labels = all_dataset_packages[dtype]
            eval_dm = DatasetManager(test_logits, test_labels, batch_size=1,
                                    target_data_len=input_dim, test_ratio=0.99, device=device)
            current_test_loader = eval_dm.test_loader

        metrics = evaluate_model(model, current_test_loader, device)

        print(f"[{dtype.upper()}]:", f"ROC AUC = {metrics['auc']:.3f}", f"PR AUC = {metrics['pr_auc']:.3f}")
        print(f"TPR@FPR=5% = {metrics['tpr_at_fpr5']:.3f}", f"Best F1 = {metrics['best_f1']:.3f} at threshold {metrics['best_threshold']:.3f}")

    return model

def test(data_dir, mlp, device="cuda:0"):
    path_parts = data_dir.strip("/").split("/")
    if len(path_parts) < 4:
        print("Warning: Invalid path format (expected: prefill_type/model_name/task)")
        return

    prefill_type = path_parts[1]   # e.g., 'general'
    model_name = path_parts[2]     # e.g., 'Qwen2.5-14B-Instruct'
    task = path_parts[3]           # e.g., 'sys_prompt'

    print(f">>> Task detected: {task} | Model: {model_name} | Type: {prefill_type}")

    data_types = ["unseen_all"]
    all_dataset_packages = {}


    for dtype in data_types:
        atk_file = f"{dtype}_attack-prefill-{model_name}.jsonl"
        ben_file = f"{dtype}_benign-prefill-{model_name}.jsonl"

        atk_data = load_jsonl(os.path.join(data_dir, atk_file))
        ben_data = load_jsonl(os.path.join(data_dir, ben_file))

        _, labels, logits = get_label_data(atk_data, ben_data)
        if labels:
            all_dataset_packages[dtype] = (logits, labels)

    input_dim = 20
    model = mlp.to(device)

    for dtype in data_types:
        if dtype not in all_dataset_packages:
            continue

        test_logits, test_labels = all_dataset_packages[dtype]
        eval_dm = DatasetManager(test_logits, test_labels, batch_size=1,
                                target_data_len=input_dim, test_ratio=0.99, device=device)
        current_test_loader = eval_dm.test_loader

        metrics = evaluate_model(model, current_test_loader, device)

        print(f"[{dtype.upper()}]:", f"ROC AUC = {metrics['auc']:.3f}", f"PR AUC = {metrics['pr_auc']:.3f}")
        print(f"TPR@FPR=5% = {metrics['tpr_at_fpr5']:.3f}", f"Best F1 = {metrics['best_f1']:.3f} at threshold {metrics['best_threshold']:.3f}")

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str,
                        default="results_verbatim/general/Llama-3.1-8B-Instruct/sys_prompt",
                        help="Training data directory")
    parser.add_argument("--extra_test_path", type=str, nargs="*", default=None,
                        help="Extra test directories, can pass multiple separated by spaces; skip testing if not passed")

    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--train_lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--training_batch", type=int, default=128, help="Training batch size")

    parser.add_argument("--model_save_dir", type=str, default="probe_models/",
                        help="Directory to save the best model; pass empty string to disable saving")

    parser.add_argument("--device", type=str, default=None,
                        help="Runtime device, e.g., cuda:0 / cpu; if not passed, will be selected automatically")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    model_save_dir = args.model_save_dir if args.model_save_dir else None

    mlp = train_test(
        args.target_path,
        epochs=args.epochs,
        train_lr=args.train_lr,
        training_batch=args.training_batch,
        device=device,
        model_save_dir=model_save_dir,
    )

    if args.extra_test_path:
        for p_i in args.extra_test_path:
            test(
                p_i,
                mlp=mlp,
                device=device,
            )


if __name__ == "__main__":
    main()