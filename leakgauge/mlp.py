import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
import json
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

class BinaryMlp(nn.Module):
    def __init__(self, input_dim):
        """
        input_dim: 输入特征维度
        """
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits
    
    def save(self, save_path, meta=None):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        save_dict = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
        }

        torch.save(save_dict, save_path)

        if meta is not None:
            meta_path = os.path.splitext(save_path)[0] + ".meta.json"

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f">>> Best model saved to: {save_path}")

    @classmethod
    def load(cls, load_path, map_location="cpu"):
        checkpoint = torch.load(load_path, map_location=map_location)

        model = cls(
            input_dim=checkpoint["input_dim"]
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.to(map_location)

        print(f">>> Model loaded from: {load_path} (device: {map_location})")

        return model


class MlpTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=1e-3,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=10, fix_threshold=0.4):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,    
            eta_min=1e-4      
        )

        for epoch in range(epochs):

            self.model.train()

            total_loss = 0

            for x, y in self.train_loader:

                x = x.to(self.device)
                y = y.to(self.device).float().unsqueeze(1)

                logits = self.model(x)

                loss = self.criterion(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()

            if self.val_loader:
                self.evaluate(fix_threshold)

    def evaluate(self, fix_threshold=0.4):

        self.model.eval()

        def evaluate_loader(loader):

            all_probs = []
            all_labels = []

            with torch.no_grad():

                for x, y in loader:

                    x = x.to(self.device)
                    y = y.to(self.device).float().unsqueeze(1)

                    logits = self.model(x)

                    probs = torch.sigmoid(logits)

                    all_probs.extend(probs.cpu().numpy().flatten())
                    all_labels.extend(y.cpu().numpy().flatten())

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)

            if fix_threshold is not None:
                best_threshold = fix_threshold
            else:
                precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
                f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
                best_idx = f1_scores.argmax()
                best_threshold = thresholds[best_idx]


            preds = [1 if p > best_threshold else 0 for p in all_probs]
            acc = accuracy_score(all_labels, preds)
            precision = precision_score(all_labels, preds)
            recall = recall_score(all_labels, preds)
            f1 = f1_score(all_labels, preds)

            try:
                auc = roc_auc_score(all_labels, all_probs)
                pr_auc = average_precision_score(all_labels, all_probs)
            except:
                auc = 0
                pr_auc = 0

            return best_threshold, acc, precision, recall, f1, auc, pr_auc

        train_best_threshold, train_acc, train_p, train_r, train_f1, train_auc, train_pr_auc = evaluate_loader(self.train_loader)
        val_best_threshold, val_acc, val_p, val_r, val_f1, val_auc, val_pr_auc = evaluate_loader(self.val_loader)

        print(
            f"Train -> best_threshold {train_best_threshold:.4f} | ACC {train_acc:.4f} | P {train_p:.4f} | R {train_r:.4f} | F1 {train_f1:.4f} | AUC {train_auc:.4f} | PR-AUC {train_pr_auc:.4f}"
        )

        print(
            f"Val   -> best_threshold {val_best_threshold:.4f} | ACC {val_acc:.4f} | P {val_p:.4f} | R {val_r:.4f} | F1 {val_f1:.4f} | AUC {val_auc:.4f} | PR-AUC {val_pr_auc:.4f}"
        )
       