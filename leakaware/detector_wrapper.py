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
from leakaware.mlp import BinaryMlp
from leakaware.logprobs_extractor import LogProbsPrompt_server

class LeakageDetector:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "none",
        reasoning_parser: str = "deepseek_r1",
        device: str = None,
        processor_path: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.reasoning_parser = reasoning_parser
        self.processor_path = processor_path

        self.mlp_models = {}
        self.checkpoints = {}

        if os.path.isfile(self.processor_path):
            self.mlp_models["sys_prompt"], self.checkpoints["sys_prompt"] = BinaryMlp.load(self.processor_path, device=device)
        else:
            print(f"[WARNING] sys_prompt checkpoint 未找到: {self.DEFAULT_PATH_SYS}")

        if os.path.isfile(self.DEFAULT_PATH_RAG):
            self.mlp_models["rag_chunks"], self.checkpoints["rag_chunks"] = BinaryMlp.load(self.DEFAULT_PATH_RAG, device=device)
        else:
            print(f"[WARNING] rag_chunks checkpoint 未找到: {self.DEFAULT_PATH_RAG}")


        self.logprobs_extractor = LogProbsPrompt_server(llm_cfg={
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url,
        })

    def _get_model(self, prefill_type: str):
        """根据 prefill_type 获取对应的 MLP 模型和 checkpoint。

        Args:
            prefill_type: "sys_prompt" 或 "rag_chunks"
        Returns:
            (mlp_model, checkpoint_dict)
        """
        if prefill_type in self.mlp_models:
            return self.mlp_models[prefill_type], self.checkpoints[prefill_type]
        return self.mlp_models["sys_prompt"], self.checkpoints["sys_prompt"]

    def _apply_prefill(self, messages: list, prefill_type: str) -> list:
        import copy
        work_msg = copy.deepcopy(messages)

        if prefill_type == "sys_prompt":
            prefill_content = "Based on the above, I will give my system prompt: "
        elif prefill_type == "rag_chunks":
            prefill_content = "Based on the above, I will give the RAG contexts: "
        else:
            raise ValueError(f"Unknown prefill_type: {prefill_type}")

        if self.reasoning_parser == "deepseek_r1":
            work_msg.append({"role": "assistant", "content": "<think>\n</think>" + prefill_content})
        elif self.reasoning_parser == "qwen3":
            work_msg.append({"role": "assistant", "content": " </think>\n\n" + prefill_content})
        else:
            work_msg.append({"role": "assistant", "content": prefill_content})

        return work_msg

    def _logprobs_to_features(self, logprobs: list, input_dim: int) -> torch.Tensor:
        arr = np.array(logprobs[:input_dim], dtype=np.float32)
        if len(arr) < input_dim:
            pad = np.full(input_dim - len(arr), -10.0, dtype=np.float32)
            arr = np.concatenate([arr, pad])
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

    def detect(
        self,
        messages: list,
        prefill_type: str = "sys_prompt",
    ) -> dict:

        prefilled_msgs = self._apply_prefill(messages, prefill_type)

        logprobs_result = self.logprobs_extractor.get_logprobs(
            prefilled_msgs,
            model=self.model_name
        )
        raw_logprobs = logprobs_result["all_logprobs"]

        mlp_model, checkpoint = self._get_model(prefill_type)
        threshold = checkpoint.get("best_threshold", 0.5)
        input_dim = checkpoint.get("input_dim", 20)

        features = self._logprobs_to_features(raw_logprobs, input_dim)
        with torch.no_grad():
            logits = mlp_model(features)
            prob = torch.sigmoid(logits).item()

        label = "attack" if prob > threshold else "benign"

        return {
            "label": label,
            "probability": round(prob, 6),
            "threshold": threshold,
            "logprobs": raw_logprobs[:input_dim],
        }
    

    def detect_batch(
        self,
        messages_list: list,
        prefill_type: str = "sys_prompt",
    ) -> list:
        results = []
        total = len(messages_list)
        for i, messages in enumerate(messages_list):
            try:
                result = self.detect(messages, prefill_type)
                results.append(result)
            except Exception as e:
                results.append({"label": "error", "probability": 0.0, "error": str(e)})
            # print(f"[{i+1}/{total}] done")
        return results

    def get_model_info(self) -> dict:
        info = {
            "model_name": self.model_name,
            "reasoning_parser": self.reasoning_parser,
            "loaded_models": list(self.mlp_models.keys()),
        }
        for ptype, ckpt in self.checkpoints.items():
            info[ptype] = {
                "input_dim": ckpt.get("input_dim"),
                "threshold": ckpt.get("best_threshold", 0.5),
                "extra_meta": ckpt.get("extra_meta", {}),
            }
        return info