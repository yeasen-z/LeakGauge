import torch
import numpy as np
import os
import json
from leakaware.mlp import BinaryMlp
from leakaware.logprobs import LogProbsPrompt


class LeakageDetector:
    """
    End-to-end leakage detector. Supports both vLLM server and offline mode.

    Server mode:
        detector = LeakageDetector(
            base_url="http://127.0.0.1:22991/v1",
            processor_path="probe_models/.../best_model.pt"
        )

    Offline mode:
        from vllm import LLM
        llm = LLM(model="path/to/model")
        detector = LeakageDetector(
            llm=llm,
            processor_path="probe_models/.../best_model.pt"
        )
    """

    def __init__(
        self,
        processor_path: str,
        # server mode
        base_url: str = None,
        api_key: str = "none",
        # offline mode
        llm=None,
        # misc
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # load MLP checkpoint and meta
        self.mlp_models = {}
        self.meta = {}
        reasoning_parser = "none"

        if os.path.isfile(processor_path):
            self.mlp_models["sys_prompt"] = BinaryMlp.load(
                processor_path, map_location=device
            )
            meta_path = os.path.splitext(processor_path)[0] + ".meta.json"
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.meta["sys_prompt"] = json.load(f)
                reasoning_parser = self.meta["sys_prompt"].get("reasoning_parser", "none")
            else:
                self.meta["sys_prompt"] = {}
        else:
            raise FileNotFoundError(f"Processor checkpoint not found: {processor_path}")

        # init logprobs extractor (server or offline)
        if base_url is not None:
            self.logprobs_extractor = LogProbsPrompt(
                reasoning_parser=reasoning_parser,
                base_url=base_url,
                api_key=api_key,
            )
        elif llm is not None:
            self.logprobs_extractor = LogProbsPrompt(
                reasoning_parser=reasoning_parser,
                llm=llm,
            )
        else:
            raise ValueError("Must provide either base_url (server mode) or llm (offline mode)")

        self.model_name = self.logprobs_extractor.model or self.logprobs_extractor.model_root

    def _get_model(self, prefill_type: str):
        if prefill_type in self.mlp_models:
            return self.mlp_models[prefill_type], self.meta.get(prefill_type, {})
        return self.mlp_models["sys_prompt"], self.meta.get("sys_prompt", {})

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

        prefilled_msgs = self.logprobs_extractor.apply_prefill(
            messages, prefill_type=prefill_type, intent=True
        )

        logprobs_result = self.logprobs_extractor.get_logprobs(prefilled_msgs)
        raw_logprobs = logprobs_result["all_logprobs"]

        mlp_model, meta = self._get_model(prefill_type)
        threshold = meta.get("best_threshold", 0.5)
        input_dim = meta.get("input_dim", 20)

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
        for messages in messages_list:
            try:
                result = self.detect(messages, prefill_type)
                results.append(result)
            except Exception as e:
                results.append({"label": "error", "probability": 0.0, "error": str(e)})
        return results

    def get_model_info(self) -> dict:
        info = {
            "model_name": self.model_name,
            "reasoning_parser": self.logprobs_extractor.reasoning_parser,
            "mode": self.logprobs_extractor.mode,
            "loaded_models": list(self.mlp_models.keys()),
        }
        for ptype, m in self.meta.items():
            info[ptype] = {
                "input_dim": m.get("input_dim"),
                "threshold": m.get("best_threshold", 0.5),
            }
        return info
