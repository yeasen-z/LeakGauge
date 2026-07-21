import copy
from typing import Dict, Optional
from transformers import AutoTokenizer
from leakgauge.config import PREFILL_SUFFIX, REASONING_PREFIX


def safe_token_concat(tokens1, tokens2):
    if hasattr(tokens1, "get"):
        tokens1 = tokens1["input_ids"]
        if isinstance(tokens1[0], list):
            tokens1 = tokens1[0]
    if hasattr(tokens2, "get"):
        tokens2 = tokens2["input_ids"]
        if isinstance(tokens2[0], list):
            tokens2 = tokens2[0]

    return list(tokens1) + list(tokens2)


class LogProbsPrompt:
    """Unified logprobs extractor supporting both vLLM server API and offline modes.

    Server mode: pass base_url (+ optional api_key, model).
    Offline mode: pass llm (vllm.LLM instance).
    """

    def __init__(
        self,
        reasoning_parser: str = "none",
        # server mode args
        base_url: Optional[str] = None,
        api_key: str = "none",
        temperature: float = 0,
        top_p: float = 0.95,
        # offline mode args
        llm=None,
    ):
        """
        Server mode: only base_url is required. model name, tokenizer, and
            logprobs permission are auto-detected from the server.
        Offline mode: only llm is required. tokenizer is auto-loaded from the LLM.
        """
        self.model = None       # server API model id (may be alias)
        self.model_root = None  # actual model path on disk
        self.reasoning_parser = reasoning_parser

        if base_url is not None:
            self.mode = "server"
            self.base_url = base_url
            self.api_key = api_key
            self.temperature = temperature
            self.top_p = top_p
            self._auto_detect_server_info()
            print(f"[LogProbsPrompt] Server mode: {self.base_url} | model: {self.model}")
        elif llm is not None:
            self.mode = "offline"
            self.llm = llm
            self.model_root = self.llm.llm_engine.model_config.model
            print(f"[LogProbsPrompt] Offline mode | model: {self.model_root}")
        else:
            raise ValueError("Must provide either base_url (server mode) or llm (Offline mode)")

        # auto-load tokenizer
        if self.model_root:
            print(f"[LogProbsPrompt] Auto-loading tokenizer from: {self.model_root}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_root, trust_remote_code=True)
        else:
            raise ValueError("Cannot auto-load tokenizer: no model_root available.")

    def _auto_detect_server_info(self):
        """Query vLLM server to get model name, path, and check logprobs permission."""
        from openai import OpenAI
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            models = client.models.list()
            if not models.data:
                return

            server_model = models.data[0]
            self.model = server_model.id
            self.model_root = getattr(server_model, "root", None)

            # check logprobs permission
            permissions = getattr(server_model, "permission", [])
            if permissions:
                perm = permissions[0] if isinstance(permissions[0], dict) else permissions[0].__dict__
                allow_logprobs = perm.get("allow_logprobs", None)
                if allow_logprobs is False:
                    raise PermissionError(f"Server model '{self.model}' does not allow logprobs")

            print(f"[LogProbePrompt] Auto-detected model: {self.model} | root: {self.model_root}")
        except PermissionError:
            raise
        except Exception as e:
            print(f"[LogProbePrompt] Could not query server for model info: {e}")

    def _tokenize(self, msgs):
        msgs_tokens = self.tokenizer.apply_chat_template(
            msgs[:-1],
            tokenize=True,
            add_generation_prompt=True
        )
        prefill_tokens = self.tokenizer.encode(msgs[-1]["content"], add_special_tokens=False)
        all_input_tokens = safe_token_concat(msgs_tokens, prefill_tokens)
        return all_input_tokens, prefill_tokens

    def apply_prefill(self, msgs, prefill_type="sys_prompt", intent=True, prefill_length=20) -> list:
        """
        Append a prefill assistant message to the conversation.

        Args:
            msgs: raw messages (system + user).
            prefill_type: "sys_prompt", "rag_chunks", or "universe".
            intent: if True, use intent-based suffix from config;
                    if False, use first prefill_length tokens of the content.
            prefill_length: number of tokens to use when intent=False.
        Returns:
            messages with the prefill assistant message appended.
        """
        work_msg = copy.deepcopy(msgs)

        if intent:
            suffix = PREFILL_SUFFIX.get(prefill_type)
            if suffix is None:
                raise ValueError(f"Unknown prefill_type: {prefill_type}")
            token_get = self.tokenizer.encode(suffix, add_special_tokens=False)
        else:
            if prefill_type == "sys_prompt":
                token_get = self.tokenizer.encode(work_msg[0]["content"], add_special_tokens=False)[:prefill_length]
            elif prefill_type == "rag_chunks":
                token_get = self.tokenizer.encode(work_msg[1]["content"], add_special_tokens=False)[:prefill_length]
            elif prefill_type == "universe":
                raise ValueError("universe prefill_type must be used with intent=True")
            else:
                raise ValueError(f"Unknown prefill_type: {prefill_type}")

        decoded = self.tokenizer.decode(token_get)
        prefix = REASONING_PREFIX.get(self.reasoning_parser, "")

        work_msg.append({"role": "assistant", "content": prefix + decoded})
        return work_msg

    def get_logprobs(self, msgs, logprobs_num=2) -> Dict:
        """
        Get log probabilities for the prefill part of the messages.

        msgs: list of messages, the last one must be the prefill (assistant) message.
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris is"}
            ]

        Tip: use apply_prefill() to build the prefilled messages from raw conversations.
        """
        all_input_tokens, prefill_tokens = self._tokenize(msgs)

        if self.mode == "server":
            return self._get_logprobs_server(all_input_tokens, prefill_tokens, logprobs_num)
        else:
            return self._get_logprobs_offline(all_input_tokens, prefill_tokens, logprobs_num)

    def _get_logprobs_server(self, all_input_tokens, prefill_tokens, logprobs_num):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        response = client.completions.create(
            model=self.model,
            prompt=all_input_tokens,
            max_tokens=1,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body={
                "prompt_logprobs": logprobs_num
            }
        )

        choice = response.choices[0]
        prefill_logprobs_dict = choice.prompt_logprobs[-len(prefill_tokens):]

        return {
            "text": choice.text,
            "prefill_logprobs_dict": prefill_logprobs_dict,
            "all_logprobs": [list(d.values())[0]["logprob"] for d in prefill_logprobs_dict],
            "all_rank": [list(d.values())[0]["rank"] for d in prefill_logprobs_dict]
        }

    def _get_logprobs_offline(self, all_input_tokens, prefill_tokens, logprobs_num):
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            prompt_logprobs=logprobs_num,
            logprobs=logprobs_num
        )
        outputs = self.llm.generate(
            prompts=[all_input_tokens],
            sampling_params=sampling_params
        )
        output = outputs[0]
        output_prefill_logprobs = output.prompt_logprobs[-len(prefill_tokens):]

        all_logprobs = []
        all_rank = []
        all_prefill_logprobs_dict = []

        for token_dict in output_prefill_logprobs:
            if token_dict is None:
                all_logprobs.append(None)
                all_rank.append(None)
                continue

            token_id = list(token_dict.keys())[0]
            token_info = token_dict[token_id]
            all_logprobs.append(token_info.logprob)
            all_rank.append(token_info.rank)
            all_prefill_logprobs_dict.append({
                token_id: {
                    "logprob": token_info.logprob,
                    "rank": token_info.rank,
                    "decoded_token": token_info.decoded_token
                }
            })

        return {
            "text": output.outputs[0].text,
            "prefill_logprobs_dict": all_prefill_logprobs_dict,
            "all_logprobs": all_logprobs,
            "all_rank": all_rank
        }
