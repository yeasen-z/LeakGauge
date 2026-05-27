from typing import Dict, Optional
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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

class LogProbsPrompt_server:
    def __init__(self,
                 vllm_cfg: Optional[Dict] = None,
                 tokenizer: Optional[AutoTokenizer] = None
                ):
        self.vllm_cfg = vllm_cfg or {}
        self.api_key = self.vllm_cfg.get("api_key")
        self.base_url = self.vllm_cfg.get("base_url")
        self.tokenizer = tokenizer

    def get_logprobs(self, msgs, model=None, logprobs_num = 2) -> Dict:
        """
            Get log probabilities for the given query using the LLM. 
            msgs is a list of messages which include the prefill part of assistant output. give an example of msgs:
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris is"}
            ]
            the last message is the prefill part, and we want to get logprobs for it. 
        """
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        msgs_tokens = self.tokenizer.apply_chat_template(
            msgs[:-1],
            tokenize=True,
            add_generation_prompt=True
        )

        prefill_tokens = self.tokenizer.encode(msgs[-1]["content"], add_special_tokens=False)

        all_input_tokens = safe_token_concat(msgs_tokens, prefill_tokens)

        if model == None:
            model = self.vllm_cfg["model"]

        generate_token = 1

        response = client.completions.create(
            model=model,
            prompt=all_input_tokens,
            max_tokens=generate_token,
            temperature=self.vllm_cfg.get("temperature", 0.7),
            top_p=self.vllm_cfg.get("top_p", 0.95),
            extra_body={
                "return_token_ids": True,
                "prompt_logprobs": logprobs_num
            }
        )

        choice = response.choices[0]

        prefill_logprobs_dict = choice.prompt_logprobs[-len(prefill_tokens):]

        print(msgs[-1]["content"])
        print([list(d.values())[0]["logprob"] for d in prefill_logprobs_dict])

        results = {
            "text": choice.text,
            "prefill_logprobs_dict": prefill_logprobs_dict,
            "all_logprobs": [list(d.values())[0]["logprob"] for d in prefill_logprobs_dict],
            "all_rank": [list(d.values())[0]["rank"] for d in prefill_logprobs_dict]
        }

        return results

            
class LogProbsPrompt_sdk:
    def __init__(self,
                 llm: Optional[LLM] = None,
                 tokenizer: Optional[AutoTokenizer] = None
                ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.tokenizer = tokenizer

    def get_logprobs(self, msgs, logprobs_num = 2) -> Dict:
        """``
            Get log probabilities for the given query using the LLM. 
            msgs is a list of messages which include the prefill part of assistant output. give an example of msgs:
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris is"}
            ]
            the last message is the prefill part, and we want to get logprobs for it. 
        """
        msgs_tokens = self.tokenizer.apply_chat_template(
            msgs[:-1],
            tokenize=True,
            add_generation_prompt=True
        )
        msgs_tokens = msgs_tokens.input_ids
            
        prefill_tokens = self.tokenizer.encode(
            msgs[-1]["content"],
            add_special_tokens=False
        )
        all_input_tokens = msgs_tokens + prefill_tokens
        
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
        full_prompt_logprobs = output.prompt_logprobs

        output_prefill_logprobs = full_prompt_logprobs[-len(prefill_tokens):]

        all_logprobs = []
        all_rank = []
        all_prefill_logprobs_dict = []

        for token_dict in output_prefill_logprobs:

            if token_dict is None:
                all_logprobs.append(None)
                all_rank.append(None)
                continue

            # current token id
            token_id = list(token_dict.keys())[0]

            token_info = token_dict[token_id]
            all_logprobs.append(token_info.logprob)
            all_rank.append(token_info.rank)
            all_prefill_logprobs_dict.append({token_id: {"logprob": token_info.logprob, 
                                                         "rank": token_info.rank,
                                                         "decoded_token": token_info.decoded_token
                                                         }
                                              })
            
        
        results = {
            "text": output.outputs[0].text,
            "prefill_logprobs_dict": all_prefill_logprobs_dict,
            "all_logprobs": all_logprobs,
            "all_rank": all_rank
        }
        return results
