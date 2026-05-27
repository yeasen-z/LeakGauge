# Eliciting LLM Leakage Awareness from Prefill Log Probabilities

here we support Python sdk and vLLM local server version.


# Build demo test datasets
for seperate system prompt or RAG chunks, run
```bash
python -m scripts.data_prepare.data_prepare_rag
python -m scripts.data_prepare.data_prepare_sys
```
if you want test mixed leakage data, run
```bash
python -m scripts.data_prepare.data_prepare_leak
```

# Test on vLLM Python SDK (i.e. system prompts)
here set `Llama-3.1-8B-Instruct` as a example model

for intent, get logprob data
```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.get_logprobs_sdk \
  --model path/to/Llama-3.1-8B-Instruct/ \
  --tensor_parallel_size 1 \
  --prefill_type sys_prompt \
  --intent \
  --msg_dir data_input/sys_mixed
```

train and evalute mlp
```bash
python -m scripts.train_and_evaluate \
  --target_path logprobs/intent/Llama-3.1-8B-Instruct/sys_prompt \
  --epochs 20 --train_lr 0.005 --training_batch 64 \
  --device cuda:0
```

# Test on vLLM local server (i.e. system prompts)
incase the model and server is `Llama-3.1-8B-Instruct` at port `22991`

for intent, get logprob data
```bash
python -m scripts.get_logprobs_server \
  --model model_name_of_server \
  --tokenizer path/to/the/tokenizer/ \
  --reasoning_parser none \
  --base_url http://127.0.0.1:22991/v1 \
  --intent \
  --prefill_type sys_prompt \ 
  --msg_dir data_input/sys_mixed
```

train and evalute mlp
```bash
python -m scripts.train_and_evaluate \
  --target_path logprobs/intent/Llama-3.1-8B-Instruct/sys_prompt \
  --epochs 20 --train_lr 0.005 --training_batch 64 \
  --device cuda:0
```




# Test on mixed leakage and experimental universe suffix （server version）
incase the model and server is `Llama-3.1-8B` at port `22991`

```bash
python -m scripts.get_logprobs_server \
  --model /share/workspace/models/meta/Llama-3.1-8B-Instruct \
  --tokenizer /share/workspace/models/meta/Llama-3.1-8B-Instruct \
  --reasoning_parser none \
  --base_url http://127.0.0.1:22991/v1 \
  --intent \
  --msg_dir data_input/leak_mixed
```

train and evalute mlp

```bash
python -m scripts.train_and_evaluate \
  --target_path logprobs/intent/Llama-3.1-8B-Instruct/universe_leak_mixed \
  --epochs 20 --train_lr 0.005 --training_batch 64 \
  --device cuda:0
```


---

server版差部署代码
