# leakgauge

Core library for LeakGauge — detecting LLM leakage intent from prefill log probabilities.

## File Structure

| File | Description |
|------|-------------|
| `config.py` | Prefill suffix and reasoning prefix configurations |
| `logprobs.py` | `LogProbsPrompt` — unified logprobs extraction (server + offline) |
| `mlp.py` | `BinaryMlp` model and `MlpTrainer` for training/evaluation |
| `dataset.py` | `SimpleDataset` and `DatasetManager` for data loading and splitting |
| `detector.py` | `LeakageDetector` — end-to-end detection (logprobs + MLP inference) |


## config.py

Stores prefill configurations used by `LogProbsPrompt.apply_prefill()`.

- `PREFILL_SUFFIX`: intent-based suffix for each prefill type (`sys_prompt`, `rag_chunks`, `universe`)
- `REASONING_PREFIX`: think-tag prefix for reasoning models (`deepseek_r1`, `qwen3`, `none`)


## LogProbsPrompt (logprobs.py)

Unified interface for extracting prefill logprobs from vLLM.

### Init

| Parameter | Required | Description |
|-----------|----------|-------------|
| `base_url` | Server mode | vLLM server URL. Model name, path, and tokenizer are auto-detected |
| `api_key` | Server mode | API key, default `"none"` |
| `temperature` | Server mode | Generation temperature, default `0` |
| `top_p` | Server mode | Top-p sampling, default `0.95` |
| `llm` | offline mode | `vllm.LLM` instance. Tokenizer is auto-loaded from the LLM |
| `reasoning_parser` | Both | `"deepseek_r1"`, `"qwen3"`, or `"none"` (default) |

### Key Methods

- `apply_prefill(msgs, prefill_type, intent, prefill_length)` — append prefill assistant message to raw conversations
- `get_logprobs(msgs, logprobs_num)` — extract logprobs for the prefill part of already-prefilled messages

### Auto-detected Attributes

| Attribute | Server mode | Offline mode |
|-----------|------------|---------|
| `self.model` | `server_model.id` (for API calls) | `None` |
| `self.model_root` | `server_model.root` (actual path) | `llm.model_config.model` |
| `self.tokenizer` | auto-loaded from `model_root` | auto-loaded from `model_root` |


## BinaryMlp (mlp.py)

Simple binary classifier (Linear→ReLU→Linear) for attack/benign classification on logprobs features.

| Method | Description |
|--------|-------------|
| `save(path, meta)` | Save weights to `.pt`, optional meta to `.meta.json` |
| `BinaryMlp.load(path, map_location)` | Load model from `.pt` checkpoint to specified device |

### MlpTrainer

Trains `BinaryMlp` with BCEWithLogitsLoss + Adam + CosineAnnealingLR.

| Parameter | Description |
|-----------|-------------|
| `model` | `BinaryMlp` instance |
| `train_loader` | Training DataLoader |
| `val_loader` | Validation DataLoader |
| `lr` | Learning rate |
| `device` | `cuda` or `cpu` |


## DatasetManager (dataset.py)

Handles logprobs feature padding, train/test splitting, and DataLoader creation.

| Parameter | Description |
|-----------|-------------|
| `X, y` | Feature arrays and labels |
| `batch_size` | Batch size, default `64` |
| `test_ratio` | Test split ratio, default `0.2` |
| `target_data_len` | Pad/truncate features to this length, default `50` |
| `pad_value` | Padding value, default `-10` |


## LeakageDetector (detector.py)

End-to-end wrapper: logprobs extraction + MLP inference. Supports both server and offline mode.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `processor_path` | Yes | Path to trained MLP `.pt` checkpoint. `reasoning_parser` is auto-read from `.meta.json` |
| `base_url` | Server mode | vLLM server URL |
| `api_key` | Server mode | API key |
| `llm` | offline mode | `vllm.LLM` instance |
| `device` | Optional | Inference device, auto-detected if not set |

### Key Methods

- `detect(messages, prefill_type)` — single message detection, returns `{label, probability, threshold, logprobs}`
- `detect_batch(messages_list, prefill_type)` — batch detection
- `get_model_info()` — return model and checkpoint metadata
