# GPT-2 ChatBot from Scratch

A character-level GPT-2 language model built from scratch using PyTorch. Trains on the OpenWebText dataset and runs an interactive text-completion chatbot using the trained model.

This is a learning/research project — the full transformer architecture (self-attention, multi-head attention, feed-forward blocks, layer norm) is implemented manually without any high-level abstractions.

---

## How It Works

```
extract_dataset.py  →  generate_training_model.py  →  chatBot.py
   (prepare data)          (train the model)           (chat with it)
```

1. **Extract** — decompress OpenWebText `.xz` files into train/val text splits and build a character vocabulary
2. **Train** — train a GPT-2 model on those splits, save weights to `model-01.pkl`
3. **Chat** — load the saved model and interactively generate text completions from your prompts

---

## Requirements

**Python 3.8+** and the following packages:

```shell
pip install pylzma numpy tqdm ipykernel jupyter torch --index-url https://download.pytorch.org/whl/cu118
```

> **No NVIDIA GPU?** That's fine. The code automatically falls back to CPU (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`). Training will be significantly slower on CPU.

---

## Setup & Usage

### Step 1 — Prepare the dataset

Download the [OpenWebText dataset](https://skylion007.github.io/OpenWebTextCorpus/) and place it at `D:/openwebtext/openwebtext` (or edit the `folder_path` variable in `extract_dataset.py`).

```shell
python extract_dataset.py
```

**Outputs:**
- `output_train.txt` — 90% of the data, used for training
- `output_val.txt` — 10% of the data, used for validation
- `vocab.txt` — all unique characters found in the dataset

### Step 2 — Train the model

```shell
python generate_training_model.py
```

Trains for 200 iterations and saves the model weights to `model-01.pkl`.

**Key hyperparameters (edit in the script):**

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 32 | Training samples per iteration |
| `block_size` | 128 | Context window (characters) |
| `max_iters` | 200 | Total training iterations |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `n_embd` | 384 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `n_layer` | 4 | Number of transformer blocks |
| `dropout` | 0.2 | Dropout rate |

### Step 3 — Run the chatbot

```shell
python chatBot.py -batch_size 32
```

You'll get an interactive prompt. Type anything and the model will generate a 150-character continuation.

```
Prompt:
Once upon a time

Completion:
Once upon a time there was a great king who lived in a land far away...
```

Press `Ctrl+C` to exit.

---

## Model Architecture

```
GPTLanguageModel
├── Token Embedding        (vocab_size → 384)
├── Position Embedding     (128 → 384)
├── Transformer Blocks × 4
│   ├── MultiHeadAttention (4 heads × 96 dims)
│   │   └── Head × 4
│   │       ├── Key, Query, Value projections
│   │       └── Causal (masked) self-attention
│   ├── FeedForward        (384 → 1536 → 384, ReLU)
│   └── LayerNorm × 2      (pre-norm residual connections)
├── LayerNorm (final)
└── Linear output head     (384 → vocab_size)
```

---

## Project Files

| File | Purpose |
|---|---|
| `extract_dataset.py` | Decompresses OpenWebText `.xz` files, creates train/val splits and `vocab.txt` |
| `generate_training_model.py` | Defines and trains the GPT-2 model, saves `model-01.pkl` |
| `chatBot.py` | Loads `model-01.pkl` and runs the interactive text-generation loop |

---

## Known Issues

- `chatBot.py` has a syntax error on line 32 (missing closing parenthesis) and is missing `import torch.nn.functional as F` — needs a fix before it can run
- `generate_training_model.py` is incomplete — the training loop and model save are cut off at line 180
- The `FeedFoward` class in `generate_training_model.py` is a typo (missing an `r`)
