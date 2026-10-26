# Support Ticket Triage — classical ML vs DistilBERT vs LoRA

Classify a customer's message into one of **77 intents** (`card_arrival`, `lost_or_stolen_card`,
`refund_not_showing_up`, …) and compare three approaches on the same data:

| Level | Approach | Where it runs |
|---|---|---|
| 1 | TF-IDF (word + character n-grams) + Logistic Regression / Linear SVM | Laptop (CPU, seconds) |
| 2a | DistilBERT, full fine-tuning (all 67M weights) | Kaggle GPU |
| 2b | DistilBERT + LoRA (≈1.4% of weights trained) | Kaggle GPU |

**Dataset:** [Banking77](https://huggingface.co/datasets/mteb/banking77): about 13k real banking-support messages.
The official test set (3,076) is kept untouched. 10% of train is the validation set used to pick models.
Duplicates, and train messages that also appear in test, are removed first.

> Why not the Bitext customer-support dataset? I tried it first: TF-IDF reached **99.9%**, because its
> messages are generated from templates. With nothing left to improve, it can't show whether
> deep learning is worth it, so I switched to a harder, real-query benchmark.

## Results (test set)

| Model | Macro-F1 | Trainable params | Train time | Latency / msg |
|---|---|---|---|---|
| TF-IDF + LinearSVC | 0.905 | – | 3 s (CPU) | 0.08 ms (CPU, batched) |
| DistilBERT full FT (8 epochs) | 0.919 | 67.0M (100%) | 4.8 min (2× T4) | 0.98 ms (GPU, batched) |
| **DistilBERT + LoRA r=16** (15 epochs) | **0.929** | **0.94M (1.4%)** | 7.5 min (2× T4) | 1.00 ms (GPU, batched) |

**What this shows**
- Classical ML is a strong, cheap baseline: 90.5% in 3 seconds on a laptop. Deep learning adds +2.4 points.
- LoRA beat full fine-tuning while training **1.4%** of the weights. The saved adapter is **3.7 MB**,
  compared with about 268 MB for a full model copy, so one base model can serve many tasks with
  small swappable adapters. Freezing the pretrained weights also acts as regularisation on a
  small dataset (about 9k examples), which likely explains why it scored higher.
- LoRA needed more epochs (15 vs 8) and a 20× higher learning rate (1e-3 vs 5e-5) to get there.
- **First attempt underfit:** with 3 epochs at batch size 64, full fine-tuning scored 0.833, *below* TF-IDF,
  with validation loss still falling. Smaller batches, more epochs and a learning-rate warm-up fixed it.

## Try it

The trained LoRA adapter is included in `models/lora_adapter/`. After cloning, build the TF-IDF model once
with `uv run python -m support_ticket_triage.baseline`. It takes a few seconds.

```bash
uv run python -m support_ticket_triage.predict "I still haven't received my new card"
#   TF-IDF : card_arrival
#   LoRA   : card_arrival   (confidence 100%)
```
Run it without a message to type messages one by one. The LoRA model runs on the CPU in about 15–60 ms per message.

## How to run

```bash
uv sync                                         # create .venv with every package
uv run python -m support_ticket_triage.data     # download + split -> data/
uv run python -m support_ticket_triage.baseline # level 1 -> results/, models/
```

**Level 2 on Kaggle**
1. kaggle.com → **Create → New Notebook → File → Import Notebook** → upload `notebooks/finetune_distilbert_lora.ipynb`.
2. Right panel: **Accelerator → GPU T4 x2**, **Internet → On**.
3. **Run All** (about 10–20 minutes).
4. From the **Output** panel, download `results.json` → `results/finetune.json`, and `lora_adapter.zip` →
   unzip it into `models/lora_adapter/`.

Edit `notebooks/finetune_distilbert_lora.py`, then rebuild the notebook with
`uv run python scripts/py_to_ipynb.py notebooks/finetune_distilbert_lora.py`.
To test it on CPU in a minute: `SMOKE=1 uv run python notebooks/finetune_distilbert_lora.py`.

## Project layout

```
src/support_ticket_triage/
  data.py        download, de-duplicate, split
  baseline.py    TF-IDF + linear models, metrics, confusion matrix
  predict.py     classify a message with TF-IDF and LoRA side by side
notebooks/
  finetune_distilbert_lora.py / .ipynb   full fine-tuning vs LoRA (Kaggle)
scripts/py_to_ipynb.py                   .py -> .ipynb converter
data/  models/  results/                 generated, not committed
```
