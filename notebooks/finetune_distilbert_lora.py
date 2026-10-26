# %% [markdown]
# # Level 2: DistilBERT — full fine-tuning vs LoRA
#
# Runs on **Kaggle** (Settings → Accelerator → GPU T4 x2, Internet → On).
#
# Banking77 (77 intents), same cleaning and split as the local TF-IDF baseline, so the numbers are comparable.
# At the end, download `/kaggle/working/results.json` and `lora_adapter.zip`
# and put them in the project's `results/` and `models/` folders.

# %%
# Kaggle images ship older versions; get recent ones. (Locally this is already installed.)
import os, subprocess, sys
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "transformers", "peft", "accelerate", "datasets"], check=True)
    # Kaggle's preinstalled torchao is too old for the new peft, which then refuses to
    # build LoRA layers. We don't use torchao, so remove it.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=True)

# %%
import json, shutil, time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

SMOKE = os.environ.get("SMOKE") == "1"   # tiny CPU run, only to check the code works
SEED = 42
MODEL_ID = "distilbert-base-uncased"
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT.mkdir(exist_ok=True)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")

# %% [markdown]
# ## Data — identical cleaning and split to `src/support_ticket_triage/data.py`

# %%
raw = load_dataset("mteb/banking77")
train_df = raw["train"].to_pandas().rename(columns={"label_text": "intent"})[["text", "intent"]]
test_df = raw["test"].to_pandas().rename(columns={"label_text": "intent"})[["text", "intent"]]

normalize = lambda s: s.str.lower().str.split().str.join(" ")
train_df = train_df.loc[~normalize(train_df["text"]).duplicated()]
train_df = train_df.loc[~normalize(train_df["text"]).isin(set(normalize(test_df["text"])))]
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["intent"], random_state=SEED)

labels = sorted(train_df["intent"].unique())
if SMOKE:
    train_df, val_df, test_df = train_df[:256], val_df[:128], test_df[:128]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} labels={len(labels)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def to_dataset(frame):
    ds = Dataset.from_dict({"text": frame["text"].tolist(),
                            "label": [label2id[l] for l in frame["intent"]]})
    # Messages are short; 64 tokens covers almost all of them and keeps training fast.
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=64), batched=True)

train_ds, val_ds, test_ds = map(to_dataset, (train_df, val_df, test_df))
collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(pred):
    y_pred = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(pred.label_ids, y_pred),
            "macro_f1": f1_score(pred.label_ids, y_pred, average="macro")}

# %% [markdown]
# ## One training function for both runs

# %%
def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def run(name, model, lr, epochs):
    trainable, total = count_params(model)
    print(f"\n=== {name}: training {trainable:,} of {total:,} parameters ({trainable / total:.2%}) ===")
    args = TrainingArguments(
        output_dir=str(OUT / f"ckpt_{name}"),
        learning_rate=lr,
        num_train_epochs=1 if SMOKE else epochs,
        max_steps=4 if SMOKE else -1,
        # 16 per GPU x 2 GPUs = 32 per step. Smaller batches give more update steps,
        # which this small dataset (~9k messages) needs.
        per_device_train_batch_size=16,
        per_device_eval_batch_size=128,
        warmup_steps=0.1,          # first 10% of steps: ramp the learning rate up gently; protects pretrained weights
        logging_strategy="epoch",  # show training loss each epoch (was "No log")
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator, compute_metrics=compute_metrics,
                      processing_class=tokenizer)
    start = time.perf_counter()
    trainer.train()
    train_sec = time.perf_counter() - start

    start = time.perf_counter()
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    ms_per_msg = (time.perf_counter() - start) / len(test_ds) * 1000
    result = {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "train_minutes": round(train_sec / 60, 2),
        "test_accuracy": round(test_metrics["test_accuracy"], 4),
        "test_macro_f1": round(test_metrics["test_macro_f1"], 4),
        "ms_per_message_gpu": round(ms_per_msg, 3),
    }
    print(name, result)
    return trainer, result

def fresh_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(labels), id2label=id2label, label2id=label2id)

results = {}

# %% [markdown]
# ## Run A — full fine-tuning (every weight is updated)

# %%
# First attempt (3 epochs, batch 64) underfit: val loss was still 1.88 and macro-F1 0.83,
# below the TF-IDF baseline. 77 classes from ~9k examples needs more passes.
_, results["distilbert_full"] = run("full", fresh_model(), lr=5e-5, epochs=8)

# %% [markdown]
# ## Run B — LoRA (base model frozen, small low-rank adapters trained)
#
# LoRA adds `W' = W + B·A` to the attention query and value projections, where A and B
# are rank-16 matrices. The classifier head is new, so it is trained fully (`modules_to_save`).

# %%
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],                 # DistilBERT's attention Q and V layers
    modules_to_save=["pre_classifier", "classifier"],  # the randomly initialised head
)
lora_model = get_peft_model(fresh_model(), lora_cfg)
lora_model.print_trainable_parameters()
# LoRA usually needs a much higher learning rate and more epochs than full fine-tuning,
# because it updates only ~1.4% of the weights.
lora_trainer, results["distilbert_lora_r16"] = run("lora", lora_model, lr=1e-3, epochs=15)

# %% [markdown]
# ## Save results and the LoRA adapter (only a few MB, unlike the full model)

# %%
adapter_dir = OUT / "lora_adapter"
lora_trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
shutil.make_archive(str(OUT / "lora_adapter"), "zip", adapter_dir)

(OUT / "results.json").write_text(json.dumps(results, indent=2))
for name, r in results.items():
    print(f"{name:22s} macro-F1={r['test_macro_f1']:.4f}  trainable={r['trainable_pct']}%  "
          f"time={r['train_minutes']}min")
print("Download:", OUT / "results.json", "and", OUT / "lora_adapter.zip")
