"""Classify customer messages with the TF-IDF baseline and the LoRA model side by side.

Run:  uv run python -m support_ticket_triage.predict "I still haven't received my card"
      uv run python -m support_ticket_triage.predict          (type messages interactively)

Needs models/baseline.joblib (from baseline.py) and models/lora_adapter/ (from Kaggle).
"""

import sys
import time
from pathlib import Path

import joblib
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from support_ticket_triage.data import load_splits

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT / "models" / "lora_adapter"
BASE_MODEL = "distilbert-base-uncased"


class LoraClassifier:
    def __init__(self):
        # The adapter stores only the LoRA weights and the classifier head, not the label
        # names. They were numbered in sorted order during training, so rebuild that order.
        train, _, _ = load_splits()
        labels = sorted(train["intent"].unique())
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=len(labels),
            id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)},
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
        self.labels = labels

    @torch.no_grad()
    def predict(self, text: str):
        inputs = self.tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
        probs = self.model(**inputs).logits.softmax(-1)[0]
        conf, idx = probs.max(-1)
        return self.labels[idx], conf.item()


def main():
    baseline = joblib.load(ROOT / "models" / "baseline.joblib")
    lora = LoraClassifier()

    def show(text):
        start = time.perf_counter()
        base_label = baseline.predict([text])[0]
        base_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        lora_label, conf = lora.predict(text)
        lora_ms = (time.perf_counter() - start) * 1000
        print(f"  TF-IDF : {base_label:35s} ({base_ms:.1f} ms)")
        print(f"  LoRA   : {lora_label:35s} ({lora_ms:.1f} ms, confidence {conf:.0%})")

    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return
    print("Type a customer message (empty line to quit).")
    while text := input("> ").strip():
        show(text)


if __name__ == "__main__":
    main()
