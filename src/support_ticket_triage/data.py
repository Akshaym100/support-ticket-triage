"""Download the Banking77 customer-support dataset and make train/val/test splits.

Banking77: ~13k real customer messages to a bank, labelled with 77 fine-grained intents
(card_arrival, lost_or_stolen_card, refund_not_showing_up, ...).

Run:  uv run python -m support_ticket_triage.data
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Parquet copy of PolyAI/banking77 (the original repo uses a loading script that
# newer `datasets` versions no longer run).
DATASET_ID = "mteb/banking77"
SEED = 42
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _normalize(texts: pd.Series) -> pd.Series:
    return texts.str.lower().str.split().str.join(" ")


def load_raw():
    ds = load_dataset(DATASET_ID)
    train = ds["train"].to_pandas().rename(columns={"label_text": "intent"})[["text", "intent"]]
    test = ds["test"].to_pandas().rename(columns={"label_text": "intent"})[["text", "intent"]]
    return train, test


def split(train: pd.DataFrame, test: pd.DataFrame):
    # Drop repeated messages, and any training message that also appears in the test
    # set — otherwise the model is graded on sentences it memorised.
    train = train.loc[~_normalize(train["text"]).duplicated()]
    train = train.loc[~_normalize(train["text"]).isin(set(_normalize(test["text"])))]

    # Keep the official test set untouched; carve a stratified 10% validation set
    # from train for choosing models and hyperparameters.
    train, val = train_test_split(train, test_size=0.1, stratify=train["intent"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_splits():
    """Read the saved CSVs, creating them first if they don't exist."""
    paths = {name: DATA_DIR / f"{name}.csv" for name in ("train", "val", "test")}
    if not all(p.exists() for p in paths.values()):
        main()
    return tuple(pd.read_csv(p) for p in paths.values())


def main():
    DATA_DIR.mkdir(exist_ok=True)
    train, val, test = split(*load_raw())
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(DATA_DIR / f"{name}.csv", index=False)
    print(f"{train['intent'].nunique()} intents")
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  -> saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
