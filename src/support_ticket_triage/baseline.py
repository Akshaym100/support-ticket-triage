"""Level 1: classical ML baseline — TF-IDF features + linear classifiers.

Run:  uv run python -m support_ticket_triage.baseline
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from support_ticket_triage.data import load_splits

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def build_features():
    # Word n-grams capture phrases ("cancel order"); character n-grams make the
    # model robust to typos, which this dataset deliberately contains ("oorder").
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])


MODELS = {
    "tfidf_logreg": lambda: LogisticRegression(max_iter=2000, C=10),
    "tfidf_linearsvc": lambda: LinearSVC(C=1.0),
}


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    train, val, test = load_splits()

    results = {}
    best_name, best_f1, best_pipe = None, -1.0, None
    for name, make_clf in MODELS.items():
        pipe = make_pipeline(build_features(), make_clf())
        start = time.perf_counter()
        pipe.fit(train["text"], train["intent"])
        train_sec = time.perf_counter() - start

        val_f1 = f1_score(val["intent"], pipe.predict(val["text"]), average="macro")
        results[name] = {"val_macro_f1": round(val_f1, 4), "train_seconds": round(train_sec, 1)}
        print(f"{name:18s} val macro-F1 = {val_f1:.4f}   ({train_sec:.1f}s to train)")
        if val_f1 > best_f1:
            best_name, best_f1, best_pipe = name, val_f1, pipe

    # The test set is touched once, only for the model chosen on validation.
    start = time.perf_counter()
    test_pred = best_pipe.predict(test["text"])
    ms_per_msg = (time.perf_counter() - start) / len(test) * 1000
    results["best"] = {
        "model": best_name,
        "test_accuracy": round(accuracy_score(test["intent"], test_pred), 4),
        "test_macro_f1": round(f1_score(test["intent"], test_pred, average="macro"), 4),
        "ms_per_message_cpu": round(ms_per_msg, 3),
    }
    print(f"\nBest on val: {best_name} -> test {results['best']}")
    print(classification_report(test["intent"], test_pred, digits=3))

    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay.from_predictions(
        test["intent"], test_pred, ax=ax, xticks_rotation=90, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"{best_name} — test confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline_confusion_matrix.png", dpi=120)

    (RESULTS_DIR / "baseline.json").write_text(json.dumps(results, indent=2))
    joblib.dump(best_pipe, MODELS_DIR / "baseline.joblib")
    print(f"Saved results to {RESULTS_DIR} and model to {MODELS_DIR / 'baseline.joblib'}")


if __name__ == "__main__":
    main()
