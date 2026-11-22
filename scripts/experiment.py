"""候选模型横向对比实验。

用法（项目根目录）：
    python scripts/experiment.py
输出：docs/experiment_models.json，同时打印对比表。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage.config import DOCS_DIR, SEED  # noqa: E402
from triage.data import load_newsgroups  # noqa: E402
from triage.features import clean_text  # noqa: E402
from triage.models import MODEL_SPECS, build_pipeline  # noqa: E402

MODEL_ORDER = ["mnb", "lr", "svc", "calibrated_svc"]


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(proba, axis=1)[:, -k:]
    hit = [int(y) in topk[i] for i, y in enumerate(y_true)]
    return float(np.mean(hit))


def main() -> None:
    train_texts, y_train, test_texts, y_test, names = load_newsgroups()
    print(f"训练集 {len(train_texts)} 条，测试集 {len(test_texts)} 条，类别 {len(names)} 个", flush=True)

    t0 = time.time()
    train_clean = [clean_text(t) for t in train_texts]
    test_clean = [clean_text(t) for t in test_texts]
    print(f"清洗完成 {time.time() - t0:.1f}s", flush=True)

    results = {}

    # 基线：永远预测多数类
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(train_clean), 1)), y_train)
    pred = dummy.predict(np.zeros((len(test_clean), 1)))
    results["majority"] = {
        "label": "基线：多数类",
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "top3": None,
        "train_seconds": 0.0,
        "infer_ms_per_doc": 0.0,
    }
    print(f"多数类基线 accuracy={results['majority']['accuracy']}", flush=True)

    for name in MODEL_ORDER:
        t0 = time.time()
        pipe = build_pipeline(name)
        pipe.fit(train_clean, y_train)
        train_sec = time.time() - t0

        t1 = time.time()
        pred = pipe.predict(test_clean)
        infer_ms = (time.time() - t1) / len(test_clean) * 1000

        clf = pipe.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            proba = pipe.predict_proba(test_clean)
            top3 = top_k_accuracy(proba, y_test)
        else:
            decision = pipe.decision_function(test_clean)
            top3 = top_k_accuracy(decision, y_test)

        results[name] = {
            "label": MODEL_SPECS[name][0],
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_test, pred, average="weighted")), 4),
            "top3": round(float(top3), 4),
            "train_seconds": round(train_sec, 2),
            "infer_ms_per_doc": round(infer_ms, 3),
            "has_proba": bool(hasattr(clf, "predict_proba")),
        }
        print(
            f"{name:15s} acc={results[name]['accuracy']:.4f} "
            f"macroF1={results[name]['macro_f1']:.4f} top3={results[name]['top3']:.4f} "
            f"train={train_sec:.1f}s infer={infer_ms:.2f}ms/doc",
            flush=True,
        )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "experiment_models.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入", out)


if __name__ == "__main__":
    main()
