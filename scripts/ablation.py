"""消融实验：特征配置对比 + 决策层规则的作用。

用法（需先 train 出主模型）：python scripts/ablation.py
输出：docs/ablation.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage.agent import TriageAgent  # noqa: E402
from triage.config import DOCS_DIR, MODEL_PATH  # noqa: E402
from triage.data import load_newsgroups  # noqa: E402
from triage.features import clean_text  # noqa: E402

CONFIGS = {
    "完整配置（1–2 gram + 停用词 + sublinear）": dict(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english"),
    "仅一元词（1 gram）": dict(ngram_range=(1, 1), min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english"),
    "不去停用词": dict(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, stop_words=None),
    "min_df=1（更大词表）": dict(ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True, stop_words="english"),
    "关闭 sublinear_tf": dict(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=False, stop_words="english"),
}


def main() -> None:
    train_texts, y_train, test_texts, y_test, _ = load_newsgroups()
    train_clean = [clean_text(t) for t in train_texts]
    test_clean = [clean_text(t) for t in test_texts]

    rows = {}
    for label, kw in CONFIGS.items():
        t0 = time.time()
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(token_pattern=r"(?u)\b\w{2,}\b", **kw)),
            ("clf", LinearSVC(C=1.0)),
        ])
        pipe.fit(train_clean, y_train)
        pred = pipe.predict(test_clean)
        rows[label] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            "vocab_size": int(len(pipe.named_steps["tfidf"].vocabulary_)),
            "train_seconds": round(time.time() - t0, 1),
        }
        print(f"{label:35s} acc={rows[label]['accuracy']:.4f} "
              f"macroF1={rows[label]['macro_f1']:.4f} vocab={rows[label]['vocab_size']}", flush=True)

    # 决策层：未登录词规则是否开启，对复核队列的影响
    bundle = joblib.load(MODEL_PATH)
    base = TriageAgent.load()
    variants = {
        "仅阈值 0.55（关闭未登录词规则）": TriageAgent(
            base.pipeline, base.class_names, base.keyword_table,
            threshold=0.55, oov_ratio_max=2.0, explain_coef=base.explain_coef,
        ),
        "阈值 0.55 + 未登录词 45%（过紧）": TriageAgent(
            base.pipeline, base.class_names, base.keyword_table,
            threshold=0.55, oov_ratio_max=0.45, explain_coef=base.explain_coef,
        ),
        "阈值 0.55 + 未登录词 70%（校准后）": base,
    }
    y = np.asarray(y_test)
    decision_rows = {}
    for label, agent in variants.items():
        decisions = [agent.triage(t) for t in test_texts]
        actions = np.array([d.action for d in decisions])
        pred = np.array([agent.class_names.index(d.label) if d.label else -1 for d in decisions])
        auto = actions == "auto_route"
        wrong = pred != y
        auto_n = int(auto.sum())
        review_n = len(y) - auto_n
        decision_rows[label] = {
            "auto_ratio": round(auto_n / len(y), 4),
            "auto_accuracy": round(float((pred[auto] == y[auto]).mean()) if auto_n else 0.0, 4),
            "review_ratio": round(review_n / len(y), 4),
            "review_error_share": round(float(wrong[~auto].sum() / max(int(wrong.sum()), 1)), 4),
            "upper_accuracy": round((int((pred[auto] == y[auto]).sum()) + review_n) / len(y), 4),
        }
        print(label, decision_rows[label], flush=True)

    payload = {"features": rows, "decision": decision_rows}
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from triage.plots import ablation as plot_ablation  # noqa: E402

    plot_ablation(rows, DOCS_DIR / "figures" / "fig6_ablation.png")
    print("已写入", DOCS_DIR / "ablation.json")


if __name__ == "__main__":
    main()
