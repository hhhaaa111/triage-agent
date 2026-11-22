"""评估工具：指标、阈值扫描、错误案例与混淆对。"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, names: list[str]) -> dict:
    """返回常用指标与逐类别结果。"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(names)), zero_division=0
    )
    per_class = {
        names[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(names))
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "per_class": per_class,
    }


def threshold_sweep(proba: np.ndarray, y_true, thresholds: np.ndarray | None = None) -> list[dict]:
    """扫描置信度阈值，观察自动分流比例与准确率的变化。

    upper_accuracy 的含义是：进入人工复核的部分若全部被人工纠正，
    整体准确率的上限。它只作为参考，不代表系统真实准确率。
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.96, 0.05), 2)
    y_true = np.asarray(y_true)
    pmax = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = pred == y_true
    n = len(y_true)

    rows = []
    for t in thresholds:
        auto = pmax >= t
        auto_n = int(auto.sum())
        review_n = n - auto_n
        auto_acc = float(correct[auto].mean()) if auto_n else 0.0
        rows.append(
            {
                "threshold": float(t),
                "auto_ratio": round(auto_n / n, 4),
                "auto_accuracy": round(auto_acc, 4),
                "review_ratio": round(review_n / n, 4),
                "upper_accuracy": round((int(correct[auto].sum()) + review_n) / n, 4),
            }
        )
    return rows


def confusion_pairs(cm: np.ndarray, names: list[str], k: int = 8) -> list[dict]:
    """列出最容易被混淆的类别对（排除对角线）。"""
    pairs = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": names[i], "pred": names[j], "count": int(cm[i, j])})
    return sorted(pairs, key=lambda x: x["count"], reverse=True)[:k]


def error_examples(
    texts: list[str],
    y_true,
    y_pred,
    proba: np.ndarray,
    names: list[str],
    k: int = 10,
) -> list[dict]:
    """挑出置信度较高但判错的例子，这类错误比低置信度错误更值得分析。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pmax = proba.max(axis=1)
    wrong = np.where(y_true != y_pred)[0]
    wrong = wrong[np.argsort(-pmax[wrong])][:k]
    rows = []
    for i in wrong:
        snippet = " ".join(texts[i].split())[:120]
        rows.append(
            {
                "true": names[int(y_true[i])],
                "pred": names[int(y_pred[i])],
                "confidence": round(float(pmax[i]), 4),
                "snippet": snippet,
            }
        )
    return rows


def channel_mix(y_pred, names: list[str], mapping: dict[str, str]) -> dict[str, int]:
    """预测结果落到各频道的数量分布。"""
    counter = Counter(mapping.get(names[int(i)], "未映射") for i in y_pred)
    return dict(counter.most_common())
