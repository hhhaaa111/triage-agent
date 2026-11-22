"""候选模型：朴素贝叶斯、逻辑回归、线性 SVM 与带概率校准的 SVM。"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_vectorizer

# 名称 -> (构造函数, 是否输出概率)
MODEL_SPECS = {
    "mnb": ("朴素贝叶斯（MultinomialNB）", True),
    "lr": ("逻辑回归（LogisticRegression）", True),
    "svc": ("线性 SVM（LinearSVC）", False),
    "calibrated_svc": ("线性 SVM + Sigmoid 概率校准", True),
}


def build_pipeline(name: str, vectorizer: TfidfVectorizer | None = None) -> Pipeline:
    """构造"向量化 + 分类器"的一条流水线。

    vectorizer 可传入已拟合的实例以便复用词表，默认新建一个。
    """
    vec = vectorizer if vectorizer is not None else build_vectorizer()
    if name == "mnb":
        clf = MultinomialNB(alpha=0.5)
    elif name == "lr":
        clf = LogisticRegression(max_iter=2000, C=3.0)
    elif name == "svc":
        clf = LinearSVC(C=1.0)
    elif name == "calibrated_svc":
        clf = CalibratedClassifierCV(LinearSVC(C=1.0), method="sigmoid", cv=3, n_jobs=-1)
    else:
        raise ValueError(f"未知模型: {name}")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def scores_of(pipeline: Pipeline, texts: list[str]) -> tuple[list[str], "object"]:
    """统一拿到预测标签与置信度分数：有概率用 max proba，否则用决策间隔。"""
    clf = pipeline.named_steps["clf"]
    labels = pipeline.predict(texts)
    if hasattr(clf, "predict_proba"):
        return list(labels), pipeline.predict_proba(texts)
    decision = pipeline.decision_function(texts)
    return list(labels), decision
