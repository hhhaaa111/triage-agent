"""Agent 主体：把校验、向量化、分类、置信度判定、频道路由与关键词解释串成一条流程。

设计要点
--------
1. 先校验再推理：空输入、过短、超长都在进入模型之前处理，避免用异常兜底。
2. 置信度门槛：低于阈值的条目一律转人工，不硬给结论。
3. 领域外检测：未登录词比例过高说明文本与训练语料分布差得远，同样转人工。
4. 可解释：每个结论附带命中该类的判别词，运营能复核理由而不只是看标签。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .config import (
    CATEGORY_TO_CHANNEL,
    DEFAULT_THRESHOLD,
    KEYWORD_PATH,
    MAX_LEN,
    MIN_LEN,
    MODEL_PATH,
    OOV_RATIO_MAX,
)
from .features import clean_text, load_keyword_table, top_contributions


@dataclass
class Decision:
    """一条输入的完整判定结果。"""

    text: str
    label: str | None
    channel: str | None
    confidence: float
    action: str  # auto_route 或 review
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_row(self) -> dict:
        return {
            "snippet": " ".join(self.text.split())[:80],
            "label": self.label or "",
            "channel": self.channel or "",
            "confidence": round(self.confidence, 4),
            "action": self.action,
            "reasons": "；".join(self.reasons),
            "notes": "；".join(self.notes),
            "latency_ms": round(self.latency_ms, 2),
        }


class TriageAgent:
    """评论 / 工单文本分拣 Agent。"""

    def __init__(
        self,
        pipeline,
        class_names: list[str],
        keyword_table: dict,
        threshold: float = DEFAULT_THRESHOLD,
        oov_ratio_max: float = OOV_RATIO_MAX,
        min_len: int = MIN_LEN,
        max_len: int = MAX_LEN,
        explain_coef=None,
    ) -> None:
        self.pipeline = pipeline
        self.class_names = list(class_names)
        self.keyword_table = keyword_table
        self.explain_coef = explain_coef
        self.threshold = float(threshold)
        self.oov_ratio_max = float(oov_ratio_max)
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self._vocab = pipeline.named_steps["tfidf"].vocabulary_
        self._feature_names = None

    # ---------------- 构造与持久化 -----------------------------

    @classmethod
    def load(
        cls,
        model_path: Path = MODEL_PATH,
        keyword_path: Path = KEYWORD_PATH,
        threshold: float | None = None,
    ) -> "TriageAgent":
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"未找到模型文件 {model_path}，请先运行 python -m triage train"
            )
        bundle = joblib.load(model_path)
        agent = cls(
            pipeline=bundle["pipeline"],
            class_names=bundle["class_names"],
            keyword_table=load_keyword_table(keyword_path),
            threshold=bundle.get("threshold", DEFAULT_THRESHOLD) if threshold is None else threshold,
            explain_coef=bundle.get("explain_coef"),
        )
        return agent

    # ---------------- 内部工具 -----------------------------

    def _validate(self, raw) -> tuple[str, list[str]]:
        """输入校验，返回截断后的文本与提示信息。"""
        notes: list[str] = []
        if raw is None:
            return "", ["输入为空"]
        text = str(raw).strip()
        if not text:
            return "", ["输入为空"]
        if len(text) < self.min_len:
            notes.append(f"文本过短（{len(text)} 字符），信息不足")
        if len(text) > self.max_len:
            text = text[: self.max_len]
            notes.append(f"文本超长，已截断至 {self.max_len} 字符")
        return text, notes

    def _oov_ratio(self, cleaned: str) -> float:
        tokens = [t for t in cleaned.split() if len(t) >= 2]
        if not tokens:
            return 1.0
        unknown = sum(1 for t in tokens if t not in self._vocab)
        return unknown / len(tokens)

    def _predict(self, cleaned: str) -> tuple[int, float]:
        proba = self.pipeline.predict_proba([cleaned])[0]
        index = int(np.argmax(proba))
        return index, float(proba[index])

    def _evidence(self, cleaned: str, label: str, index: int, k: int = 3) -> list[str]:
        """先看模型局部贡献，取不到再退回全局判别词表。

        局部贡献 = TF-IDF 权重 × 该类别线性权重，能保证列出的词真实出现在原文中。
        """
        if self.explain_coef is not None:
            vectorizer = self.pipeline.named_steps["tfidf"]
            if self._feature_names is None:
                self._feature_names = vectorizer.get_feature_names_out()
            hits = top_contributions(
                vectorizer, self.explain_coef[index], cleaned, k=k, feature_names=self._feature_names
            )
            if hits:
                return [f"{word}(+{score:.2f})" for word, score in hits]

        tokens = set(cleaned.split())
        table_hits = [
            item
            for item in self.keyword_table.get(label, [])
            if item["word"] in tokens or (" " in item["word"] and item["word"] in cleaned)
        ]
        table_hits.sort(key=lambda item: item["score"], reverse=True)
        if not table_hits:
            return ["未命中该类的典型判别词"]
        return [f"{item['word']}(+{item['score']:.2f})" for item in table_hits[:k]]

    # ---------------- 对外接口 -----------------------------

    def triage(self, raw_text) -> Decision:
        """对单条文本执行完整流程。"""
        t0 = time.perf_counter()
        text, notes = self._validate(raw_text)
        force_review = any("过短" in note for note in notes)
        if not text:
            return Decision(
                text=str(raw_text or ""),
                label=None,
                channel=None,
                confidence=0.0,
                action="review",
                reasons=[],
                notes=notes,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        cleaned = clean_text(text)
        if not cleaned:
            notes.append("清洗后无有效词，转人工")
            return Decision(
                text=text,
                label=None,
                channel=None,
                confidence=0.0,
                action="review",
                reasons=[],
                notes=notes,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        index, confidence = self._predict(cleaned)
        label = self.class_names[index]
        channel = CATEGORY_TO_CHANNEL.get(label)
        reasons = self._evidence(cleaned, label, index)

        oov = self._oov_ratio(cleaned)
        if oov > self.oov_ratio_max:
            notes.append(f"未登录词比例 {oov:.0%}，疑似领域外文本")

        action = "auto_route"
        if force_review:
            action = "review"
        if confidence < self.threshold:
            action = "review"
            notes.append(f"置信度 {confidence:.2f} 低于阈值 {self.threshold:.2f}")
        if oov > self.oov_ratio_max:
            action = "review"

        return Decision(
            text=text,
            label=label,
            channel=channel,
            confidence=confidence,
            action=action,
            reasons=reasons,
            notes=notes,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def triage_many(self, texts: list[str]) -> list[Decision]:
        return [self.triage(t) for t in texts]

    def describe(self) -> dict:
        return {
            "classes": len(self.class_names),
            "channels": len(set(CATEGORY_TO_CHANNEL.values())),
            "threshold": self.threshold,
            "vocab_size": len(self._vocab),
            "oov_ratio_max": self.oov_ratio_max,
        }
