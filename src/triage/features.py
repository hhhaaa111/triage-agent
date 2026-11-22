"""文本清洗、向量化与"关键词证据库"。

关键词证据库是一张离线表：对每个类别，用带平滑的对数几率比
（log-odds ratio）给词打分，取分数最高的若干词。它的作用是让 Agent
能说出"为什么把这条帖子分到这个类别"，而不是只给一个黑箱标签。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

_EMAIL_RE = re.compile(r"\S+@\S+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_HEX_RE = re.compile(r"\b[x][-+]?[0-9a-f]{4,}\b")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def clean_text(text: str) -> str:
    """把原始帖子压成适合词袋模型的纯小写词序列。

    处理顺序：去 HTML 残留 → 去邮箱/网址/十六进制签名 → 去掉引用行（以 > 开头）
    → 非字母数字替换为空格 → 合并空白 → 转小写。
    """
    if not isinstance(text, str):
        text = str(text)
    text = _EMAIL_RE.sub(" emailaddr ", text)
    text = _URL_RE.sub(" urladdr ", text)
    text = _HEX_RE.sub(" hexnum ", text)
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    text = " ".join(lines)
    text = _NON_WORD_RE.sub(" ", text.lower())
    return " ".join(text.split())


def build_vectorizer() -> TfidfVectorizer:
    """TF-IDF 词袋：一元 + 二元词组，去掉英文停用词与低频噪声。"""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words="english",
        token_pattern=r"(?u)\b\w{2,}\b",
        lowercase=True,
    )


def build_keyword_table(
    texts: Sequence[str],
    labels: Sequence[int],
    class_names: Sequence[str],
    top_k: int = 30,
    min_df: int = 5,
    alpha: float = 0.5,
) -> dict[str, list[dict[str, float]]]:
    """按类别统计对数几率比，得到每类最具判别力的词。

    对类别 c 与词 w：score = log P(w|c) - log P(w|非 c)，两端都做 alpha 平滑。
    只保留正分（即更偏向该类）的前 top_k 个词。
    """
    cleaned = [clean_text(t) for t in texts]
    counter = CountVectorizer(
        ngram_range=(1, 1), min_df=min_df, stop_words="english", lowercase=True
    )
    matrix = counter.fit_transform(cleaned)
    vocab = counter.get_feature_names_out()
    y = np.asarray(labels)

    total = np.asarray(matrix.sum(axis=0)).ravel() + alpha
    n_docs = len(cleaned)
    v_size = len(vocab)
    table: dict[str, list[dict[str, float]]] = {}

    for index, name in enumerate(class_names):
        in_class = y == index
        pos = np.asarray(matrix[in_class].sum(axis=0)).ravel() + alpha
        pos_total = pos.sum()
        neg = total - pos + alpha
        neg_total = neg.sum()
        score = np.log(pos / (pos_total + alpha * v_size)) - np.log(
            neg / (neg_total + alpha * v_size)
        )
        positive = np.argsort(score)[::-1][:top_k]
        table[name] = [
            {"word": str(vocab[i]), "score": round(float(score[i]), 4)}
            for i in positive
            if score[i] > 0
        ]
    return table


def save_keyword_table(table: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")


def load_keyword_table(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def top_contributions(
    vectorizer: TfidfVectorizer,
    coef_row: np.ndarray,
    cleaned: str,
    k: int = 3,
    feature_names: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """计算当前文本对某一类别的局部贡献，返回命中的词及其贡献值。

    贡献 = TF-IDF 权重 × 该类别的线性权重。只保留文本中真实出现且贡献为正的词，
    因此这条理由是可复核的：运营能在原文里找到这些词。
    只遍历稀疏矩阵的非零元素，避免把 14 万维向量展开成稠密数组。
    """
    matrix = vectorizer.transform([cleaned]).tocsr()
    data = matrix.data
    indices = matrix.indices
    if data.size == 0:
        return []
    coef = np.asarray(coef_row).ravel()
    scores = data * coef[indices]
    if feature_names is None:
        feature_names = vectorizer.get_feature_names_out()

    order = np.argsort(-scores)
    tokens = set(cleaned.split())
    picked: list[tuple[str, float]] = []
    for pos in order:
        if scores[pos] <= 0:
            break
        word = str(feature_names[indices[pos]])
        present = word in tokens if " " not in word else word in cleaned
        if present:
            picked.append((word, float(scores[pos])))
        if len(picked) >= k:
            break
    return picked
