"""数据加载：20 Newsgroups 语料与本地 JSONL 工单。"""

from __future__ import annotations

import json
import os
from typing import Any

from sklearn.datasets import fetch_20newsgroups

from .config import DATA_DIR, SEED


def load_newsgroups(remove: tuple[str, ...] = ("headers", "footers", "quotes")):
    """加载 20 Newsgroups 的官方 train / test 划分。

    数据由 scikit-learn 自带下载器获取，缓存默认位于 ~/scikit_learn_data；
    如需改到项目内，可在环境变量 SCIKIT_LEARN_DATA 指向 DATA_DIR。
    返回 (训练文本, 训练标签, 测试文本, 测试标签, 类别名列表)。
    """
    os.environ.setdefault("SCIKIT_LEARN_DATA", str(DATA_DIR))
    train = fetch_20newsgroups(
        subset="train", remove=remove, shuffle=True, random_state=SEED
    )
    test = fetch_20newsgroups(
        subset="test", remove=remove, shuffle=True, random_state=SEED
    )
    return train.data, train.target, test.data, test.target, list(train.target_names)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """读取一行一条工单的 JSONL 文件，字段至少包含 text。"""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
            if "text" not in obj:
                raise ValueError(f"{path} 第 {line_no} 行缺少 text 字段")
            records.append(obj)
    return records
