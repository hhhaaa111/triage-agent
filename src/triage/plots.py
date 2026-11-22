"""报告与 README 用到的图表生成，中文字体统一用微软雅黑。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"

PRIMARY = "#2F5D8A"
ACCENT = "#C8642A"
GREY = "#7A7A7A"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def model_comparison(results: dict, path: Path) -> None:
    """四个候选模型 + 多数类基线的宏 F1 与准确率对比。"""
    keys = ["majority", "mnb", "lr", "svc", "calibrated_svc"]
    labels = ["多数类基线", "朴素贝叶斯", "逻辑回归", "线性 SVM", "SVM+概率校准"]
    acc = [results[k]["accuracy"] for k in keys]
    f1 = [results[k]["macro_f1"] for k in keys]

    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    b1 = ax.bar(x - width / 2, acc, width, label="准确率", color=PRIMARY)
    b2 = ax.bar(x + width / 2, f1, width, label="宏平均 F1", color=ACCENT)
    ax.bar_label(b1, fmt="%.3f", fontsize=7.5, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=7.5, padding=2)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("分数", fontsize=9)
    ax.set_title("候选模型在测试集（7532 条）上的表现", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    _save(fig, path)


def confusion(cm: np.ndarray, names: list[str], path: Path) -> None:
    """20×20 混淆矩阵热力图，对角线归一化后着色。"""
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(norm, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(names)), names, fontsize=6.5)
    ax.set_xlabel("预测类别", fontsize=9)
    ax.set_ylabel("真实类别", fontsize=9)
    ax.set_title("混淆矩阵（按真实类别行归一化）", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def threshold_sweep(sweep: list[dict], path: Path) -> None:
    """置信度阈值 -> 自动分流比例 / 自动分流准确率 / 人工复核量。"""
    t = [s["threshold"] for s in sweep]
    auto_ratio = [s["auto_ratio"] for s in sweep]
    auto_acc = [s["auto_accuracy"] for s in sweep]
    upper = [s["upper_accuracy"] for s in sweep]

    fig, ax1 = plt.subplots(figsize=(7.2, 3.6))
    ax1.plot(t, auto_ratio, color=PRIMARY, marker="o", ms=3, label="自动分流比例")
    ax1.set_xlabel("置信度阈值", fontsize=9)
    ax1.set_ylabel("自动分流比例", fontsize=9, color=PRIMARY)
    ax1.tick_params(axis="y", labelcolor=PRIMARY)
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.25, linestyle=":")

    ax2 = ax1.twinx()
    ax2.plot(t, auto_acc, color=ACCENT, marker="s", ms=3, label="自动分流准确率")
    ax2.plot(t, upper, color=GREY, linestyle="--", label="复核全纠正时的上限")
    ax2.set_ylabel("准确率", fontsize=9)
    ax2.set_ylim(0.4, 1.02)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=7.5, loc="lower left")
    ax1.set_title("阈值扫描：分流覆盖与准确率的权衡", fontsize=10.5)
    _save(fig, path)


def per_class_f1(f1s: dict[str, float], path: Path) -> None:
    """逐类别 F1 横向条形图，便于看出哪些类别难分。"""
    items = sorted(f1s.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [ACCENT if v < 0.5 else PRIMARY for v in vals]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(names, vals, color=colors, height=0.68)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1", fontsize=9)
    ax.set_title("逐类别 F1（测试集）", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    _save(fig, path)


def ablation(features: dict, path: Path) -> None:
    """特征配置消融：五种配置的准确率与宏 F1。"""
    short = {
        "完整配置（1–2 gram + 停用词 + sublinear）": "完整配置",
        "仅一元词（1 gram）": "仅一元词",
        "不去停用词": "去停用词",
        "min_df=1（更大词表）": "min_df=1",
        "关闭 sublinear_tf": "关闭 sublinear",
    }
    labels = [short.get(k, k) for k in features]
    acc = [features[k]["accuracy"] for k in features]
    f1 = [features[k]["macro_f1"] for k in features]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    b1 = ax.bar(x - width / 2, acc, width, label="准确率", color=PRIMARY)
    b2 = ax.bar(x + width / 2, f1, width, label="宏平均 F1", color=ACCENT)
    ax.bar_label(b1, fmt="%.3f", fontsize=7.5, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=7.5, padding=2)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("分数", fontsize=9)
    ax.set_title("特征配置消融（线性 SVM，同一训练/测试划分）", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    _save(fig, path)


def pipeline_diagram(path: Path) -> None:
    """Agent 工作流示意图：输入 → 校验 → 表示 → 分类 → 判定 → 输出。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, title, body, color="#EAF1F8"):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.12",
            linewidth=0.8, edgecolor="#4A4A4A", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=8.8, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.14, body, ha="center", va="center", fontsize=7.2, color="#333333")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=0.9, color="#4A4A4A"))

    box(0.25, 4.15, 2.0, 1.35, "① 输入", "单条文本 / JSONL\n批量工单")
    box(2.7, 4.15, 2.0, 1.35, "② 输入校验", "空值、过短、超长\n清洗引用与签名")
    box(5.15, 4.15, 2.0, 1.35, "③ TF-IDF 表示", "1–2 gram\n词表约 14 万维")
    box(7.6, 4.15, 2.1, 1.35, "④ 分类推理", "线性 SVM\nSigmoid 概率校准", "#FDF1E7")

    box(7.6, 2.25, 2.1, 1.25, "⑤ 判定", "阈值 0.55 + 未登录词\n比例 > 70% 转人工", "#FDF1E7")
    box(5.15, 2.25, 2.0, 1.25, "⑥ 关键词证据", "命中判别词 3 个\n给出可读理由")
    box(2.7, 2.25, 2.0, 1.25, "⑦ 频道路由", "20 细类 → 6 频道\n运营侧派单")
    box(0.25, 2.25, 2.0, 1.25, "⑧ 输出", "自动分流 / 复核队列\nCSV + 理由")

    arrow(2.25, 4.82, 2.7, 4.82)
    arrow(4.7, 4.82, 5.15, 4.82)
    arrow(7.15, 4.82, 7.6, 4.82)
    arrow(8.65, 4.15, 8.65, 3.5)
    arrow(7.6, 2.87, 7.15, 2.87)
    arrow(5.15, 2.87, 4.7, 2.87)
    arrow(2.7, 2.87, 2.25, 2.87)

    ax.text(5.0, 1.55, "人工复核队列（低置信度 / 领域外 / 异常输入）→ 复核结果回流为下一次训练样本",
            ha="center", fontsize=7.6, color="#444444")
    _save(fig, path)
