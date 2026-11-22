"""命令行入口：train / evaluate / triage / demo。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from .agent import TriageAgent
from .config import (
    CATEGORY_TO_CHANNEL,
    DEFAULT_THRESHOLD,
    DEMO_DIR,
    DOCS_DIR,
    FIG_DIR,
    KEYWORD_PATH,
    MODEL_PATH,
)
from .data import load_newsgroups, read_jsonl
from .evaluate import classification_metrics, confusion_pairs, error_examples, threshold_sweep
from .features import build_keyword_table, build_vectorizer, clean_text, save_keyword_table
from .models import MODEL_SPECS, build_pipeline
from .plots import confusion, model_comparison, per_class_f1, pipeline_diagram, threshold_sweep as plot_sweep


def _cleaned(texts: list[str]) -> list[str]:
    return [clean_text(t) for t in texts]


def cmd_train(args: argparse.Namespace) -> int:
    train_texts, y_train, _, _, names = load_newsgroups()
    train_clean = _cleaned(train_texts)
    print(f"[训练] {len(train_clean)} 条，{len(names)} 类，模型={args.model}", flush=True)

    pipeline = build_pipeline(args.model)
    pipeline.fit(train_clean, y_train)
    print("[训练] 主模型完成", flush=True)

    cv_pipe = build_pipeline("svc")
    cv = cross_val_score(cv_pipe, train_clean, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"[训练] 5 折交叉验证 宏F1 = {cv.mean():.4f} ± {cv.std():.4f}", flush=True)

    table = build_keyword_table(train_texts, y_train, names)
    save_keyword_table(table, KEYWORD_PATH)

    # 解释器：在同一词表上单独拟合一个线性 SVM，用其权重做局部贡献解释
    from sklearn.svm import LinearSVC

    vectorizer = pipeline.named_steps["tfidf"]
    explainer = LinearSVC(C=1.0).fit(vectorizer.transform(train_clean), y_train)

    bundle = {
        "pipeline": pipeline,
        "class_names": names,
        "threshold": args.threshold,
        "model_name": args.model,
        "model_label": MODEL_SPECS[args.model][0],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_size": len(train_clean),
        "cv_macro_f1_mean": round(float(cv.mean()), 4),
        "cv_macro_f1_std": round(float(cv.std()), 4),
        "vocab_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "explain_coef": explainer.coef_.astype("float32"),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"[训练] 已保存模型 {MODEL_PATH}（词表 {bundle['vocab_size']} 维）", flush=True)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    bundle = joblib.load(MODEL_PATH)
    pipeline, names = bundle["pipeline"], bundle["class_names"]
    agent = TriageAgent.load(threshold=args.threshold)

    _, _, test_texts, y_test, _ = load_newsgroups()
    test_clean = _cleaned(test_texts)
    proba = pipeline.predict_proba(test_clean)
    pred = proba.argmax(axis=1)

    metrics = classification_metrics(y_test, pred, names)
    sweep = threshold_sweep(proba, y_test)

    # Agent 级统计：把校验、OOV 规则也算进来
    decisions = agent.triage_many(test_texts)
    actions = np.array([d.action for d in decisions])
    auto = actions == "auto_route"
    pred_agent = np.array(
        [names.index(d.label) if d.label else -1 for d in decisions]
    )
    auto_correct = int((pred_agent[auto] == np.asarray(y_test)[auto]).sum())
    agent_stats = {
        "threshold": agent.threshold,
        "auto_ratio": round(float(auto.mean()), 4),
        "auto_accuracy": round(auto_correct / max(int(auto.sum()), 1), 4),
        "review_ratio": round(float((~auto).mean()), 4),
        "upper_accuracy": round((auto_correct + int((~auto).sum())) / len(y_test), 4),
        "avg_latency_ms": round(float(np.mean([d.latency_ms for d in decisions])), 3),
    }

    cm = confusion_matrix(y_test, pred, labels=range(len(names)))
    payload = {
        "model": bundle.get("model_label"),
        "trained_at": bundle.get("trained_at"),
        "cv_macro_f1": [bundle.get("cv_macro_f1_mean"), bundle.get("cv_macro_f1_std")],
        "test_size": int(len(y_test)),
        "metrics": metrics,
        "agent": agent_stats,
        "threshold_sweep": sweep,
        "top_confusions": confusion_pairs(cm, names),
        "error_examples": error_examples(test_texts, y_test, pred, proba, names),
        "channel_mix": {
            k: int(v) for k, v in sorted(
                {c: int(sum(1 for p in pred if CATEGORY_TO_CHANNEL.get(names[int(p)]) == c))
                 for c in dict.fromkeys(CATEGORY_TO_CHANNEL.values())}.items(),
                key=lambda kv: -kv[1],
            )
        },
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    exp = DOCS_DIR / "experiment_models.json"
    if exp.exists():
        model_comparison(json.loads(exp.read_text(encoding="utf-8")), FIG_DIR / "fig2_models.png")
    confusion(cm, names, FIG_DIR / "fig3_confusion.png")
    plot_sweep(sweep, FIG_DIR / "fig4_threshold.png")
    per_class_f1(
        {k: v["f1"] for k, v in metrics["per_class"].items()}, FIG_DIR / "fig5_per_class.png"
    )

    print(f"准确率 {metrics['accuracy']:.4f} | 宏F1 {metrics['macro_f1']:.4f} | "
          f"加权F1 {metrics['weighted_f1']:.4f}")
    print(f"Agent：自动分流 {agent_stats['auto_ratio']:.1%}，其中准确率 "
          f"{agent_stats['auto_accuracy']:.1%}，复核算 {agent_stats['review_ratio']:.1%}，"
          f"平均 {agent_stats['avg_latency_ms']:.2f} ms/条")
    print(f"图表与指标已写入 {DOCS_DIR}")
    return 0


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["snippet", "label", "channel", "confidence", "action", "reasons", "notes", "latency_ms"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_decision(index: int, d) -> None:
    print(f"\n[{index}] {d.label or '—'} | 频道 {d.channel or '—'} | 置信度 {d.confidence:.2f} "
          f"| {('自动分流' if d.action == 'auto_route' else '转人工')} | {d.latency_ms:.2f} ms")
    print(f"    理由：{'；'.join(d.reasons) if d.reasons else '—'}")
    if d.notes:
        print(f"    备注：{'；'.join(d.notes)}")


def cmd_triage(args: argparse.Namespace) -> int:
    agent = TriageAgent.load(threshold=args.threshold)
    if args.text:
        d = agent.triage(args.text)
        _print_decision(1, d)
        if args.out:
            _write_csv([d.to_row()], Path(args.out))
        return 0
    if not args.file:
        print("请用 --text 传入一条文本，或用 --file 传入 JSONL 文件", file=sys.stderr)
        return 2
    records = read_jsonl(args.file)
    decisions = [agent.triage(r["text"]) for r in records]
    for i, d in enumerate(decisions, start=1):
        _print_decision(i, d)
    out = Path(args.out) if args.out else DOCS_DIR / "decisions.csv"
    _write_csv([d.to_row() for d in decisions], out)
    print(f"\n已写入 {out}（{len(decisions)} 条）")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    demo_file = DEMO_DIR / "sample_tickets.jsonl"
    if not demo_file.exists():
        print(f"缺少演示文件 {demo_file}", file=sys.stderr)
        return 2
    args.file = str(demo_file)
    args.text = None
    return cmd_triage(args)


def cmd_info(args: argparse.Namespace) -> int:
    if not MODEL_PATH.exists():
        print("模型尚未训练，请先运行 python run.py train", file=sys.stderr)
        return 2
    agent = TriageAgent.load()
    print(json.dumps(agent.describe(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage", description="评论 / 工单文本智能分拣 Agent"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="训练主模型并生成关键词证据库")
    p_train.add_argument("--model", default="calibrated_svc", choices=list(MODEL_SPECS))
    p_train.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="在官方测试集上评估并生成图表")
    p_eval.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_eval.set_defaults(func=cmd_evaluate)

    p_triage = sub.add_parser("triage", help="对单条文本或 JSONL 批量分拣")
    p_triage.add_argument("--text", default=None)
    p_triage.add_argument("--file", default=None)
    p_triage.add_argument("--out", default=None, help="结果 CSV 输出路径")
    p_triage.add_argument("--threshold", type=float, default=None)
    p_triage.set_defaults(func=cmd_triage)

    p_demo = sub.add_parser("demo", help="运行内置演示工单")
    p_demo.add_argument("--out", default=None)
    p_demo.add_argument("--threshold", type=float, default=None)
    p_demo.set_defaults(func=cmd_demo)

    p_info = sub.add_parser("info", help="查看当前模型概况")
    p_info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "threshold"):
        args.threshold = None
    return args.func(args)
