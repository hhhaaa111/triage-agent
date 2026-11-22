# TriageAgent：评论 / 工单文本智能分拣 Agent

《机器学习》课程期末项目。面向社区运营的高频场景：帖子、评论、工单每天涌入，
人工逐条判断该进哪个板块既慢又不一致。本项目把这件事做成一个可运行的 Agent：

输入一条文本（或一批 JSONL）→ 输入校验 → TF-IDF 表示 → 线性 SVM 分类（20 类）
→ 置信度与领域外判定 → 频道路由（20 类归并为 6 个业务频道）→ 输出自动分流或人工复核队列，
并附上命中的判别词作为理由。

## 1. 项目速览

| 项目 | 说明 |
| --- | --- |
| 任务 | 文本分拣 + 路由 + 低置信度转人工 |
| 数据 | 20 Newsgroups（18846 条帖子，20 个类别，官方 train/test 划分） |
| 特征 | TF-IDF，1–2 gram，约 30 万维，去掉英文停用词 |
| 模型 | 线性 SVM + Sigmoid 概率校准（对比：朴素贝叶斯、逻辑回归、裸 SVM） |
| 指标 | 准确率、宏平均 F1、逐类 F1、top-3 准确率、阈值扫描下的分流覆盖率 |
| Agent 决策 | 置信度低于 0.55 或未登录词比例高于 45% 时转人工 |

## 2. 环境与安装

- Python 3.10 及以上（本机验证 3.13.18）
- 无需 GPU，CPU 上训练约 1–2 分钟

```bash
python -m pip install -r requirements.txt
```

首次运行会自动下载 20 Newsgroups 语料（约 20 MB），缓存到 `data/sklearn_data/`。
如网络受限，可先手工下载后放到同一目录。

## 3. 运行步骤

```bash
# 1) 训练主模型，同时生成关键词证据库
python run.py train

# 2) 在官方测试集评估，生成 docs/metrics.json 与 docs/figures/*.png
python run.py evaluate

# 3) 单条文本分拣
python run.py triage --text "Looking for a cheap car, 1995 Honda Civic, runs fine, $2500"

# 4) 批量分拣（一行一条 JSON，字段至少含 text）
python run.py triage --file demo/sample_tickets.jsonl --out out/decisions.csv

# 5) 内置演示
python run.py demo

# 6) 查看当前模型概况
python run.py info
```

## 4. 目录结构

```
triage-agent/
├─ run.py                 # 命令入口
├─ requirements.txt
├─ src/triage/
│  ├─ config.py           # 路径、20 类与 6 频道映射、阈值
│  ├─ data.py             # 数据加载（新闻组语料 / JSONL）
│  ├─ features.py         # 清洗、TF-IDF、关键词证据库
│  ├─ models.py           # 四个候选模型
│  ├─ evaluate.py         # 指标、阈值扫描、混淆对、错误案例
│  ├─ agent.py            # Agent 主流程
│  ├─ plots.py            # 报告图表
│  └─ cli.py              # 子命令
├─ scripts/experiment.py  # 候选模型横向对比实验
├─ demo/sample_tickets.jsonl
├─ tests/test_agent.py
└─ docs/                  # metrics.json 与图表（运行后生成）
```

## 5. 输出说明

`triage` 的每条结果包含：`label`（20 细类）、`channel`（6 频道）、`confidence`、
`action`（`auto_route` / `review`）、`reasons`（命中的判别词及分数）、`notes`（转人工原因）、
`latency_ms`。批量结果写成 CSV（UTF-8 BOM，Excel 直接可读）。

## 6. 复用组件与许可

| 组件 | 版本 | 来源 | 许可证 | 用途 |
| --- | --- | --- | --- | --- |
| scikit-learn | 1.9.1 | https://scikit-learn.org | BSD-3-Clause | TF-IDF、模型、交叉验证、概率校准 |
| NumPy | 2.5.5 | https://numpy.org | BSD-3-Clause | 数值运算 |
| matplotlib | 3.11.2 | https://matplotlib.org | PSF-based | 图表 |
| joblib | 1.6.0 | https://joblib.readthedocs.io | BSD-3-Clause | 模型持久化 |
| 20 Newsgroups | 1997 原始语料 | http://qwone.com/~jason/20Newsgroups/ | 公开学术语料，见原站说明 | 训练与评估 |

本项目新增代码以 MIT 许可证发布，见 `LICENSE`。

## 7. 已知限制

- 语料为英文新闻组帖子，换到中文评论或业务工单需要重新标注数据并重训。
- 20 类中有若干高度重叠类别（如 `comp.os.ms-windows.misc` 与 `comp.windows.x`），
  混淆矩阵显示这两类互错最多，实际业务中建议合并或交给人工。
- 关键词证据只解释"为什么像这一类"，不能证明分类必真；高影响场景仍需人工确认。
