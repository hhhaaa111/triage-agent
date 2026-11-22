"""从测试集里抽 12 条真实帖子 + 3 条边界输入，生成演示工单。

用法：python scripts/make_demo.py
输出：demo/sample_tickets.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage.config import CATEGORY_TO_CHANNEL, DEMO_DIR  # noqa: E402
from triage.data import load_newsgroups  # noqa: E402


def main() -> None:
    _, _, test_texts, y_test, names = load_newsgroups()
    channels = list(dict.fromkeys(CATEGORY_TO_CHANNEL.values()))
    picked = []
    for channel in channels:
        count = 0
        for text, label in zip(test_texts, y_test):
            if CATEGORY_TO_CHANNEL[names[int(label)]] != channel:
                continue
            body = " ".join(text.split())
            if len(body) < 120:
                continue
            picked.append({"id": f"T{len(picked) + 1:02d}", "channel_hint": channel, "text": body[:900]})
            count += 1
            if count == 2:
                break

    picked.append({"id": "E01", "channel_hint": "边界", "text": "  "})
    picked.append({"id": "E02", "channel_hint": "边界", "text": "help"})
    picked.append({
        "id": "E03",
        "channel_hint": "边界",
        "text": "zzqqxx vvbbnn mmkkll qqwwee rrttyy uuiioo ppaass ddffgg hhjjkk",
    })

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    out = DEMO_DIR / "sample_tickets.jsonl"
    out.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in picked) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {out}，共 {len(picked)} 条")


if __name__ == "__main__":
    main()
