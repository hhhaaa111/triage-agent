"""Agent 行为的最小单元测试：无外部依赖，使用自造的小语料。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from triage.agent import TriageAgent  # noqa: E402
from triage.config import CATEGORIES, CATEGORY_TO_CHANNEL  # noqa: E402
from triage.data import read_jsonl  # noqa: E402
from triage.features import build_keyword_table, clean_text  # noqa: E402
from triage.models import build_pipeline  # noqa: E402

TRAIN = {
    "rec.autos": [
        "engine oil change brake pads honda civic mileage",
        "my car wont start battery alternator mechanic",
        "rear wheel drive handling suspension tires",
        "spark plugs fuel injection carburetor tuning",
        "transmission clutch shifting gearbox repair",
    ],
    "sci.med": [
        "patient fever symptoms diagnosis treatment hospital",
        "blood pressure medication dosage doctor visit",
        "xray mri scan radiology imaging result",
        "surgery recovery pain management nursing care",
        "allergy reaction skin rash dermatology clinic",
    ],
    "comp.graphics": [
        "opengl rendering polygon texture shading library",
        "image format jpeg gif compression pixel color",
        "ray tracing algorithm scene lighting render",
        "graphics card driver display resolution monitor",
        "3d modeling mesh vertices animation software",
    ],
}


def build_small_agent() -> TriageAgent:
    texts, labels, names = [], [], list(TRAIN)
    for index, (name, items) in enumerate(TRAIN.items()):
        for item in items:
            texts.append(item)
            labels.append(index)
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
    pipe = build_pipeline("mnb", vectorizer=vec)
    pipe.fit(texts, labels)
    table = build_keyword_table(texts, labels, names, top_k=10, min_df=1)
    return TriageAgent(pipe, names, table, threshold=0.30)


class TestCleaning(unittest.TestCase):
    def test_removes_email_url_and_quote(self):
        raw = "Check this http://example.com and mail me a@b.com\n> quoted line\nENGINE oil"
        cleaned = clean_text(raw)
        self.assertNotIn("example.com", cleaned)
        self.assertNotIn("a b com", cleaned)
        self.assertIn("engine", cleaned)
        self.assertNotIn("quoted", cleaned)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = build_small_agent()

    def test_empty_input_goes_to_review(self):
        d = self.agent.triage("   ")
        self.assertEqual(d.action, "review")
        self.assertIsNone(d.label)
        self.assertIn("输入为空", "；".join(d.notes))

    def test_short_input_forced_to_review(self):
        d = self.agent.triage("help")
        self.assertEqual(d.action, "review")

    def test_normal_input_is_labelled(self):
        d = self.agent.triage("my car battery is dead and the alternator will not charge")
        self.assertIn(d.label, TRAIN)
        self.assertEqual(d.channel, CATEGORY_TO_CHANNEL[d.label])
        self.assertTrue(d.reasons)

    def test_gibberish_triggers_domain_check(self):
        d = self.agent.triage("zzqqxx vvbbnn mmkkll qqwwee rrttyy uuiioo")
        self.assertEqual(d.action, "review")
        self.assertTrue(any("未登录词" in note for note in d.notes))

    def test_long_input_is_truncated(self):
        d = self.agent.triage("engine oil brake " * 500)
        self.assertLessEqual(len(d.text), self.agent.max_len)
        self.assertTrue(any("截断" in note for note in d.notes))


class TestConfigAndIO(unittest.TestCase):
    def test_channel_mapping_covers_all_categories(self):
        for name in CATEGORIES:
            self.assertIn(name, CATEGORY_TO_CHANNEL)

    def test_jsonl_reader_rejects_missing_text(self):
        tmp = ROOT / "tests" / "_tmp_missing.jsonl"
        tmp.write_text('{"id": 1}\n', encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                read_jsonl(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
