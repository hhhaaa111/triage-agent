"""全局配置：路径、随机种子、频道映射、默认阈值。"""

from pathlib import Path

SEED = 20261122

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"
DEMO_DIR = PROJECT_ROOT / "demo"

MODEL_PATH = MODEL_DIR / "triage_model.joblib"
KEYWORD_PATH = MODEL_DIR / "keyword_table.json"

# 20 Newsgroups 的 20 个原始类别，顺序与数据集 target_names 一致
CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]

# 二级路由：把 20 个细类归并成 6 个业务频道，供运营侧分流
CHANNELS = {
    "计算机与技术": [
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
    ],
    "科学与工程": ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
    "政治与社会": ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc"],
    "信仰与宗教": ["alt.atheism", "soc.religion.christian", "talk.religion.misc"],
    "娱乐与休闲": ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
    "交易与杂项": ["misc.forsale"],
}

CATEGORY_TO_CHANNEL = {
    category: channel for channel, cats in CHANNELS.items() for category in cats
}

# 低于该置信度的帖子不自动分流，进入人工复核队列
DEFAULT_THRESHOLD = 0.55

# 输入校验阈值
MIN_LEN = 20
MAX_LEN = 4000
# 文本中未登录词比例超过该值视为领域外输入，转人工。
# 依据：训练集内该比例中位数 0.52、95 分位 0.66，取 0.70 留出余量；
# 纯乱码文本该比例可达 1.00，仍会被拦下。
OOV_RATIO_MAX = 0.70
