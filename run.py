"""项目入口：把 src 加入路径后交给 cli。用法 python run.py train / evaluate / demo。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from triage.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
