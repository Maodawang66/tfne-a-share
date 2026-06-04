import sys
from pathlib import Path

# scripts/_bootstrap.py -> code/
ROOT = Path(__file__).resolve().parents[1]


def setup():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


setup()
