"""Quick launcher: `python run.py`"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
