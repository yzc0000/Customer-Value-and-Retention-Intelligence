"""Run isolated revenue experiments without changing the selected models."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_intelligence.revenue_research import main

if __name__ == "__main__":
    main()
