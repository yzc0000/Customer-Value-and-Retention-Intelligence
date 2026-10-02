"""Run GAM, Pareto/NBD, NGBoost and probabilistic LSTM revenue comparisons."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_intelligence.revenue_challengers import main

if __name__ == "__main__":
    main()
