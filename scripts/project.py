"""Prepare data and run the customer intelligence experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_intelligence.experiments import fit_bg_nbd_comparison, fit_inactivity, fit_revenue, fit_revenue_90d
from customer_intelligence.pipeline import prepare_dataset
from customer_intelligence.segmentation import fit_segmentation
from customer_intelligence.retention import build_retention_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Customer Value & Retention Intelligence.")
    parser.add_argument(
        "command",
        choices=["prepare", "inactivity", "revenue", "clv", "segments", "retention", "all"],
        help="Select a pipeline stage; all runs the full local analysis.",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_dataset(ROOT)
    elif args.command == "inactivity":
        fit_inactivity(ROOT)
    elif args.command == "revenue":
        fit_revenue_90d(ROOT)
        fit_revenue(ROOT)
    elif args.command == "clv":
        fit_bg_nbd_comparison(ROOT)
    elif args.command == "segments":
        fit_segmentation(ROOT)
    elif args.command == "retention":
        build_retention_review(ROOT)
    elif args.command == "all":
        prepare_dataset(ROOT)
        fit_inactivity(ROOT)
        fit_revenue_90d(ROOT)
        fit_revenue(ROOT)
        fit_bg_nbd_comparison(ROOT)
        fit_segmentation(ROOT)
        build_retention_review(ROOT)


if __name__ == "__main__":
    main()
