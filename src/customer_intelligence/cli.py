"""Installed command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import fit_bg_nbd_comparison, fit_inactivity, fit_revenue, fit_revenue_90d
from .pipeline import prepare_dataset
from .segmentation import fit_segmentation
from .retention import build_retention_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer Value & Retention Intelligence")
    parser.add_argument("command", choices=["prepare", "inactivity", "revenue", "clv", "segments", "retention", "all"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    actions = {
        "prepare": lambda: prepare_dataset(args.root),
        "inactivity": lambda: fit_inactivity(args.root),
        "revenue": lambda: (fit_revenue_90d(args.root), fit_revenue(args.root)),
        "clv": lambda: fit_bg_nbd_comparison(args.root),
        "segments": lambda: fit_segmentation(args.root),
        "retention": lambda: build_retention_review(args.root),
        "all": lambda: (
            prepare_dataset(args.root), fit_inactivity(args.root),
            fit_revenue_90d(args.root), fit_revenue(args.root),
            fit_bg_nbd_comparison(args.root), fit_segmentation(args.root),
            build_retention_review(args.root),
        ),
    }
    actions[args.command]()


if __name__ == "__main__":
    main()
