"""Exercise the production refit with labels mature only after its origin."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from customer_intelligence import experiments
from customer_intelligence.features import NUMERIC_FEATURES
from customer_intelligence.revenue_challengers import select_from_available_outcomes


class SpyRegressor:
    fits = []

    def fit(self, x, y):
        self.fits.append(np.asarray(y).tolist())
        self.mean = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean)


class RevenueTimingTest(unittest.TestCase):
    def test_selection_excludes_outcomes_after_final_origin(self):
        dates = pd.date_range("2010-12-01", "2011-05-01", freq="MS")
        panel = pd.DataFrame(dict(cutoff=dates, horizon_end_exclusive=[d + pd.DateOffset(months=6) for d in dates]))
        records = []
        for date in dates:
            for name in ["early_winner", "late_winner"]:
                error = (50 if name == "early_winner" else 60) if date.month == 12 else (100 if name == "early_winner" else 1)
                records.append(dict(model=name, cutoff=str(date.date()), status="completed", mae_gbp=error, rmse_gbp=error, aggregate_bias_pct=0.0, tweedie_deviance_p1_5=error, revenue_capture_at_capacity=0.5))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            selected = select_from_available_outcomes(root, "6m", pd.DataFrame(records), panel, "fixture")
        self.assertEqual(selected, "early_winner")

    def test_refit_excludes_labels_unavailable_at_origin(self):
        dates = pd.to_datetime(["2010-06-01", "2010-09-01", "2010-12-01", "2011-03-01", "2011-06-01"])
        features = pd.DataFrame({c: np.ones(len(dates)) for c in NUMERIC_FEATURES})
        features["customer_id"], features["cutoff"], features["country"] = 1, dates, "United Kingdom"
        labels = pd.DataFrame(dict(customer_id=1, cutoff=dates, horizon="6m", label_mature=True,
            horizon_end_exclusive=[d + pd.DateOffset(months=6) for d in dates],
            future_revenue_gbp=[100.0, 200.0, 300.0, 400.0, 500.0]))
        # All labels are complete at source end, but March's is unavailable in June.
        prepared = (features, pd.DataFrame(), labels, dict(source_sha256="fixture"))
        SpyRegressor.fits = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            (root / "data" / "processed").mkdir(parents=True)
            with patch.object(experiments, "load_prepared", return_value=prepared), \
                 patch.object(experiments, "revenue_candidates", side_effect=lambda: {"spy": SpyRegressor()}), \
                 patch("builtins.print"):
                experiments.fit_revenue(root)
        self.assertEqual(SpyRegressor.fits, [[100.0], [100.0, 200.0, 300.0]])


if __name__ == "__main__":
    unittest.main()
