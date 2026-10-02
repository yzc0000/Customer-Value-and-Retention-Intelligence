"""Protect forecast-date joins, outcome exclusion and filter/capacity scope."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from customer_intelligence.retention import HISTORY, GROUPS, assign_review_groups, combine_customer_review, compare_policies, filter_review, rank_review


class RetentionReviewTest(unittest.TestCase):
    def setUp(self):
        self.cutoff = pd.Timestamp("2011-09-01")
        keys = dict(customer_id=[10, 20, 30], cutoff=self.cutoff)
        self.features = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in HISTORY if c != "country"})
        self.features["customer_id"], self.features["cutoff"] = keys["customer_id"], self.cutoff
        self.features["country"] = ["UK", "UK", "France"]
        self.features["avg_order_value_gbp"] = [100.0, 200.0, 300.0]
        self.risk = pd.DataFrame(dict(**keys, lstm_weekly_risk=[0.9, 0.6, 0.2], inactive=[1, 0, 0]))
        self.revenue = pd.DataFrame(dict(**keys, predicted_revenue_gbp=[50.0, 200.0, 1000.0], actual_revenue_gbp=[0.0, 100.0, 500.0]))
        self.segments = pd.DataFrame(dict(**keys, kmeans_segment=[1, 0, 0], gmm_segment=[1, 0, 2], gmm_max_membership=[0.8, 0.7, 0.9]))

    def combined(self, risk=None, revenue=None):
        return combine_customer_review(self.features, self.risk if risk is None else risk,
            self.revenue if revenue is None else revenue, self.segments, self.cutoff, "lstm_weekly", "xgboost")

    def test_join_uses_keys_and_excludes_future_outcomes(self):
        actual = self.combined(self.risk.iloc[[2, 0, 1]], self.revenue.iloc[[1, 2, 0]])
        np.testing.assert_array_equal(actual.inactive_probability_90d, [0.9, 0.6, 0.2])
        np.testing.assert_array_equal(actual.predicted_revenue_90d_gbp, [50, 200, 1000])
        self.assertFalse({"actual_revenue_gbp", "inactive", "future_revenue_gbp"} & set(actual.columns))
        altered_risk, altered_revenue = self.risk.copy(), self.revenue.copy()
        altered_risk["inactive"], altered_revenue["actual_revenue_gbp"] = 999, 1e12
        pd.testing.assert_frame_equal(actual, self.combined(altered_risk, altered_revenue))

    def test_rejects_different_dates_missing_customers_and_duplicates(self):
        different = self.risk.assign(cutoff=pd.Timestamp("2011-06-01"))
        with self.assertRaisesRegex(ValueError, "shared forecast cutoff"):
            self.combined(different)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            self.combined(self.risk.iloc[:2])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.combined(pd.concat([self.risk, self.risk.iloc[:1]]))

    def test_filters_determine_queue_capacity_and_policy_comparisons(self):
        rows = assign_review_groups(self.combined(), 0.5, 200.0)
        filtered = filter_review(rows, [0], "UK", 0.5)
        ranked = rank_review(filtered, capacity=0.10)
        self.assertEqual(ranked.customer_id.tolist(), [20])
        self.assertEqual(ranked.in_review_queue.tolist(), [True])
        self.assertEqual(ranked.filtered_population_customers.tolist(), [1])
        comparison = compare_policies(filtered, capacity=0.10)
        self.assertTrue(comparison.customers.eq(1).all())
        self.assertTrue(comparison.predicted_revenue_90d_gbp.eq(200).all())

    def test_ranking_rounding_and_ties_are_deterministic(self):
        rows = self.combined()
        ranked = rank_review(rows, capacity=0.34)
        self.assertEqual(ranked.loc[ranked.in_review_queue, "customer_id"].tolist(), [20, 10])
        tied = rows.assign(review_priority_score=1.0, inactive_probability_90d=0.5, predicted_revenue_90d_gbp=10.0)
        self.assertEqual(rank_review(tied.iloc[::-1]).customer_id.tolist(), [10, 20, 30])

    def test_group_boundaries_and_thresholds_are_preserved_in_exports(self):
        rows = assign_review_groups(self.combined(), 0.6, 200.0)
        self.assertEqual(rows.review_group.tolist(), [GROUPS[2], GROUPS[0], GROUPS[1]])
        self.assertTrue(rows.risk_threshold.eq(0.6).all())
        self.assertTrue(rows.historical_order_value_threshold_gbp.eq(200).all())

    def test_empty_and_zero_capacity_are_distinct(self):
        rows = self.combined()
        empty = filter_review(rows, [])
        self.assertTrue(rank_review(empty).empty)
        zero = rank_review(rows, capacity=0)
        self.assertEqual(len(zero), 3)
        self.assertFalse(zero.in_review_queue.any())
        self.assertTrue(compare_policies(rows, capacity=0).customers.eq(0).all())

    def test_invalid_probabilities_are_rejected(self):
        for value in [-0.1, 1.1, np.nan]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.combined(self.risk.assign(lstm_weekly_risk=value))


if __name__ == "__main__":
    unittest.main()
