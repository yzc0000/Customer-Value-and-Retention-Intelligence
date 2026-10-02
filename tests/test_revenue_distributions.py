"""Independent mathematical checks for the new probabilistic forecasts."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import norm

from customer_intelligence.bgnbd import MONTH_DAYS
from customer_intelligence.pareto_nbd import ParetoNBD, absolute_day, high_season_days
from customer_intelligence.revenue_challenger_models import mixture_quantile


class RevenueDistributionsTest(unittest.TestCase):
    def rows(self):
        return pd.DataFrame(dict(first_day=pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"]),
            last_day=pd.to_datetime(["2010-05-01", "2010-02-01", "2010-05-29"]),
            cutoff=pd.Timestamp("2010-06-01"), frequency=[3, 0, 15], q4_repeat_count=0, foreign=0.0))

    def test_equal_rates_closed_likelihood(self):
        model = ParetoNBD()
        r, alpha, s, beta = 0.8, 8.0, 1.4, 8.0
        params = np.log([r, alpha, s, beta])
        arrays = model._arrays(self.rows())
        actual = model.likelihood_parts(params, arrays)[0]
        x, T = arrays["x"], arrays["T"]
        tx = (absolute_day(self.rows().last_day) - arrays["first"]) / MONTH_DAYS
        power = r + s + x
        mixture = ((r + x) / power * (alpha + T) ** (-power)
                   + s / power * (alpha + tx) ** (-power))
        expected = gammaln(r + x) - gammaln(r) + (r + s) * np.log(alpha) + np.log(mixture)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=False)

    def test_unequal_rates_against_adaptive_quadrature(self):
        model = ParetoNBD()
        params = np.log([0.6, 2.0, 0.4, 9.0])
        arrays = model._arrays(self.rows())
        actual = model.likelihood_parts(params, arrays)[0]
        r, a, s, b = np.exp(params)
        for i, row in self.rows().iterrows():
            x, T = arrays["x"][i], arrays["T"][i]
            tx = (absolute_day([row.last_day])[0] - arrays["first"][i]) / MONTH_DAYS
            alive = (a + T) ** (-r - x) * (b + T) ** (-s)
            dead = quad(lambda t: s * (a + t) ** (-r - x) * (b + t) ** (-s - 1), tx, T, epsabs=1e-14)[0]
            expected = gammaln(r + x) - gammaln(r) + r * np.log(a) + s * np.log(b) + np.log(alive + dead)
            self.assertAlmostEqual(actual[i], expected, places=9)

    def test_expected_count_closed_form_and_s_one_limit(self):
        for shape in [0.4, 1.0, 1.4]:
            model = ParetoNBD()
            model.parameters_ = np.log([0.8, 8.0, shape, 10.0])
            rows = self.rows()
            arrays = model._arrays(rows)
            _, alive, a, b, exposure = model.likelihood_parts(model.parameters_, arrays)
            h = 90 / MONTH_DAYS
            B = b + arrays["T"]
            integrated = B * np.log1p(h / B) if shape == 1 else B * (-np.expm1((1 - shape) * np.log1p(h / B))) / (shape - 1)
            expected = alive * (0.8 + arrays["x"]) / (a + exposure) * integrated
            np.testing.assert_allclose(model.expected_purchases(rows, 90), expected, rtol=1e-10)
            np.testing.assert_array_equal(model.expected_purchases(rows, 0), np.zeros(len(rows)))
            self.assertTrue(np.all(model.expected_purchases(rows, 180) >= expected))

    def test_seasonal_exposure_october_to_january(self):
        start, end = absolute_day(["2010-10-01", "2011-01-01"])
        self.assertEqual(high_season_days(np.array([end]))[0] - high_season_days(np.array([start]))[0], 92)

    def test_seasonal_likelihood_reduces_to_base_at_zero_coefficients(self):
        base, seasonal = ParetoNBD(), ParetoNBD(seasonal=True)
        params = np.log([0.6, 2.0, 0.4, 9.0])
        a = base.likelihood_parts(params, base._arrays(self.rows()))[0]
        b = seasonal.likelihood_parts(np.r_[params, 0, 0, 0], seasonal._arrays(self.rows()))[0]
        np.testing.assert_array_equal(a, b)

    def test_mixture_quantile_atom_and_continuous_cdf(self):
        p = np.array([0.03, 0.4, 0.99])
        q = 0.90
        result = mixture_quantile(p, lambda t: np.exp(norm.ppf(t)), q)
        self.assertEqual(result[0], 0)
        cdf = 1 - p[1:] + p[1:] * norm.cdf(np.log(result[1:]))
        np.testing.assert_allclose(cdf, q, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
