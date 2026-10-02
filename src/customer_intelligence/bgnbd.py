"""Penalized likelihood BG/NBD and Gamma-Gamma reference implementation.

The likelihoods and conditional expectation formulae follow the public
``lifetimes`` implementation of the published BG/NBD and Gamma-Gamma models.
This small local version estimates point parameters only; it does not claim
posterior or parameter uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

MONTH_DAYS = 30.436875


class BetaGeoNBD:
    def __init__(self, penalizer: float = 1e-3):
        self.penalizer = penalizer

    @staticmethod
    def _unpack(log_params):
        return np.exp(np.clip(log_params, -20, 20))

    @staticmethod
    def _negative_log_likelihood(log_params, frequency, recency, age, penalizer):
        r, alpha, a, b = np.exp(np.clip(log_params, -20, 20))
        x = frequency
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a3 = -(r + x) * np.log(alpha + age)
        a4 = np.log(a) - np.log(b + np.maximum(x, 1) - 1) - (r + x) * np.log(recency + alpha)
        mixture = np.logaddexp(a3, a4 + np.where(x > 0, 0.0, -np.inf))
        return float(-np.mean(a1 + a2 + mixture) + penalizer * np.sum(np.exp(2 * np.clip(log_params, -20, 20))))

    def fit(self, frequency, recency, age) -> "BetaGeoNBD":
        x = np.asarray(frequency, dtype=float)
        tx = np.asarray(recency, dtype=float)
        T = np.asarray(age, dtype=float)
        valid = np.isfinite(x) & np.isfinite(tx) & np.isfinite(T) & (x >= 0) & (tx >= 0) & (T > 0) & (tx <= T)
        if valid.sum() < 10:
            raise ValueError("BG/NBD needs at least ten valid customer histories.")
        x, tx, T = x[valid], tx[valid], T[valid]
        starts = [np.log([0.8, 8.0, 1.4, 3.0]), np.log([1.2, 15.0, 2.0, 6.0]), np.log([0.5, 4.0, 1.15, 2.0])]
        bounds = [(-7, 5), (-6, 7), (-7, 7), (-7, 7)]
        fits = [minimize(self._negative_log_likelihood, start, args=(x, tx, T, self.penalizer), method="L-BFGS-B", bounds=bounds, options={"maxiter": 1200, "ftol": 1e-10}) for start in starts]
        result = min(fits, key=lambda item: item.fun if np.isfinite(item.fun) else np.inf)
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError("BG/NBD likelihood optimization did not produce finite parameters.")
        self.r_, self.alpha_, self.a_, self.b_ = np.exp(result.x)
        self.finite_lifetime_expectation_ = bool(self.a_ > 1.0)
        self.negative_log_likelihood_ = float(result.fun)
        self.optimizer_success_ = bool(result.success)
        self.optimizer_message_ = str(result.message)
        self.n_customers_ = int(valid.sum())
        return self

    def expected_purchases(self, horizon, frequency, recency, age) -> np.ndarray:
        # This closed form is for a finite horizon.  a <= 1 rules out a finite
        # infinite-horizon expectation, but does not make a finite-horizon
        # purchase forecast undefined.
        r, alpha, a, b = self.r_, self.alpha_, self.a_, self.b_
        x = np.asarray(frequency, dtype=float)
        tx = np.asarray(recency, dtype=float)
        T = np.asarray(age, dtype=float)
        t = np.broadcast_to(np.asarray(horizon, dtype=float), x.shape)
        aa, bb, cc = r + x, b + x, a + b + x - 1
        z = t / (alpha + T + t)
        hyper = hyp2f1(aa, bb, cc, z)
        log_hyper = np.log(np.maximum(hyper, 1e-300))
        second = 1 - np.exp(np.clip(
            log_hyper + (r + x) * np.log((alpha + T) / (alpha + t + T)), -700, 50,
        ))
        denominator = 1 + (x > 0) * (a / np.maximum(b + x - 1, 1e-12)) * np.exp(np.clip(
            (r + x) * np.log((alpha + T) / (alpha + tx)), -100, 100,
        ))
        output = ((a + b + x - 1) / (a - 1)) * second / denominator
        return np.maximum(np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0), 0)

    def probability_alive(self, frequency, recency, age) -> np.ndarray:
        r, alpha, a, b = self.r_, self.alpha_, self.a_, self.b_
        x, tx, T = map(lambda value: np.asarray(value, dtype=float), (frequency, recency, age))
        log_ratio = (r + x) * np.log((alpha + T) / (alpha + tx)) + np.log(a / (b + np.maximum(x, 1) - 1))
        alive = np.where(x == 0, 1.0, 1.0 / (1.0 + np.exp(np.clip(log_ratio, -80, 80))))
        return np.clip(alive, 0, 1)


class GammaGamma:
    def __init__(self, penalizer: float = 1e-4):
        self.penalizer = penalizer

    @staticmethod
    def _negative_log_likelihood(log_params, frequency, amount, penalizer):
        p = np.exp(np.clip(log_params[0], -15, 15))
        q = np.exp(np.clip(log_params[1], -15, 15))
        v = np.exp(np.clip(log_params[2], -15, 15))
        x, m = frequency, amount
        values = (
            gammaln(p * x + q) - gammaln(p * x) - gammaln(q)
            + q * np.log(v) + (p * x - 1) * np.log(m)
            + (p * x) * np.log(x) - (p * x + q) * np.log(x * m + v)
        )
        return float(-np.mean(values) + penalizer * (p * p + q * q + v * v))

    def fit(self, frequency, mean_spend) -> "GammaGamma":
        x = np.asarray(frequency, dtype=float)
        m = np.asarray(mean_spend, dtype=float)
        valid = np.isfinite(x) & np.isfinite(m) & (x > 0) & (m > 0)
        if valid.sum() < 20:
            raise ValueError("Gamma-Gamma needs at least twenty repeat-buyer histories.")
        # Scaling keeps the monetary parameters in a numerically useful range.
        self.money_scale_ = 100.0
        x, m = x[valid], m[valid] / self.money_scale_
        self.training_mean_spend_ = float(np.mean(m) * self.money_scale_)
        starts = [np.log([2.0, 2.0, 2.0]), np.log([4.0, 4.0, 8.0]), np.log([1.0, 0.8, 4.0])]
        bounds = [(-7, 7), (-7, 7), (-8, 8)]
        fits = [minimize(self._negative_log_likelihood, start, args=(x, m, self.penalizer), method="L-BFGS-B", bounds=bounds, options={"maxiter": 1200, "ftol": 1e-10}) for start in starts]
        result = min(fits, key=lambda item: item.fun if np.isfinite(item.fun) else np.inf)
        p = np.exp(result.x[0])
        q = np.exp(result.x[1])
        v = np.exp(result.x[2])
        if not np.isfinite(result.fun):
            raise RuntimeError("Gamma-Gamma fit did not yield finite parameters.")
        self.p_, self.q_, self.v_ = float(p), float(q), float(v * self.money_scale_)
        self.negative_log_likelihood_ = float(result.fun)
        self.optimizer_success_ = bool(result.success)
        self.optimizer_message_ = str(result.message)
        self.finite_population_mean_ = bool(self.q_ > 1.0)
        self.population_mean_ = self.v_ * self.p_ / (self.q_ - 1) if self.finite_population_mean_ else np.inf
        self.n_customers_ = int(valid.sum())
        return self

    def expected_average_spend(self, frequency, observed_mean_spend) -> np.ndarray:
        x = np.maximum(np.asarray(frequency, dtype=float), 0)
        m = np.maximum(np.asarray(observed_mean_spend, dtype=float), 0)
        denominator = self.p_ * x + self.q_ - 1
        output = np.full(np.broadcast_shapes(x.shape, m.shape), self.training_mean_spend_, dtype=float)
        if self.finite_population_mean_:
            output[...] = self.population_mean_
        repeat_buyer = x > 0
        valid_repeat = repeat_buyer & (denominator > 0)
        if np.any(repeat_buyer & ~valid_repeat):
            raise ValueError("Gamma-Gamma conditional spend is not finite for one or more repeat buyers.")
        # Algebraically equivalent to the standard weighted-mean expression,
        # while remaining usable for repeat buyers when q <= 1.  In that case
        # one-time buyers use the observed repeat-buyer mean as a finite prior
        # proxy; this fallback is explicitly recorded in the comparison.
        output[valid_repeat] = (
            self.p_ * (self.v_ + x[valid_repeat] * m[valid_repeat])
            / denominator[valid_repeat]
        )
        return np.maximum(np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0), 0)


def customer_day_summary(lines: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    history = lines.loc[lines["invoice_date"].lt(cutoff)]
    daily = history.groupby(["customer_id", "purchase_day"], sort=False).agg(
        daily_spend=("revenue", "sum"),
    ).reset_index()
    daily = daily.sort_values("purchase_day", kind="stable")
    grouped = daily.groupby("customer_id", sort=False)
    summary = grouped.agg(first_day=("purchase_day", "min"), last_day=("purchase_day", "max"), occasions=("purchase_day", "nunique"))
    summary["frequency"] = (summary["occasions"] - 1).clip(lower=0).astype(int)
    summary["recency_months"] = (summary["last_day"] - summary["first_day"]).dt.days / MONTH_DAYS
    summary["age_months"] = (cutoff.normalize() - summary["first_day"]).dt.days / MONTH_DAYS
    first_days = grouped["purchase_day"].min()
    daily_with_repeat = daily.join(first_days.rename("first_day"), on="customer_id")
    repeat = daily_with_repeat.loc[daily_with_repeat["purchase_day"].gt(daily_with_repeat["first_day"])]
    repeat_summary = repeat.groupby("customer_id", sort=False).agg(
        repeat_day_count=("purchase_day", "nunique"),
        repeat_mean_daily_spend=("daily_spend", "mean"),
    )
    return summary.join(repeat_summary)

