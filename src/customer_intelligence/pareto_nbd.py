"""Finite-horizon Pareto/NBD with optional calendar purchase intensity.

Base assumptions and likelihood: Fader & Hardie, note 009 (2005).
The extension uses a deterministic October-December purchase-rate multiplier
and a UK/other country covariate on purchase/dropout gamma rate parameters.
Death hazard is constant conditional on the latent customer rate. Integration
splits at calendar discontinuities; this is a point fit, not Bayesian sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from .bgnbd import MONTH_DAYS, GammaGamma, customer_day_summary


DAY = pd.Timedelta(days=1).value


def absolute_day(values) -> np.ndarray:
    # Pandas can mix microsecond Timestamp columns and nanosecond date arrays.
    return pd.DatetimeIndex(pd.to_datetime(values)).as_unit("ns").asi8.astype(float) / DAY


def high_season_days(days: np.ndarray) -> np.ndarray:
    """Calendar exposure relative to 2008; all supported data is later."""
    result = np.zeros_like(np.asarray(days, dtype=float))
    for year in range(2008, 2015):
        start = pd.Timestamp(year=year, month=10, day=1).value / DAY
        end = pd.Timestamp(year=year + 1, month=1, day=1).value / DAY
        result += np.clip(days - start, 0, end - start)
    return result


def quadrature(first, begin, end, nodes: int = 24):
    """Return time, seasonal exposure and month-unit weights after begin."""
    first, begin, end = np.broadcast_arrays(first, begin, end)
    lo, hi = float(begin.min()), float(end.max())
    boundaries = [lo, hi]
    for year in range(2008, 2015):
        for month in (1, 10):
            date = pd.Timestamp(year=year, month=month, day=1).value / DAY
            if lo < date < hi:
                boundaries.append(date)
    boundaries = sorted(set(boundaries))
    if len(boundaries) == 1:
        boundaries.append(boundaries[0])
    location, weight = leggauss(nodes)
    times, seasons, weights = [], [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        a, b = np.maximum(begin, left), np.minimum(end, right)
        width = np.maximum(b - a, 0)
        position = a[:, None] + width[:, None] * (location + 1) / 2
        times.append((position - first[:, None]) / MONTH_DAYS)
        seasons.append((high_season_days(position) - high_season_days(first)[:, None]) / MONTH_DAYS)
        weights.append(width[:, None] * weight / (2 * MONTH_DAYS))
    return np.concatenate(times, axis=1), np.concatenate(seasons, axis=1), np.concatenate(weights, axis=1)


def history_summary(sales: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    summary = customer_day_summary(sales, cutoff)
    lines = sales.loc[sales.invoice_date < cutoff]
    days = lines[["customer_id", "purchase_day"]].drop_duplicates().sort_values("purchase_day")
    first = days.groupby("customer_id").purchase_day.transform("min")
    repeat_q4 = days.loc[(days.purchase_day > first) & (days.purchase_day.dt.month >= 10)]
    summary["q4_repeat_count"] = repeat_q4.groupby("customer_id").size().reindex(summary.index, fill_value=0)
    country = lines.sort_values("invoice_date", kind="stable").groupby("customer_id").country.last()
    summary["foreign"] = country.reindex(summary.index).ne("United Kingdom").astype(float)
    summary["cutoff"] = cutoff
    return summary


class ParetoNBD:
    def __init__(self, seasonal: bool = False, quadrature_nodes: int = 24, penalizer: float = 0.002):
        self.seasonal = seasonal
        self.quadrature_nodes = quadrature_nodes
        self.penalizer = penalizer

    def _arrays(self, rows: pd.DataFrame) -> dict:
        first = absolute_day(rows.first_day)
        last = absolute_day(rows.last_day)
        end = absolute_day(rows.cutoff)
        time, season, weights = quadrature(first, last, end, self.quadrature_nodes)
        log_weights = np.full_like(weights, -np.inf)
        np.log(weights, out=log_weights, where=weights > 0)
        return dict(x=rows.frequency.to_numpy(dtype=float), q4=rows.q4_repeat_count.to_numpy(dtype=float),
                    z=rows.foreign.to_numpy(dtype=float), first=first, end=end,
                    T=(end - first) / MONTH_DAYS,
                    H=(high_season_days(end) - high_season_days(first)) / MONTH_DAYS,
                    t=time, h=season, log_w=log_weights)

    def likelihood_parts(self, parameters, arrays):
        r, alpha, s, beta = np.exp(parameters[:4])
        if self.seasonal:
            calendar, purchase_cov, dropout_cov = parameters[4:]
        else:
            calendar = purchase_cov = dropout_cov = 0.0
        a = alpha * np.exp(-purchase_cov * arrays["z"])
        b = beta * np.exp(-dropout_cov * arrays["z"])
        scale = np.expm1(calendar)
        exposure_T = arrays["T"] + scale * arrays["H"]
        exposure_t = arrays["t"] + scale * arrays["h"]
        x = arrays["x"]
        alive = -(r + x) * np.log(a + exposure_T) - s * np.log(b + arrays["T"])
        death_integrand = (np.log(s) - (r + x)[:, None] * np.log(a[:, None] + exposure_t)
                          - (s + 1) * np.log(b[:, None] + arrays["t"]) + arrays["log_w"])
        death = logsumexp(death_integrand, axis=1)
        mixture = np.logaddexp(alive, death)
        common = gammaln(r + x) - gammaln(r) + r * np.log(a) + s * np.log(b) + calendar * arrays["q4"]
        return common + mixture, np.exp(alive - mixture), a, b, exposure_T

    def fit(self, rows: pd.DataFrame):
        if len(rows) < 20:
            raise ValueError("Pareto/NBD needs at least 20 customer histories.")
        arrays = self._arrays(rows)
        if np.any(arrays["T"] <= 0):
            raise ValueError("Every history must have positive observed age.")

        def objective(parameters):
            likelihood = self.likelihood_parts(parameters, arrays)[0]
            # Weak Gaussian penalties on log parameters and covariate effects.
            penalty = self.penalizer * np.sum(parameters[:4] ** 2)
            if self.seasonal:
                penalty += 0.01 * np.sum(parameters[4:] ** 2)
            return -float(likelihood.mean()) + penalty

        starts = [np.log([0.7, 8.0, 0.3, 8.0]), np.log([1.0, 15.0, 0.8, 20.0])]
        if self.seasonal:
            starts = [np.r_[p, 0.0, 0.0, 0.0] for p in starts]
        bounds = [(-5, 5), (-5, 7), (-5, 5), (-5, 7)] + ([(-2, 2)] * 3 if self.seasonal else [])
        results = [minimize(objective, p, method="L-BFGS-B", bounds=bounds,
                            options=dict(maxiter=250, ftol=1e-8)) for p in starts]
        valid = [r for r in results if r.success and np.isfinite(r.fun) and np.isfinite(r.x).all()]
        if not valid:
            raise RuntimeError("Pareto/NBD optimizer did not converge: " + "; ".join(str(r.message) for r in results))
        best = min(valid, key=lambda r: r.fun)
        self.parameters_ = best.x
        self.fit_info_ = dict(optimizer_success=True, objective=float(best.fun),
            r=float(np.exp(best.x[0])), alpha=float(np.exp(best.x[1])), s=float(np.exp(best.x[2])), beta=float(np.exp(best.x[3])),
            q4_purchase_multiplier=float(np.exp(best.x[4])) if self.seasonal else 1.0,
            purchase_foreign_multiplier=float(np.exp(best.x[5])) if self.seasonal else 1.0,
            dropout_foreign_multiplier=float(np.exp(best.x[6])) if self.seasonal else 1.0,
            at_parameter_boundary=bool(any(abs(v - lo) < 1e-4 or abs(v - hi) < 1e-4 for v, (lo, hi) in zip(best.x, bounds))))
        return self

    def expected_purchases(self, rows: pd.DataFrame, horizon_days: int) -> np.ndarray:
        if horizon_days < 0:
            raise ValueError("Horizon cannot be negative.")
        arrays = self._arrays(rows)
        _, alive, a, b, exposure = self.likelihood_parts(self.parameters_, arrays)
        r, _, s, _ = np.exp(self.parameters_[:4])
        calendar = self.parameters_[4] if self.seasonal else 0.0
        t, high, weights = quadrature(arrays["end"], arrays["end"], arrays["end"] + horizon_days, self.quadrature_nodes)
        # high' = 1 in Q4; evaluate the known calendar at each quadrature node.
        day_position = arrays["end"][:, None] + t * MONTH_DAYS
        is_q4 = high_season_days(day_position + 0.001) - high_season_days(day_position) > 0.0005
        purchase_intensity = np.exp(calendar * is_q4)
        survival = ((b + arrays["T"])[:, None] / ((b + arrays["T"])[:, None] + t)) ** s
        future_exposure = np.sum(weights * purchase_intensity * survival, axis=1)
        result = alive * (r + arrays["x"]) / (a + exposure) * future_exposure
        if not np.isfinite(result).all() or np.any(result < 0):
            raise RuntimeError("Invalid finite-horizon purchase expectation.")
        return result


class ParetoRevenue:
    def __init__(self, seasonal: bool = False):
        self.seasonal = seasonal

    def fit(self, sales: pd.DataFrame, cutoff: pd.Timestamp):
        self.summary_ = history_summary(sales, cutoff)
        self.purchase_ = ParetoNBD(seasonal=self.seasonal).fit(self.summary_)
        repeat = (self.summary_.frequency > 0) & self.summary_.repeat_mean_daily_spend.notna()
        self.spend_ = GammaGamma().fit(self.summary_.loc[repeat, "frequency"], self.summary_.loc[repeat, "repeat_mean_daily_spend"])
        if not self.spend_.finite_population_mean_:
            raise RuntimeError("Gamma-Gamma fitted population mean is not finite.")
        self.fit_info_ = dict(**self.purchase_.fit_info_, gamma_q=self.spend_.q_)
        return self

    def predict(self, x: pd.DataFrame, horizon_days: int):
        summary = self.summary_.reindex(x.customer_id)
        if summary.first_day.isna().any():
            raise ValueError("Customer history is missing from the fitted forecast origin.")
        count = self.purchase_.expected_purchases(summary, horizon_days)
        value = self.spend_.expected_average_spend(summary.frequency, summary.repeat_mean_daily_spend.fillna(0))
        return count * value
