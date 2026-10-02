"""Additive and distributional models of finite-horizon customer revenue."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, LogisticRegression, TweedieRegressor
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from torch import nn

from .config import SEED
from .features import NUMERIC_FEATURES
from .revenue_research import CALENDAR, EXTRA, hist_parameters, transform


VALUE_FEATURES = list(NUMERIC_FEATURES) + CALENDAR + EXTRA
GAM_FEATURES = ["recency_days", "tenure_days", "purchase_count", "avg_order_value_gbp",
                "spend_90d_gbp", "spend_365d_gbp", "last_order_value_gbp", "order_value_std_gbp"]


class AdditiveFeatures:
    """Independent logged feature splines with constant extrapolation."""

    def fit_transform(self, x):
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        log_values = np.log1p(self.imputer.fit_transform(x[GAM_FEATURES]))
        self.variable = np.flatnonzero(log_values.std(axis=0) > 1e-10)
        self.splines = SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="constant")
        self.country = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20)
        self.scaler = StandardScaler()
        z = np.column_stack([self.splines.fit_transform(log_values[:, self.variable]),
                             x[CALENDAR].to_numpy(), x[GAM_FEATURES].isna().to_numpy(),
                             self.country.fit_transform(x[["country"]])])
        return self.scaler.fit_transform(z)

    def transform(self, x):
        log_values = np.log1p(self.imputer.transform(x[GAM_FEATURES]))
        z = np.column_stack([self.splines.transform(log_values[:, self.variable]),
                             x[CALENDAR].to_numpy(), x[GAM_FEATURES].isna().to_numpy(),
                             self.country.transform(x[["country"]])])
        return self.scaler.transform(z)


class RevenueGAM:
    def __init__(self, hurdle: bool = False):
        self.hurdle = hurdle

    def fit(self, x, y):
        self.features = AdditiveFeatures()
        z = self.features.fit_transform(x)
        positive = y > 0
        if self.hurdle:
            self.propensity = LogisticRegression(C=0.3, max_iter=1500, random_state=SEED).fit(z, positive)
            self.spend = GammaRegressor(alpha=0.3, max_iter=1500).fit(z[positive], y[positive])
        else:
            self.spend = TweedieRegressor(power=1.5, alpha=0.3, max_iter=1500).fit(z, y)
        return self

    def predict(self, x):
        z = self.features.transform(x)
        value = self.spend.predict(z)
        return value * self.propensity.predict_proba(z)[:, 1] if self.hurdle else value


def mixture_quantile(purchase_probability, positive_quantile, quantile):
    """An atom at zero plus a continuous strictly positive distribution."""
    p = np.asarray(purchase_probability)
    conditional_quantile = np.clip((quantile - (1 - p)) / np.maximum(p, 1e-12), 1e-8, 1 - 1e-8)
    return np.where(quantile <= 1 - p, 0, positive_quantile(conditional_quantile))


class HurdleNGBoost:
    money_scale = 1000.0

    def fit(self, x, y):
        from ngboost import NGBRegressor
        from ngboost.distns import LogNormal

        self.features = transform(VALUE_FEATURES)
        z = self.features.fit_transform(x)
        positive = y > 0
        self.propensity = HistGradientBoostingClassifier(**hist_parameters()).fit(z, positive)
        self.spend = NGBRegressor(Dist=LogNormal, n_estimators=250, learning_rate=0.03,
            Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=25, random_state=SEED),
            natural_gradient=True, verbose=False, random_state=SEED)
        self.spend.fit(z[positive], y[positive] / self.money_scale)
        return self

    def components(self, x):
        z = self.features.transform(x)
        return self.propensity.predict_proba(z)[:, 1], self.spend.pred_dist(z)

    def predict(self, x):
        p, distribution = self.components(x)
        # The distribution's arithmetic mean, not exp(mean(log(revenue))).
        return p * distribution.mean() * self.money_scale

    def interval(self, x, level=0.90):
        p, distribution = self.components(x)
        lo = (1 - level) / 2
        quantile = lambda q: distribution.ppf(q) * self.money_scale
        return mixture_quantile(p, quantile, lo), mixture_quantile(p, quantile, 1 - lo)


class RevenueLSTMNet(nn.Module):
    def __init__(self, static_size, hidden=24):
        super().__init__()
        self.recurrent = nn.LSTM(6, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + static_size, 32), nn.ReLU(),
                                  nn.Dropout(0.15), nn.Linear(32, 3))

    def forward(self, weeks, static):
        encoded, _ = self.recurrent(weeks)
        output = self.head(torch.cat([encoded[:, -1], static], dim=1))
        return output[:, 0], output[:, 1].clamp(-12, 8), output[:, 2].clamp(-2, 0.8)


class HurdleRevenueLSTM:
    money_scale = 1000.0

    def __init__(self, epochs=30, seed=SEED):
        self.epochs, self.seed = epochs, seed

    def fit(self, x: pd.DataFrame, y, weeks: np.ndarray):
        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.features = transform(VALUE_FEATURES)
        static = self.features.fit_transform(x).astype(np.float32)
        weekly_log = np.log1p(np.maximum(weeks, 0)).astype(np.float32)
        self.week_mean = weekly_log.reshape(-1, 6).mean(axis=0)
        self.week_scale = np.maximum(weekly_log.reshape(-1, 6).std(axis=0), 1e-4)
        weekly = (weekly_log - self.week_mean) / self.week_scale
        self.net = RevenueLSTMNet(static.shape[1])
        positive = np.asarray(y) > 0
        log_amount = np.log(np.maximum(y, 1e-6) / self.money_scale)
        last = self.net.head[-1]
        with torch.no_grad():
            last.bias[0] = float(np.log(positive.mean() / (1 - positive.mean())))
            last.bias[1] = float(log_amount[positive].mean())
            last.bias[2] = float(np.log(max(log_amount[positive].std(), 0.5)))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.002, weight_decay=0.003)
        w, s = torch.as_tensor(weekly), torch.as_tensor(static)
        returned = torch.as_tensor(positive.astype(np.float32))
        amount = torch.as_tensor(log_amount.astype(np.float32))
        self.training_loss_ = []
        for epoch in range(self.epochs):
            self.net.train()
            permutation = torch.randperm(len(y))
            total_loss = 0.0
            for start in range(0, len(y), 512):
                batch = permutation[start:start + 512]
                optimizer.zero_grad(set_to_none=True)
                logits, mean, log_std = self.net(w[batch], s[batch])
                purchase_loss = nn.functional.binary_cross_entropy_with_logits(logits, returned[batch], reduction="none")
                spend_loss = log_std + 0.5 * ((amount[batch] - mean) * torch.exp(-log_std)) ** 2
                loss = (purchase_loss + returned[batch] * spend_loss).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite LSTM hurdle likelihood.")
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch)
            self.training_loss_.append(total_loss / len(y))
        self.net.eval()
        return self

    def components(self, x, weeks):
        static = self.features.transform(x).astype(np.float32)
        weekly = ((np.log1p(np.maximum(weeks, 0)) - self.week_mean) / self.week_scale).astype(np.float32)
        output = []
        self.net.eval()
        with torch.no_grad():
            for start in range(0, len(x), 1024):
                p, mean, log_std = self.net(torch.as_tensor(weekly[start:start + 1024]), torch.as_tensor(static[start:start + 1024]))
                output.append(np.column_stack([torch.sigmoid(p).numpy(), mean.numpy(), torch.exp(log_std).numpy()]))
        result = np.concatenate(output)
        return result[:, 0], result[:, 1], result[:, 2]

    def predict(self, x, weeks):
        p, mean, std = self.components(x, weeks)
        return p * np.exp(mean + std ** 2 / 2) * self.money_scale

    def interval(self, x, weeks, level=0.90):
        p, mean, std = self.components(x, weeks)
        quantile = lambda q: np.exp(mean + std * norm.ppf(q)) * self.money_scale
        lo = (1 - level) / 2
        return mixture_quantile(p, quantile, lo), mixture_quantile(p, quantile, 1 - lo)
