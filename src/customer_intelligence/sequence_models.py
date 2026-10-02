"""Compact neural challengers over fixed weekly customer histories."""

from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn

from .config import SEED
from .features import NUMERIC_FEATURES


class CustomerSequenceNet(nn.Module):
    def __init__(self, name: str, static_size: int, week_size: int = 6):
        super().__init__()
        self.name = name
        if name == "lstm_weekly":
            self.recurrent = nn.LSTM(week_size, 16, num_layers=1, batch_first=True)
            seq_size = 16
        elif name == "gru_weekly":
            self.recurrent = nn.GRU(week_size, 16, num_layers=1, batch_first=True)
            seq_size = 16
        else:
            self.recurrent = None
            seq_size = week_size * 26
        hidden = 48 if name == "mlp_weekly" else 24
        self.head = nn.Sequential(
            nn.Linear(seq_size + static_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, weeks, static):
        if self.recurrent is None:
            temporal = weeks.flatten(start_dim=1)
        else:
            encoded, _ = self.recurrent(weeks)
            temporal = encoded[:, -1, :]
        return self.head(torch.cat([temporal, static], dim=1)).squeeze(-1)


class SequencePredictor:
    def __init__(self, name: str, model, week_mean, week_scale, static_imputer, static_scaler):
        self.name = name
        self.model = model
        self.week_mean = week_mean
        self.week_scale = week_scale
        self.static_imputer = static_imputer
        self.static_scaler = static_scaler

    def transform(self, features: pd.DataFrame, weeks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weekly = np.log1p(np.maximum(np.asarray(weeks, dtype=np.float32), 0))
        weekly = (weekly - self.week_mean) / self.week_scale
        static = self.static_scaler.transform(self.static_imputer.transform(features[NUMERIC_FEATURES]))
        return weekly.astype(np.float32), static.astype(np.float32)

    def predict_proba(self, features: pd.DataFrame, weeks: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        weekly, static = self.transform(features, weeks)
        self.model.eval()
        output = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                w = torch.as_tensor(weekly[start:start + batch_size])
                s = torch.as_tensor(static[start:start + batch_size])
                output.append(torch.sigmoid(self.model(w, s)).cpu().numpy())
        p = np.concatenate(output) if output else np.array([], dtype=float)
        return np.column_stack([1 - p, p])


def fit_sequence_model(
    name: str,
    train_features: pd.DataFrame,
    train_weeks: np.ndarray,
    train_y,
    validation_features: pd.DataFrame,
    validation_weeks: np.ndarray,
    validation_y,
    seed: int = SEED,
    fixed_epochs: int | None = None,
) -> tuple[SequencePredictor, np.ndarray, int]:
    """Fit on training snapshots; use development data only for early stopping."""
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_week_raw = np.log1p(np.maximum(np.asarray(train_weeks, dtype=np.float32), 0))
    week_mean = train_week_raw.reshape(-1, train_week_raw.shape[-1]).mean(axis=0).reshape(1, 1, -1)
    week_scale = train_week_raw.reshape(-1, train_week_raw.shape[-1]).std(axis=0).reshape(1, 1, -1)
    week_scale[week_scale < 1e-5] = 1.0
    train_week = ((train_week_raw - week_mean) / week_scale).astype(np.float32)
    validation_week_raw = np.log1p(np.maximum(np.asarray(validation_weeks, dtype=np.float32), 0))
    validation_week = ((validation_week_raw - week_mean) / week_scale).astype(np.float32)

    static_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    static_scaler = StandardScaler()
    train_static = static_scaler.fit_transform(static_imputer.fit_transform(train_features[NUMERIC_FEATURES])).astype(np.float32)
    validation_static = static_scaler.transform(static_imputer.transform(validation_features[NUMERIC_FEATURES])).astype(np.float32)

    model = CustomerSequenceNet(name, train_static.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.002)
    loss_function = nn.BCEWithLogitsLoss()
    tw = torch.as_tensor(train_week)
    ts = torch.as_tensor(train_static)
    ty = torch.as_tensor(np.asarray(train_y, dtype=np.float32))
    vw = torch.as_tensor(validation_week)
    vs = torch.as_tensor(validation_static)
    vy = np.asarray(validation_y, dtype=int)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = fixed_epochs or 1
    max_epochs = fixed_epochs or 35
    patience = 6
    waited = 0
    batch_size = 512
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(ty))
        for start in range(0, len(ty), batch_size):
            batch = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(tw[batch], ts[batch]), ty[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(vw, vs)
            val_loss = float(loss_function(val_logits, torch.as_tensor(vy, dtype=torch.float32)).item())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            waited = 0
        else:
            waited += 1
            if waited >= patience and fixed_epochs is None:
                break
    model.load_state_dict(best_state)
    predictor = SequencePredictor(name, model, week_mean, week_scale, static_imputer, static_scaler)
    validation_probability = predictor.predict_proba(validation_features, validation_weeks)[:, 1]
    return predictor, validation_probability, best_epoch
