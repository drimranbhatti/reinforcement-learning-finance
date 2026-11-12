"""A small, transparent Q-learning trading agent.

State: (trend bucket, volatility bucket, current position). Trend is the
trailing `lookback`-day return split into terciles, volatility the trailing
standard deviation split into terciles; bucket edges come from the training
period only. Actions: target position in {0 = cash, 1 = long}. Reward: next-day
log return of the position, minus a cost on each change of position.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def features(prices: pd.Series, lookback: int = 20) -> pd.DataFrame:
    r = np.log(prices).diff()
    f = pd.DataFrame({"ret": r, "trend": np.log(prices).diff(lookback), "vol": r.rolling(lookback).std()})
    f["next_ret"] = f["ret"].shift(-1)
    return f.dropna()


def bucketize(train: pd.DataFrame, data: pd.DataFrame, n: int = 3) -> np.ndarray:
    """Map trend and volatility to n quantile buckets using edges fitted on `train`; returns a state id per row."""
    q = np.linspace(0, 1, n + 1)[1:-1]
    te, ve = train["trend"].quantile(q).to_numpy(), train["vol"].quantile(q).to_numpy()
    return np.searchsorted(te, data["trend"].to_numpy()) * n + np.searchsorted(ve, data["vol"].to_numpy())


def train_q(states: np.ndarray, next_ret: np.ndarray, n_market: int, episodes: int = 200, alpha: float = 0.0,
            gamma: float = 0.95, eps: float = 0.1, cost: float = 0.0005, seed: int = 0) -> np.ndarray:
    """Q-learning over repeated passes through the training history. Q[market_state, position, action].

    The step size is 1/(number of visits) for each state-action pair (a running
    average, which suits rewards that are mostly noise), floored at `alpha`.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_market, 2, 2)); N = np.zeros((n_market, 2, 2))
    T = len(states)
    for _ in range(episodes):
        pos = 0
        for t in range(T - 1):
            s = states[t]
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[s, pos]))
            r = a * next_ret[t] - cost * abs(a - pos)
            s2 = states[t + 1]
            N[s, pos, a] += 1
            step = max(alpha, 1.0 / N[s, pos, a])
            Q[s, pos, a] += step * (r + gamma * Q[s2, a].max() - Q[s, pos, a])
            pos = a
    return Q


def run_policy(Q: np.ndarray, states: np.ndarray, next_ret: np.ndarray, cost: float = 0.0005):
    """Apply the greedy policy; returns daily strategy log returns and positions."""
    pos = 0; rets = np.zeros(len(states)); positions = np.zeros(len(states), dtype=int)
    for t, s in enumerate(states):
        a = int(np.argmax(Q[s, pos]))
        rets[t] = a * next_ret[t] - cost * abs(a - pos)
        positions[t] = a; pos = a
    return rets, positions


def stats(log_rets: np.ndarray, periods: int = 252) -> dict:
    r = np.asarray(log_rets)
    ann = r.mean() * periods; vol = r.std(ddof=1) * np.sqrt(periods)
    wealth = np.exp(np.cumsum(r)); dd = (wealth / np.maximum.accumulate(wealth) - 1).min()
    return {"ann. return %": 100 * (np.exp(ann) - 1), "ann. vol %": 100 * vol,
            "Sharpe": ann / vol if vol > 0 else 0.0, "max drawdown %": 100 * dd}
