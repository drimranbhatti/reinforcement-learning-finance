"""Multi-armed bandit policies and a sequential asset-selection backtest.

Each policy picks one arm per step from the rewards it has seen so far; nothing
from the future is used. Rewards here are asset returns over the next period.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class Policy:
    name = "policy"

    def __init__(self, k: int, rng: np.random.Generator | None = None):
        self.k = k
        self.rng = rng or np.random.default_rng(0)
        self.n = np.zeros(k)
        self.q = np.zeros(k)

    def select(self, t: int) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]

    def _argmax(self, v):
        best = np.flatnonzero(v == v.max())
        return int(self.rng.choice(best))


class EpsilonGreedy(Policy):
    def __init__(self, k, eps=0.1, rng=None):
        super().__init__(k, rng); self.eps = eps; self.name = f"ε-greedy (ε={eps})"

    def select(self, t):
        if t < self.k:
            return t
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.k))
        return self._argmax(self.q)


class UCB1(Policy):
    """Upper confidence bound (Auer et al., 2002). `c` scales the exploration bonus to the reward units."""

    def __init__(self, k, c=1.0, rng=None):
        super().__init__(k, rng); self.c = c; self.name = "UCB1"

    def select(self, t):
        if t < self.k:
            return t
        bonus = self.c * np.sqrt(2 * np.log(t + 1) / self.n)
        return self._argmax(self.q + bonus)


class DiscountedUCB(Policy):
    """UCB with exponentially discounted statistics, for non-stationary rewards (Garivier & Moulines, 2011)."""

    def __init__(self, k, gamma=0.98, c=1.0, rng=None):
        super().__init__(k, rng); self.gamma = gamma; self.c = c
        self.s = np.zeros(k); self.name = f"discounted UCB (γ={gamma})"

    def select(self, t):
        if t < self.k:
            return t
        n = np.maximum(self.n, 1e-12)
        mean = self.s / n
        bonus = self.c * np.sqrt(2 * np.log(max(self.n.sum(), 2)) / n)
        return self._argmax(mean + bonus)

    def update(self, arm, reward):
        self.n *= self.gamma; self.s *= self.gamma
        self.n[arm] += 1; self.s[arm] += reward
        self.q = self.s / np.maximum(self.n, 1e-12)


class GaussianThompson(Policy):
    """Thompson sampling with a Gaussian posterior on each arm's mean (known noise scale `sigma`)."""

    def __init__(self, k, sigma=1.0, prior_sd=1.0, rng=None):
        super().__init__(k, rng); self.sigma = sigma; self.prior_sd = prior_sd; self.name = "Thompson sampling"

    def select(self, t):
        prec = 1 / self.prior_sd ** 2 + self.n / self.sigma ** 2
        mean = (self.n * self.q / self.sigma ** 2) / prec
        return self._argmax(self.rng.normal(mean, 1 / np.sqrt(prec)))


def simulate_stationary(policy_factory, means, sigma=1.0, horizon=2000, runs=200, seed=0):
    """Average reward and cumulative pseudo-regret of a policy on Gaussian arms."""
    means = np.asarray(means, float)
    rng = np.random.default_rng(seed)
    regret = np.zeros(horizon); optimal = np.zeros(horizon)
    for r in range(runs):
        pol = policy_factory(len(means), np.random.default_rng(rng.integers(1 << 31)))
        for t in range(horizon):
            a = pol.select(t)
            pol.update(a, rng.normal(means[a], sigma))
            regret[t] += means.max() - means[a]
            optimal[t] += a == means.argmax()
    return np.cumsum(regret / runs), optimal / runs


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance weights (inverse-covariance solution, negatives clipped and renormalised)."""
    inv = np.linalg.pinv(cov)
    w = inv @ np.ones(len(cov))
    w = np.clip(w / w.sum(), 0, None)
    return w / w.sum()


def backtest(returns: pd.DataFrame, policy: Policy, blend: float = 1.0, lookback: int = 52,
             cost: float = 0.0) -> pd.DataFrame:
    """Sequential selection on a returns panel (rows = periods, columns = assets).

    At each period the policy picks an asset using only past rewards. With
    `blend` < 1 the holding is blend × the picked asset + (1 − blend) × the
    trailing minimum-variance portfolio (a simplified risk-aware bandit in the
    spirit of Huo & Fu, 2017). `cost` is charged on turnover.
    Returns a frame with the chosen arm, portfolio return and turnover.
    """
    R = returns.to_numpy(float)
    T, k = R.shape
    w_prev = np.zeros(k)
    out = []
    for t in range(T):
        arm = policy.select(t)
        w = np.zeros(k); w[arm] = 1.0
        if blend < 1.0:
            hist = R[max(0, t - lookback):t]
            mv = min_variance_weights(np.cov(hist.T)) if len(hist) > k else np.full(k, 1 / k)
            w = blend * w + (1 - blend) * mv
        turnover = np.abs(w - w_prev).sum()
        ret = float(w @ R[t]) - cost * turnover
        policy.update(arm, R[t, arm])
        out.append((returns.index[t], returns.columns[arm], ret, turnover))
        w_prev = w * (1 + R[t]) / (1 + w @ R[t])
    return pd.DataFrame(out, columns=["date", "arm", "ret", "turnover"]).set_index("date")
