"""Finite-horizon dynamic programming for optimal trade execution.

A trader must sell `X` shares over `N` periods. Selling `n` shares in a period
costs temporary impact `eta * n**2 / tau` (linear price impact) plus permanent
impact `gamma * n * x_remaining`-style drift, and holding inventory exposes the
trader to price risk `lam * sigma**2 * tau * x**2` (mean-variance objective of
Almgren & Chriss, 2000). The optimal schedule is found by backward induction on
an integer inventory grid and compared with the closed-form solution.
"""
from __future__ import annotations
import numpy as np


def solve_execution(X: int, N: int, eta: float, sigma: float, lam: float, tau: float = 1.0):
    """Backward induction. Returns (value table V[t, x], policy table n[t, x]).

    V[t, x] is the minimum expected cost plus risk penalty of liquidating x
    shares in the remaining N − t periods. Everything must be sold by N.
    """
    V = np.full((N + 1, X + 1), np.inf)
    pol = np.zeros((N + 1, X + 1), dtype=int)
    V[N, 0] = 0.0
    xs = np.arange(X + 1)
    for t in range(N - 1, -1, -1):
        for x in xs:
            n = np.arange(x + 1)              # shares sold this period
            rem = x - n
            cost = eta * n ** 2 / tau + lam * sigma ** 2 * tau * rem ** 2 + V[t + 1, rem]
            j = int(np.argmin(cost))
            V[t, x] = cost[j]; pol[t, x] = n[j]
    return V, pol


def schedule(pol: np.ndarray, X: int) -> np.ndarray:
    """Inventory path implied by a policy table, starting from X shares."""
    x = X; path = [x]
    for t in range(pol.shape[0] - 1):
        x -= pol[t, x]; path.append(x)
    return np.array(path)


def almgren_chriss(X: float, N: int, eta: float, sigma: float, lam: float, tau: float = 1.0) -> np.ndarray:
    """Closed-form optimal inventory path x_t = X sinh(κ(T−t)) / sinh(κT) for the discrete model above."""
    if lam == 0:
        return X * (1 - np.arange(N + 1) / N)
    kt = np.arccosh(1 + lam * sigma ** 2 * tau ** 2 / (2 * eta))  # κτ for the discrete-time model
    t = np.arange(N + 1)
    return X * np.sinh(kt * (N - t)) / np.sinh(kt * N)


def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float, tol: float = 1e-10, max_iter: int = 10_000):
    """Infinite-horizon value iteration. P[a, s, s'] transition probabilities, R[s, a] expected rewards."""
    S, A = R.shape
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = R + gamma * np.einsum("ast,t->sa", P, V)
        V_new = Q.max(1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new; break
        V = V_new
    return V, Q.argmax(1)


def policy_iteration(P: np.ndarray, R: np.ndarray, gamma: float):
    """Howard's policy iteration with exact policy evaluation."""
    S, A = R.shape
    pi = np.zeros(S, dtype=int)
    while True:
        Ppi = P[pi, np.arange(S)]
        Rpi = R[np.arange(S), pi]
        V = np.linalg.solve(np.eye(S) - gamma * Ppi, Rpi)
        Q = R + gamma * np.einsum("ast,t->sa", P, V)
        new = Q.argmax(1)
        if np.array_equal(new, pi):
            return V, pi
        pi = new
