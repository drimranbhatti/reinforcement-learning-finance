"""Tabular temporal-difference control: Q-learning and SARSA, plus the cliff-walking test problem (Sutton & Barto, Example 6.6)."""
from __future__ import annotations
import numpy as np


class CliffWalk:
    """4 × 12 grid. Start bottom-left, goal bottom-right, the cells between are a cliff (reward −100, back to start)."""
    rows, cols = 4, 12
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right

    def __init__(self):
        self.start = (3, 0); self.goal = (3, 11)
        self.n_states = self.rows * self.cols; self.n_actions = 4

    def idx(self, s):
        return s[0] * self.cols + s[1]

    def step(self, s, a):
        r, c = s[0] + self.actions[a][0], s[1] + self.actions[a][1]
        r = min(max(r, 0), self.rows - 1); c = min(max(c, 0), self.cols - 1)
        if r == 3 and 0 < c < 11:
            return self.start, -100.0, False
        return (r, c), -1.0, (r, c) == self.goal


def _eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))


def td_control(env, method="q", episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=0, max_steps=10_000):
    """Train with Q-learning (`method='q'`) or SARSA (`'sarsa'`). Returns Q table and reward per episode."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        s = env.start; si = env.idx(s); a = _eps_greedy(Q, si, eps, rng); total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(s, a); s2i = env.idx(s2); total += r
            a2 = _eps_greedy(Q, s2i, eps, rng)
            target = r if done else r + gamma * (Q[s2i].max() if method == "q" else Q[s2i, a2])
            Q[si, a] += alpha * (target - Q[si, a])
            s, si, a = s2, s2i, a2
            if done:
                break
        rewards[ep] = total
    return Q, rewards


def greedy_path(env, Q, max_steps=100):
    s = env.start; path = [s]
    for _ in range(max_steps):
        a = int(np.argmax(Q[env.idx(s)]))
        s, _, done = env.step(s, a); path.append(s)
        if done:
            break
    return path
