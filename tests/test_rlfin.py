import numpy as np, pandas as pd
from rlfin import bandits, mdp, tabular, trading


def test_ucb_finds_best_arm():
    regret, opt = bandits.simulate_stationary(lambda k, rng: bandits.UCB1(k, rng=rng), [0, 0.2, 0.5], horizon=1500, runs=20)
    assert opt[-200:].mean() > 0.8
    assert regret[-1] < 0.5 * 1500 * 0.5 / 3  # far below uniform random play


def test_thompson_beats_random():
    regret, _ = bandits.simulate_stationary(lambda k, rng: bandits.GaussianThompson(k, rng=rng), [0, 0.3], horizon=800, runs=20)
    assert regret[-1] < 0.3 * 800 / 2 / 3


def test_backtest_uses_only_past_rewards():
    idx = pd.RangeIndex(6)
    R = pd.DataFrame({"a": [0.01] * 6, "b": [0.02] * 6}, index=idx)
    out = bandits.backtest(R, bandits.EpsilonGreedy(2, eps=0.0))
    assert list(out["arm"][:2]) == ["a", "b"]          # initial round-robin
    assert (out["arm"][2:] == "b").all()


def test_dp_matches_closed_form():
    X, N = 60, 6
    _, pol = mdp.solve_execution(X, N, eta=1.0, sigma=0.5, lam=0.5)
    path = mdp.schedule(pol, X)
    exact = mdp.almgren_chriss(X, N, eta=1.0, sigma=0.5, lam=0.5)
    assert path[-1] == 0
    assert np.max(np.abs(path - exact)) <= 1.0           # integer grid vs continuous solution


def test_risk_neutral_is_twap():
    _, pol = mdp.solve_execution(40, 4, eta=1.0, sigma=1.0, lam=0.0)
    assert list(mdp.schedule(pol, 40)) == [40, 30, 20, 10, 0]


def test_value_and_policy_iteration_agree():
    rng = np.random.default_rng(1)
    P = rng.random((2, 5, 5)); P /= P.sum(2, keepdims=True)
    R = rng.normal(size=(5, 2))
    V1, p1 = mdp.value_iteration(P, R, 0.9)
    V2, p2 = mdp.policy_iteration(P, R, 0.9)
    assert np.allclose(V1, V2, atol=1e-6) and (p1 == p2).all()


def test_cliff_q_learning_takes_short_path():
    env = tabular.CliffWalk()
    Q, _ = tabular.td_control(env, "q", episodes=500, seed=0)
    assert len(tabular.greedy_path(env, Q)) == 14          # 13 moves along the cliff edge


def test_trading_cost_accounting():
    Q = np.zeros((1, 2, 2)); Q[0, :, 1] = 1.0              # always long
    rets, pos = trading.run_policy(Q, np.zeros(3, int), np.array([0.01, 0.02, -0.01]), cost=0.001)
    assert np.allclose(rets, [0.009, 0.02, -0.01]) and pos.tolist() == [1, 1, 1]
