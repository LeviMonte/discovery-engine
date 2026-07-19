import numpy as np

from discovery.acquire import STRATEGIES, acquire


def test_all_strategies_return_valid_batch():
    rng = np.random.RandomState(0)
    n, batch = 50, 5
    mu = rng.normal(size=n)
    sigma = np.abs(rng.normal(size=n)) + 0.1
    for strat in STRATEGIES:
        idx = acquire(strat, mu, sigma, best_y=mu.max(), batch=batch, rng=rng)
        assert len(idx) == batch
        assert len(set(idx)) == batch                # no duplicates
        assert np.all((idx >= 0) & (idx < n))


def test_greedy_picks_highest_mean():
    rng = np.random.RandomState(0)
    mu = np.arange(20.0)
    sigma = np.ones(20)
    idx = acquire("greedy", mu, sigma, best_y=0, batch=3, rng=rng)
    assert set(idx) == {19, 18, 17}


def test_max_var_picks_highest_sigma():
    rng = np.random.RandomState(0)
    mu = np.zeros(20)
    sigma = np.arange(20.0)
    idx = acquire("max_var", mu, sigma, best_y=0, batch=2, rng=rng)
    assert set(idx) == {19, 18}


def test_ucb_between_greedy_and_maxvar():
    # ucb with beta>0 should shift picks toward high mu+sigma
    rng = np.random.RandomState(0)
    mu = np.array([10.0, 0, 0, 0, 9.0])
    sigma = np.array([0.1, 5.0, 0.1, 0.1, 2.0])
    idx = acquire("ucb", mu, sigma, best_y=0, batch=1, rng=rng, beta=1.5)
    # index 0 has score 10.15, index 4 has 12.0, index1 has 7.5 -> pick 4
    assert idx[0] == 4
