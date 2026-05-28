"""AlgoAgent UCB1 selection tests."""
import pytest
from app.tasks.algo_agent import AlgoAgent, AlgoCandidate


def test_ucb_score_unexplored_infinite():
    c = AlgoCandidate(name="x", symbol="SPY", strategy_type="manual")
    assert c.ucb_score(total_runs=10) == float("inf")


def test_ucb_score_explored_finite():
    c = AlgoCandidate(name="x", symbol="SPY", strategy_type="manual",
                       n_runs=5, total_sharpe=5.0)
    score = c.ucb_score(total_runs=100)
    assert 0 < score < float("inf")


def test_avg_sharpe_calculation():
    c = AlgoCandidate(name="x", symbol="SPY", strategy_type="manual",
                       n_runs=4, total_sharpe=4.8)
    assert c.avg_sharpe == 1.2


def test_agent_initializes_candidates():
    agent = AlgoAgent()
    assert len(agent._candidates) > 0
    assert all(isinstance(c, AlgoCandidate) for c in agent._candidates.values())


def test_agent_picks_unexplored_first():
    agent = AlgoAgent()
    # All start with n_runs=0, picker should return one of them
    candidate = agent._select_candidate()
    assert candidate.n_runs == 0


def test_leaderboard_sorted():
    agent = AlgoAgent()
    # Manually set some sharpe stats
    keys = list(agent._candidates.keys())
    agent._candidates[keys[0]].n_runs = 5
    agent._candidates[keys[0]].total_sharpe = 5.0  # avg 1.0
    agent._candidates[keys[1]].n_runs = 5
    agent._candidates[keys[1]].total_sharpe = 10.0  # avg 2.0

    leaderboard = agent.get_leaderboard()
    # Sorted descending by avg_sharpe
    sharpes = [r["avg_sharpe"] for r in leaderboard]
    assert sharpes == sorted(sharpes, reverse=True)
