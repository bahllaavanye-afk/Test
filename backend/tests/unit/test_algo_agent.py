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


def test_avg_sharpe_zero_when_unexplored():
    c = AlgoCandidate(name="x", symbol="SPY", strategy_type="manual")
    assert c.n_runs == 0
    assert c.avg_sharpe == 0.0


def test_ucb_score_with_zero_total_runs_raises():
    c = AlgoCandidate(name="x", symbol="SPY", strategy_type="manual",
                       n_runs=3, total_sharpe=3.0)
    with pytest.raises(ValueError):
        c.ucb_score(total_runs=0)


def test_ucb_score_total_runs_less_than_n_runs():
    c = AlgoCandidate(name="x", symbol="SPY", strategy_type="manual",
                       n_runs=10, total_sharpe=15.0)
    # total_runs less than n_runs; should still return a finite float without error
    score = c.ucb_score(total_runs=5)
    assert isinstance(score, float)
    assert score != float("inf")


def test_agent_initializes_candidates():
    agent = AlgoAgent()
    assert len(agent._candidates) > 0
    assert all(isinstance(c, AlgoCandidate) for c in agent._candidates.values())


def test_agent_picks_unexplored_first():
    agent = AlgoAgent()
    # All start with n_runs=0, picker should return one of them
    candidate = agent._select_candidate()
    assert candidate.n_runs == 0


def test_agent_select_candidate_when_all_explored():
    agent = AlgoAgent()
    # Force all candidates to have at least one run
    for cand in agent._candidates.values():
        cand.n_runs = 1
        cand.total_sharpe = 1.0
    selected = agent._select_candidate()
    # Should still return a candidate (no unexplored left)
    assert isinstance(selected, AlgoCandidate)
    assert selected.n_runs == 1


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


def test_leaderboard_includes_all_candidates():
    agent = AlgoAgent()
    leaderboard = agent.get_leaderboard()
    assert len(leaderboard) == len(agent._candidates)
    # All entries should have required keys
    for entry in leaderboard:
        assert "name" in entry
        assert "symbol" in entry
        assert "avg_sharpe" in entry
        assert isinstance(entry["avg_sharpe"], float)