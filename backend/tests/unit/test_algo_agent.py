"""AlgoAgent UCB1 selection tests."""
import pytest
from app.tasks.algo_agent import AlgoAgent, AlgoCandidate

# Constants
DEFAULT_NAME = "x"
DEFAULT_SYMBOL = "SPY"
DEFAULT_STRATEGY = "manual"

TOTAL_RUNS_UNEXPLORED = 10
TOTAL_RUNS_EXPLORED = 100
N_RUNS = 5
TOTAL_SHARPE = 5.0
N_RUNS_AVG = 4
TOTAL_SHARPE_AVG = 4.8
EXPECTED_AVG_SHARPE = 1.2
INF = float("inf")


def test_ucb_score_unexplored_infinite():
    c = AlgoCandidate(name=DEFAULT_NAME, symbol=DEFAULT_SYMBOL, strategy_type=DEFAULT_STRATEGY)
    assert c.ucb_score(total_runs=TOTAL_RUNS_UNEXPLORED) == INF


def test_ucb_score_explored_finite():
    c = AlgoCandidate(name=DEFAULT_NAME, symbol=DEFAULT_SYMBOL, strategy_type=DEFAULT_STRATEGY,
                       n_runs=N_RUNS, total_sharpe=TOTAL_SHARPE)
    score = c.ucb_score(total_runs=TOTAL_RUNS_EXPLORED)
    assert 0 < score < INF


def test_avg_sharpe_calculation():
    c = AlgoCandidate(name=DEFAULT_NAME, symbol=DEFAULT_SYMBOL, strategy_type=DEFAULT_STRATEGY,
                       n_runs=N_RUNS_AVG, total_sharpe=TOTAL_SHARPE_AVG)
    assert c.avg_sharpe == EXPECTED_AVG_SHARPE


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
    agent._candidates[keys[0]].n_runs = N_RUNS
    agent._candidates[keys[0]].total_sharpe = TOTAL_SHARPE  # avg 1.0
    agent._candidates[keys[1]].n_runs = N_RUNS
    agent._candidates[keys[1]].total_sharpe = 10.0  # avg 2.0

    leaderboard = agent.get_leaderboard()
    # Sorted descending by avg_sharpe
    sharpes = [r["avg_sharpe"] for r in leaderboard]
    assert sharpes == sorted(sharpes, reverse=True)