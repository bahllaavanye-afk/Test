"""
Company org chart as code. Each desk has a name, head, agents, and Slack channel.
Agents are autonomous workers that use the AgentBus and AgentMemory to collaborate.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Employee:
    name: str
    title: str
    desk: str
    responsibilities: list[str] = field(default_factory=list)
    agent_bus_channel: str | None = None


@dataclass(frozen=True)
class Desk:
    name: str
    head_title: str
    slack_channel: str
    agent_bus_channel: str | None
    employees: list[Employee] = field(default_factory=list)


DESKS: dict[str, Desk] = {
    "strategy": Desk(
        name="Strategy Desk",
        head_title="Chief Strategy Officer",
        slack_channel="#strategy-desk",
        agent_bus_channel="strategy",
        employees=[
            Employee("quant_researcher", "Quant Researcher", "strategy",
                     ["Research alpha factors", "Review academic literature"], "strategy"),
            Employee("strategy_engineer", "Strategy Engineer", "strategy",
                     ["Implement strategies", "Run backtests"], "strategy"),
            Employee("backtesting_analyst", "Backtesting Analyst", "strategy",
                     ["Walk-forward validation", "OOS performance"], "strategy"),
        ],
    ),
    "ml": Desk(
        name="ML Desk",
        head_title="Chief ML Officer",
        slack_channel="#ml-desk",
        agent_bus_channel="ml",
        employees=[
            Employee("ml_engineer", "ML Engineer", "ml",
                     ["Train LSTM/XGBoost models", "Hyperparameter tuning"], "ml"),
            Employee("feature_engineer", "Feature Engineer", "ml",
                     ["Design features", "IC/IR analysis"], "ml"),
            Employee("model_validator", "Model Validator", "ml",
                     ["Detect overfitting", "Validate OOS metrics"], "ml"),
        ],
    ),
    "risk": Desk(
        name="Risk Desk",
        head_title="Chief Risk Officer",
        slack_channel="#risk-desk",
        agent_bus_channel="risk",
        employees=[
            Employee("risk_manager", "Risk Manager", "risk",
                     ["Monitor real-time exposures", "Circuit breakers"], "risk"),
            Employee("portfolio_optimizer", "Portfolio Optimizer", "risk",
                     ["HRP optimization", "Correlation limits"], "risk"),
            Employee("regime_analyst", "Regime Analyst", "risk",
                     ["HMM regime state", "Strategy weighting"], "risk"),
        ],
    ),
    "execution": Desk(
        name="Execution Desk",
        head_title="Head of Execution",
        slack_channel="#execution-desk",
        agent_bus_channel=None,
        employees=[
            Employee("execution_trader", "Execution Trader", "execution",
                     ["TWAP/VWAP/RL routing", "Order management"], None),
            Employee("slippage_analyst", "Slippage Analyst", "execution",
                     ["Track fill prices", "Slippage reports"], None),
        ],
    ),
    "data": Desk(
        name="Data Engineering",
        head_title="Head of Data",
        slack_channel="#data-engineering",
        agent_bus_channel=None,
        employees=[
            Employee("data_pipeline", "Data Pipeline Engineer", "data",
                     ["Price feeds", "OHLCV ingestion"], None),
            Employee("alt_data", "Alternative Data Analyst", "data",
                     ["Funding rates", "On-chain metrics"], None),
        ],
    ),
    "compliance": Desk(
        name="Compliance",
        head_title="Chief Compliance Officer",
        slack_channel="#compliance",
        agent_bus_channel=None,
        employees=[
            Employee("compliance_analyst", "Compliance Analyst", "compliance",
                     ["Holistic reviews", "Promotion pipeline"], None),
            Employee("audit_manager", "Audit Manager", "compliance",
                     ["Trade audits", "Paper-first policy"], None),
        ],
    ),
}


def get_all_employees() -> list[Employee]:
    return [emp for desk in DESKS.values() for emp in desk.employees]


def get_desk(name: str) -> Desk | None:
    return DESKS.get(name)
