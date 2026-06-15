# QuantEdge Trading Company — Org Structure

## Desks and Agents

### Strategy Desk (Head: Chief Strategy Officer)
- **Quant Researcher**: Proposes new alpha factors, researches academic literature
- **Strategy Engineer**: Implements strategies in `backend/app/strategies/`
- **Backtesting Analyst**: Runs walk-forward backtests, monitors OOS performance
- Slack channel: `#strategy-desk`
- Agent bus channel: `strategy` → `agent:findings:strategy`

### ML Desk (Head: Chief ML Officer)
- **ML Engineer**: Trains and tunes models (LSTM, XGBoost, SSM)
- **Feature Engineer**: Designs and validates features in `backend/app/ml/features/`
- **Model Validator**: Runs IC/IR analysis, detects overfitting
- Slack channel: `#ml-desk`
- Agent bus channel: `ml` → `agent:findings:ml`

### Risk Desk (Head: Chief Risk Officer)
- **Risk Manager**: Monitors real-time exposures, enforces circuit breakers
- **Portfolio Optimizer**: Runs HRP/CVaR optimization, manages correlations
- **Regime Analyst**: Monitors HMM regime state, adjusts strategy weighting
- Slack channel: `#risk-desk`
- Agent bus channel: `risk` → `agent:findings:risk`

### Execution Desk (Head: Head of Execution)
- **Execution Trader**: Routes orders through TWAP/VWAP/RL algorithms
- **Slippage Analyst**: Tracks realized vs expected fill prices
- **Smart Router Engineer**: Tunes order routing logic
- Slack channel: `#execution-desk`

### Data Engineering (Head: Head of Data)
- **Data Pipeline Engineer**: Maintains price feeds, OHLCV ingestion
- **Alternative Data Analyst**: Processes funding rates, on-chain metrics
- Slack channel: `#data-engineering`

### Compliance / Backtesting (Head: Chief Compliance Officer)
- **Compliance Analyst**: Runs holistic reviews, manages promotion pipeline
- **Audit Manager**: Monitors all live trades, ensures paper-first policy
- Slack channel: `#compliance`

## Cross-Desk Protocols
1. New strategy proposed → Strategy Desk posts to `agent:findings:strategy`
2. ML Desk picks up signal improvements → posts to `agent:findings:ml`
3. Risk Desk monitors all desks → broadcasts P0 alerts to `agent:alerts:p0`
4. Holistic review runs daily at 06:00 UTC → Compliance posts results
5. Promotion-ready strategies → Slack DM to system user for approval
