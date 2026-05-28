# ML Applications Across Trading Desks

**Document owner:** ML Team  
**Last updated:** 2026-05-28  
**Purpose:** Catalogue every ML use case by desk — model type, inputs, outputs, label source, and retraining cadence. Intended as a living reference for model prioritisation, data pipeline planning, and onboarding.

---

## 1. Equity Desk

### 1.1 Intraday Price Direction (Classification)
Predicts whether a stock's close price will be higher than the current price over a 1-bar or 5-bar horizon. The model is a Gradient Boosted Tree (LightGBM) trained on the full QuantEdge feature set (technical, advanced indicators, multi-timeframe, wavelet). Labels are generated with `add_labels(horizon=5, threshold=0.002)`. Retrained nightly on a rolling 2-year window; walk-forward validated on the most recent 6 months.

### 1.2 Realized Volatility Forecasting (Regression)
Predicts next-day realized volatility using a linear blend of GARCH(1,1) residuals and an LSTM operating on Garman-Klass, Parkinson, and Yang-Zhang volatility features from the past 60 days. Input features include `gk_vol`, `yz_vol`, `vol_of_vol`, `vol_pct_rank`, `hurst_exponent`, and wavelet detail-level energies. Target label is the square root of the sum of squared 5-minute returns over the next trading day. Retrained weekly; updated daily if realized vol deviates more than 1.5σ from the forecast.

### 1.3 Statistical Arbitrage Pair Selection (Unsupervised + Classification)
Identifies cointegrated equity pairs for mean-reversion trading. Stage 1 uses a clustering model (HDBSCAN on 252-day rolling return correlation matrices) to form candidate universes; stage 2 scores each pair with a gradient-boosted classifier on cointegration test statistics, half-life of mean reversion, and spread Hurst exponent. Input data is daily OHLCV for the S&P 500 universe plus sector ETFs. The pair universe is refreshed monthly; the classifier is retrained quarterly.

### 1.4 Earnings Surprise Prediction (Classification)
Predicts whether a company will beat analyst EPS consensus by more than 5% using a TabNet model. Features include trailing 12-month revenue growth, operating-margin trend, analyst revision momentum, short interest ratio, options implied move vs historical move, and the `approx_entropy` of the trailing earnings series. Labels are sourced from FactSet actuals vs. consensus the morning of the earnings release. Trained on 10 years of earnings events; retrained after each earnings season.

### 1.5 Short Interest Prediction (Regression)
Forecasts 2-week change in short interest as a percentage of float for each stock in the trading universe. A Random Forest regressor is trained on borrowing cost trends, put-call ratio, options skew, recent price momentum, and Amihud illiquidity. Labels are derived from bi-weekly FINRA short interest reports. Retrained biweekly immediately after new short-interest data is published.

### 1.6 Sector Rotation Timing (Classification)
Predicts the sector likely to outperform the S&P 500 over the next 20 trading days using a multi-class LightGBM model with 11 sector outputs. Inputs are yield curve slope and curvature, ISM PMI momentum, credit spread changes, cross-sector relative-strength scores, and macro regime embeddings from the Macro Desk model (section 5.1). Labels are the sector with the highest total return over the forward 20-day window. Retrained monthly on a 15-year daily history.

### 1.7 Intraday Liquidity Risk Scoring (Regression)
Estimates the expected market-impact cost of executing a given order size based on real-time microstructure features. An XGBoost model uses rolling Kyle lambda, Corwin-Schultz spread, Amihud illiquidity, bid-ask spread from L2 data, and recent volume imbalance as inputs. The output is expected slippage in basis points. Labels are computed post-execution from actual fill prices vs. VWAP. Retrained daily on the prior 90 days of live execution data.

### 1.8 Overnight Gap Prediction (Classification)
Predicts the direction and magnitude of next-day opening gap (> 0.5% up / > 0.5% down / flat) using after-hours and pre-market signals combined with sentiment from earnings call transcripts (via FinBERT embeddings) and analyst note releases. Additional inputs include after-hours volume surge and options overnight skew. Labels are derived from (open_t+1 − close_t) / close_t. Retrained nightly on a 3-year rolling window.

---

## 2. Crypto Desk

### 2.1 Liquidation Cascade Risk (Classification)
Classifies market state as low / medium / high risk of a liquidation cascade in the next 4 hours using a gradient-boosted tree. Inputs are aggregate open interest by exchange, long-short ratio, estimated leverage distribution (inferred from funding rates and OI changes), realized volatility, and the wavelet `dwt_noise_ratio` and `power_high_freq` features of BTC price. Labels are derived from perpetual futures liquidation volumes: cascade = any 1-hour window where total liquidations exceed 3× the 90-day hourly average. Retrained daily on a 1-year rolling window.

### 2.2 Funding Rate Prediction (Regression)
Forecasts the next 8-hour perpetual swap funding rate for BTC, ETH, and top-10 altcoins using an LSTM (hidden=128, seq_len=48 4h bars). Inputs include trailing 7 funding rate payments, open interest change, spot-futures basis, long-short ratio, and cross-exchange volume imbalance. Target is the actual funding rate from the exchange API at the next settlement. Retrained weekly; fine-tuned intraday if funding diverges >2σ from model.

### 2.3 On-Chain Flow Prediction (Regression)
Predicts net exchange inflow (coins deposited minus withdrawn) over the next 24 hours for BTC and ETH using Transformer encoder on 14-day hourly on-chain time series. Input features include exchange balance deltas, miner outflow, stablecoin mint/burn rates, active address count change, and mean transaction value. Target is the log-transformed net exchange flow measured 24 hours forward (sourced from Glassnode/CryptoQuant). Retrained weekly on 3 years of daily on-chain history.

### 2.4 Whale Wallet Activity Signals (Classification)
Detects clusters of coordinated whale accumulation or distribution activity in the 24 hours prior to a significant price move using a graph-attention network (GAT) on wallet-to-wallet transfer graphs. Node features are wallet balance, historical trade frequency, and exchange affiliation; edge features are transfer amounts. Labels are +1 for accumulation (price up >5% in 48h) and −1 for distribution (price down >5%), with 0 as neutral. The graph snapshot is computed daily; the model is retrained monthly.

### 2.5 Exchange Flow Imbalance (Regression)
Predicts the net buy/sell pressure over the next 1-hour candle for a given exchange using a linear factor model plus XGBoost residual. Factors are order-book depth imbalance at five price levels, taker buy ratio (from exchange WebSocket), recent price impact per unit volume, and cross-exchange arbitrage spread. Labels are the realized net taker buy volume in the subsequent 1-hour window. Retrained every 6 hours on a rolling 30-day window.

### 2.6 Altcoin Beta-to-BTC Forecasting (Regression)
Predicts rolling 24-hour beta of each tracked altcoin to BTC, used to size cross-asset hedges. An ElasticNet model uses lagged beta, altcoin on-chain activity relative to BTC, project-specific liquidity ratio, and sector (DeFi / L2 / NFT) embeddings. Labels are the realized OLS beta over the next 24-hour window. Retrained daily; predictions are consumed by the risk sizing module.

### 2.7 Funding Rate Regime Detection (Classification)
Classifies the current funding rate environment as normal contango, extreme contango (> 0.1% per 8h), or backwardation, used to gate carry strategies. An LDA + logistic ensemble on rolling funding-rate percentile rank, open-interest-to-volume ratio, and cross-asset risk-off indicators (equity VIX, IG credit spread) produces regime labels. Labels are defined by funding rate threshold rules. Retrained monthly.

---

## 3. Options Desk

### 3.1 Implied Volatility Surface Prediction (Regression)
Forecasts the full IV surface (7 expiries × 9 delta strikes) for SPX and high-volume single names 1 trading day forward using a Graph Neural Network where nodes are (expiry, delta) grid points and edges encode calendar and skew relationships. Input features per node include current IV, realized vol at each tenor, term structure slope, skew curvature, and macro regime signals. Labels are the actual mid-market IV from the options chain the following morning. Retrained weekly; surface snapshots are taken at 4 PM ET close.

### 3.2 Optimal Strike Selection (Classification)
Selects the best strike and expiry for an options strategy (vertical spread, iron condor, or naked put) given the current market regime and risk budget. A multi-armed bandit with Thompson sampling tracks the Sharpe ratio of each strike-expiry combination historically. Features include delta, gamma exposure, break-even move vs. ATM IV, days-to-expiry, and the `power_low_freq` spectral feature of underlying price. Labels are the realised profit/loss at expiry normalised by premium received. Updated nightly with the day's expired positions.

### 3.3 Delta Hedging with Reinforcement Learning (RL)
Trains a Proximal Policy Optimisation (PPO) agent to dynamically adjust delta hedge ratios to minimise transaction-cost-adjusted P&L variance. The state is (delta, gamma, theta, vega, time-to-expiry, underlying returns autocorrelation, bid-ask spread). The action is the hedge ratio adjustment in [−1, +1] normalised deltas. Reward is risk-adjusted P&L minus transaction costs after each hedge interval. The agent is trained in simulation using historical options chains and retrained monthly.

### 3.4 Earnings IV Crush Timing (Regression)
Predicts the percentage IV crush that will occur in the 24 hours following an earnings announcement for each individual stock. An XGBoost model is trained on: pre-earnings ATM IV percentile rank, implied move vs. historical realised move over the prior 8 earnings, sector IV regime, and the ratio of post-earnings to pre-earnings IV over the same stock's history. Target label is (IV_post_earnings − IV_pre_earnings) / IV_pre_earnings. Retrained quarterly after each earnings season.

### 3.5 Term Structure Anomaly Detection (Unsupervised + Binary)
Identifies statistically anomalous dislocations in the VIX term structure (spot VIX vs. VIX3M vs. VIX6M) that historically precede outsized moves. An Isolation Forest is trained on term structure slope, curvature, contango/backwardation ratio, and realised-vs-implied vol gap per tenor. Anomaly scores above the 95th percentile trigger an alert for the desk. Labels for supervised validation are hand-annotated events where a dislocation preceded a > 3% S&P 500 move within 5 days. Retrained quarterly.

### 3.6 Skew Arbitrage Signal (Regression)
Predicts normalised put/call skew at the 25-delta strike 30 days forward to identify expensive or cheap wings for risk-reversal trades. A Ridge regression ensemble uses term structure features, realised skewness of underlying returns (`realized_skew`), sector stress indicators, and cross-asset correlation regime. Target is the 25-delta risk reversal value 30 calendar days forward. Retrained monthly on 5 years of daily options data.

### 3.7 Pin Risk Identification (Classification)
Predicts whether the underlying will close within 0.5% of a major open-interest strike on expiry Friday. A logistic regression model uses gamma exposure by strike (from CBOE open interest), volume concentration near strike, and distance from current price to max-pain level. Labels are binary: did the underlying pin (close within 0.5% of the strike) or not? Retrained weekly.

---

## 4. Polymarket / Prediction Markets Desk

### 4.1 NLP Event Outcome Prediction (Classification)
Predicts the resolution probability of binary political, economic, and sports markets on Polymarket using a fine-tuned RoBERTa-base model on news headlines and social media aggregates. Input is a rolling 72-hour window of news titles and tweet summaries related to the market's subject, encoded as mean token embeddings plus recency-weighted sentiment scores. Labels are the actual binary market resolution outcome. The classification head is fine-tuned weekly on newly resolved markets; the RoBERTa backbone is frozen.

### 4.2 Resolution Probability Calibration (Regression)
Calibrates raw model probability outputs (from section 4.1) against the actual resolution rate using Platt scaling and isotonic regression. A separate calibration curve is maintained per market category (elections, economic data, sports). Input is the raw model probability plus market liquidity (average daily volume), time-to-resolution, and the current market price. Target is the empirical resolution rate in that probability bin. Retrained weekly as new resolutions accumulate.

### 4.3 Market Manipulation Detection (Anomaly Detection + Classification)
Detects coordinated price manipulation attempts in prediction markets by identifying abnormal trading patterns. An Isolation Forest flags outliers on: trade size distribution, order timing clustering (seconds between trades), wallet concentration of volume, and price deviation from the NLP model's implied probability. Flagged events are reviewed by a human analyst and labelled; the supervised classifier is retrained monthly on confirmed manipulation cases.

### 4.4 Optimal Entry Timing (Regression)
Predicts the expected edge (model probability minus market price) at the best entry point over the next 24 hours for a given open market. A LightGBM model uses current time-to-resolution, rolling 1-hour price trend, news event calendar (scheduled announcements that may move the probability), and current bid-ask spread as inputs. Labels are the actual edge realised at the best achievable entry price over the subsequent 24 hours. Retrained nightly on the prior 90 days of market data.

### 4.5 Cross-Market Correlation Arbitrage (Unsupervised + Signal Generation)
Identifies pairs of Polymarket contracts that are correlated (e.g., "Fed raises rates in March" vs. "Inflation above 3% in Q1") and computes a spread signal when market prices diverge from the model-implied joint probability. A Bayesian network encodes the conditional dependencies between related market outcomes. Features are the pair of current market prices and the trailing 14-day price correlation between the pair. Retrained monthly as new market topics are listed.

### 4.6 Liquidity Timing for Block Trades (Regression)
Predicts the hour within the next 24 hours that will offer the best liquidity (tightest spread, deepest book) for executing a large position in a given prediction market. An XGBoost model uses hour-of-day, day-of-week cyclical features, proximity to a scheduled news event, and trailing 7-day hourly volume profile as inputs. Target label is the realised spread at execution divided by the median spread of that day. Retrained weekly.

---

## 5. Macro / FX Desk

### 5.1 Yield Curve Shape Prediction (Regression)
Forecasts the 3-month-ahead U.S. Treasury yield curve represented as a Nelson-Siegel factor triplet (level, slope, curvature) using a VAR(3) model with LSTM residual correction. Input features are current factor values, Fed funds rate expectations (OIS forwards), inflation breakeven rates, ISM PMI, and NFP surprise index. Labels are the actual Nelson-Siegel factors estimated from the Treasury yield curve 3 months forward. The VAR is estimated monthly; the LSTM is retrained quarterly.

### 5.2 Central Bank Pivot Detection (Classification)
Classifies the current monetary policy regime as tightening, on-hold, or pivoting-to-easing using a multi-class gradient-boosted classifier. Inputs include PCE inflation trend, unemployment gap, Fed funds rate vs. Taylor rule estimate, yield curve inversion duration, and NLP sentiment from FOMC statements (FinBERT embeddings averaged over the 500 most recent tokens). Labels are hand-annotated regime periods defined by actual policy rate direction over the subsequent 3 meetings. Retrained after each FOMC meeting.

### 5.3 Currency Regime Detection (Classification)
Detects the prevailing FX regime (risk-on trending, risk-off safe-haven, carry-driven, or range-bound) for major currency pairs (EUR/USD, USD/JPY, AUD/USD, GBP/USD) using a Hidden Markov Model with 4 states. Emission features per state are: rolling 20-day return, realised vol, carry spread (rate differential), and the wavelet `hurst_exponent` of FX returns. Labels are inferred by the Viterbi algorithm; the HMM is re-estimated monthly on 10 years of daily FX data. The regime signal gates carry vs. trend strategy allocation.

### 5.4 Cross-Asset Momentum Signal (Regression)
Predicts the expected 20-day return of each major asset class (equities, bonds, commodities, FX) based on cross-asset momentum spillover. A LASSO regression model uses trailing 1, 3, 6, and 12-month returns across 15 asset-class ETFs as inputs, capturing the Jegadeesh-Titman cross-sectional momentum effect. Labels are the actual 20-day forward return of each asset class. Retrained monthly on 20 years of daily returns; feature importance is used to identify the current dominant cross-asset momentum linkage.

### 5.5 Commodity Supply Shock Detection (Binary Classification)
Detects emerging supply shocks in energy and agricultural commodities before they are fully priced in by financial markets. A Random Forest classifier uses satellite-derived crop yield indices, shipping AIS density data, refinery utilisation rates, weather anomaly scores, and commodity futures term structure slope as inputs. Labels are derived from ex-post USDA and IEA supply revision reports: a supply shock is defined as a revision of >5% in a single month. Retrained quarterly as new supply data is published.

### 5.6 Inflation Regime Forecasting (Multi-class Classification)
Predicts the 6-month-ahead inflation regime (deflation / low / moderate / high) for the U.S., EU, and EM aggregate using a gradient-boosted ensemble. Features include commodity price trends (oil, wheat, copper), PPI-CPI spread, money supply growth, import price inflation, and wage growth proxies. Labels are derived from the actual CPI print 6 months forward, bucketed into four regimes. Used by the equity sector-rotation model (section 1.6) as a macro conditioning input. Retrained quarterly.

### 5.7 Safe-Haven Flow Intensity (Regression)
Quantifies the strength of risk-off safe-haven flows (USD, JPY, CHF, Gold, Treasuries) on a scale of 0 to 1, updated intraday. An XGBoost model uses: VIX level and 5-day change, 2-year Treasury yield change, USD index momentum, Gold-to-S&P correlation, and the equity put/call ratio. Target label is the composite Z-score of safe-haven asset returns over the next trading session. Retrained weekly; the signal is consumed by position-sizing logic across all desks.

### 5.8 G10 Interest Rate Differential Signal (Regression)
Forecasts 1-month changes in G10 interest rate differentials to generate carry-trade entry/exit signals in FX. A Ridge regression trained on inflation differentials, growth differentials (PMI spread), current account balances, and the slope of each country's yield curve. Labels are the actual 1-month change in 2-year yield spreads across G10 pairs. Retrained monthly. The signal feeds directly into the FX carry overlay strategy.

---

## Summary Table

| Desk | # Models | Model Types |
|------|----------|-------------|
| Equity | 8 | LightGBM, LSTM, Random Forest, TabNet, XGBoost, Logistic |
| Crypto | 7 | GBT, LSTM, Transformer, GAT, ElasticNet, LDA+Logistic |
| Options | 7 | GNN, PPO (RL), XGBoost, Isolation Forest, Ridge, Logistic |
| Polymarket | 6 | RoBERTa, Platt/Isotonic, Isolation Forest, LightGBM, Bayes Net, XGBoost |
| Macro/FX | 8 | VAR+LSTM, GBT, HMM, LASSO, Random Forest, XGBoost, Ridge |
| **Total** | **36** | |

---

*All models follow the walk-forward validation protocol (section "Walk-forward only" in root CLAUDE.md). No in-sample-only backtests are accepted as production-ready.*
