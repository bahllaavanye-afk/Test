# OA Feature Parity — audited against docs.optionalpha.com (2026-07-19)

Source: OA's own manual (docs.optionalpha.com/llms.txt, 104 entries). Verdicts:
have / partial / missing (queued). No claims without a mapped feature.

## Bots (core engine)
| OA feature | Status | QuantEdge mapping |
|---|---|---|
| Bots / rules engine | HAVE | Bot Builder: trigger→conditions→action, 61 templates |
| Safeguards (allocation, daily positions, position limit, day trading) | PARTIAL | allocation→size_pct, no_position guard; NO daily counter / at-once limit → queued |
| Automations (scheduled decision trees) | PARTIAL | flat trigger+conditions+action; nested decision trees missing → queued |
| Loops (watchlist iteration) | MISSING | desks loop symbols; bot engine doesn't → queued |
| Decision recipes (dozens of market/position/opportunity criteria) | PARTIAL | 6 condition types vs their catalog → queued (biggest gap) |
| Inputs (variables) | MISSING | queued (low) |
| Tags / untag statefulness | MISSING | queued |
| Automation logs | HAVE | bot run logs, last_result, /bots/{id}/activity |
| SmartPricing | MISSING | already queued (laddered repricing) |
| Webhooks (external signal triggers) | HAVE | /webhooks/tradingview (receive-only by design) |

## Finding trades
| Scanners | HAVE | desk layer + stock_scanners |
| Trade Ideas 2.0 (EV/Alpha metric) | MISSING | EV metric queued; strategies cover signal-gen |
| 0DTE Oracle | MISSING | needs real chains → unblocked by TRADIER_SANDBOX_TOKEN |
| Earnings Edge | PARTIAL | earnings_iv_crush strategy; no earnings backtest DB |
| Screener / custom watchlists | PARTIAL | Symbol Scout + scanners; no UI screener |
| Trade Grid / Top Strategies | PARTIAL | leaderboard ≈ top strategies |

## Backtesting
| Options backtester | PARTIAL | synthetic BS engine (limits stated); no historical chains |
| Compare & combine | HAVE | compare mode + leaderboard |
| Backtest → bot ("Automate your strategy") | MISSING | queued |
| Metrics | HAVE | tearsheet, Sharpe, DD, win rate |

## Managing positions
| Exit Options (TP/SL/trail/DTE) | PARTIAL | TP/SL/time/trailing yes; touch-$ and earnings-date exits missing |
| Monitor automations (custom exit logic) | PARTIAL | position_monitor task; not per-bot configurable |

## Platform & safety
| Order handling | HAVE | money-path tested, mleg, cancel-replace, UA/error-body |
| Data feeds | HAVE | Alpaca batched; ORATS-grade greeks pending Tradier token |
| Failsafe family (excessive-errors auto-off, duplicate orders, overlapping strikes, capital warnings, limits, pricing anomaly) | PARTIAL | have: cash-aware sizing, daily loss cap, double-fill guard, kill switches. Missing: per-bot excessive-errors auto-disable, overlapping-strikes check, pricing-anomaly gate → queued |
| Options Expiration Protocol (ITM handling at expiry) | MISSING | CRITICAL for the real-mleg desk → queued P1 |
| Probability (HV-based) / EV calculations | PARTIAL | BS pricer + HV vol; EV/probability decision inputs queued |
| Infrastructure & security docs | HAVE | CLAUDE.md tree + playbooks + research docs |

## Verdict
Core architecture: parity (rules engine, order handling, logs, webhooks,
backtest metrics). Real gaps, priority order: (1) Options Expiration Protocol,
(2) decision-recipe catalog + loops + tags, (3) safeguard depth, (4) failsafe
family, (5) SmartPricing, (6) EV/probability metrics, (7) backtest→bot
generator. All queued in IMPROVEMENTS.md.
