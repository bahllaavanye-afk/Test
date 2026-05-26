# Execution Agent Guide

## Your Role
You improve order execution to minimize slippage.

## Current Algorithms
| Algorithm | When Used | Avg Slippage |
|-----------|-----------|--------------|
| market | urgent / fallback | highest |
| limit_first | default crypto/equity | 5-15 bps saved |
| twap | orders >$10k | minimal market impact |
| vwap | participation rate trading | tracks volume profile |
| iceberg | very large orders | hides true size |

## Decision Logic (SmartOrderRouter)
```python
if order_size > $10k → TWAP
elif limit order → limit_first
else → market
```

## Adding a New Algorithm
1. Create `execution/my_algo.py` implementing `async execute(request) -> OrderResult`
2. Add case in `execution/smart_router.py`
3. Test with paper orders and compare slippage in Analytics dashboard

## Monitoring
Analytics page → "Slippage by Execution Algorithm" shows real-time comparison.
