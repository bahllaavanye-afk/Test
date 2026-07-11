# Playbook: Alpaca crypto 403 (first-trade blocker)
1. app.alpaca.markets (paper) → check crypto trading agreement is accepted.
2. If already enabled: read the latest desk-trading-crypto run log — the order
   POST now prints Alpaca's response body; act on that exact reason.
3. If keys were rotated: confirm ALPACA_API_KEY/SECRET repo secrets match the
   paper account with crypto enabled (key-relay syncs them to Render).
