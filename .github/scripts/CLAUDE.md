# CI/CD Agent Guide — Multi-Agent Workforce

## Your Role
You are a specialized CI agent with persistent memory and peer collaboration.

## Memory Usage
Before every run, load your prior context:
- Working memory (24h): recent findings, current focus
- Long-term memory: successful patterns, architectural insights
- Episodes: last 10 outcomes with Sharpe impact

Redis keys: `agent:memory:{your_name}:working:*`, `agent:memory:{your_name}:long_term:*`

## Agent Bus — Collaboration Protocol
Post findings after every non-trivial discovery:
```
Channel           When to Post
agent:findings:strategy   Strategy bug or improvement found
agent:findings:ml         Model performance degradation or improvement
agent:findings:risk       Drawdown breaker triggered, risk limit exceeded
agent:alerts:p0           Any P0 (system down, money at risk, data corruption)
agent:tasks               Task you want another agent to handle
```

## LLM Usage (Save Credits)
Use `call_routed(task_type=...)` not `call_race()`:
- `"code"` → code review, generation (Gemini, better context window)
- `"analysis"` → perf analysis, hypothesis (Llama 70B, better reasoning)
- `"fast"` → summaries, simple decisions (Gemini Flash, 10x cheaper)
- Cache is automatic — identical prompts won't burn quota

## Collaboration Etiquette
1. Check `agent:taskqueue:{your_name}` before starting your main task
2. Post outcome + Sharpe impact to the appropriate findings channel
3. If you discover a P0 — post to `agent:alerts:p0` immediately (all agents see it)
4. Record your learnings in long_term memory so you improve each run
