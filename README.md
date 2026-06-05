# EU Cosmetics Ingredient Compliance Agent

A neuro-symbolic ReAct agent for EU cosmetic ingredient compliance, built on
LangGraph. It queries the **live official EU CosIng API** and wraps the LLM in
three symbolic guardrails (input, tool-argument, output).

## Setup

```bash
pip install -e .
ollama pull llama3.1:8b
```

The agent fetches regulatory data live from the EU CosIng API at runtime and
caches annex exports under `data/.cosing_cache/` — no database to build.

## Apps

| App | Command | Purpose |
|-----|---------|---------|
| **Agent chat** | `streamlit run ui/chat_app.py` | Ask compliance questions |

## CLI

```bash
python -m agent.run "Is 0.8% phenoxyethanol allowed in a leave-on serum?"
```

## Observability

Tracing is optional. Copy `.env.example` to `.env` and set Langfuse keys to enable it.

| Signal | Where |
|--------|--------|
| **Traces** | Langfuse — one `compliance-turn` span per user turn (graph + LLM callbacks) |
| **Scores** | Langfuse — per-tool oracle accuracy, composite `turn_trustworthiness`, per-stage guardrail pass/fail |
| **Metadata** | Langfuse span — intent, tools used, blocked flag, guardrail summary |
| **Local logs** | Logger `eu_cosmetics.agent` — INFO summaries when Langfuse is off; set `AGENT_LOG_TURNS=true` to keep INFO with tracing |

The Streamlit chat passes a stable `session_id` so turns group into one Langfuse session.

```bash
# Example: run CLI with turn summaries on stderr
AGENT_LOG_TURNS=true python -m agent.run "Is retinol allowed at 0.3%?"
```

## Architecture

```
agent/           LangGraph StateGraph: intent → guards → ReAct loop → exposition → verify
  graph.py         explicit StateGraph wiring every node
  nodes.py         intent, input_guard, agent, tool_guard, tools, exposition, output_verify
  state.py         AgentState
guardrails/      3-stage symbolic guardrails — see docs/GUARDRAILS.md
tools/           3 tools (live API lookup, concentration, labelling)
data/            cosing_api.py (live EU CosIng client) + product_types.py (static vocab)
ui/              Streamlit chat
observability/   Langfuse integration, turn tracing, local turn logs
scoring/         CSV oracles, per-tool accuracy, composite trustworthiness
docs/SCORING.md  Full scoring spec (§2.1)
```

## Scoring

Code-based oracle scoring and the composite **turn trustworthiness** metric are
documented in [`docs/SCORING.md`](docs/SCORING.md). Scores are turn-scoped and
exported to Langfuse when tracing is enabled.

