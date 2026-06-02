"""Project paths and runtime settings.

Attributes
----------
PROJECT_ROOT : pathlib.Path
    Repository root (parent of ``config/``).
ANNEX_SNAPSHOT_DIR : pathlib.Path
    Pinned annex CSV directory for scoring oracles (see ``scoring/oracle/``).
COSING_API_BASE : str
    Base URL for the official EU CosIng API (see ``data/cosing_api.py``).
OLLAMA_MODEL : str
    Local LLM model name for the ReAct agent.
OLLAMA_BASE_URL : str
    Ollama server URL.
LANGFUSE_PUBLIC_KEY : str or None
    Langfuse public API key from the environment.
LANGFUSE_SECRET_KEY : str or None
    Langfuse secret API key from the environment.
LANGFUSE_BASE_URL : str
    Langfuse API host (default: cloud).
LANGFUSE_ENABLED : bool
    Whether tracing is configured (both keys present).
AGENT_LOG_TURNS : bool
    When ``True``, emit INFO-level turn summaries even if Langfuse is enabled.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Pinned annex CSVs used by scoring oracles (see scoring/oracle/).
ANNEX_SNAPSHOT_DIR = Path(
    os.getenv("ANNEX_SNAPSHOT_DIR", str(PROJECT_ROOT / "data" / "annex_snapshots" / "default"))
)

# Official EU CosIng API — the agent's external data source (see data/cosing_api.py).
COSING_API_BASE = "https://api.tech.ec.europa.eu/cosing20/1.0/api"

# Ollama — local LLM for the ReAct agent
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Langfuse — optional tracing and tool-accuracy scores (see observability/)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# Local turn summaries — INFO when Langfuse is off; set true to force INFO with tracing
AGENT_LOG_TURNS = os.getenv("AGENT_LOG_TURNS", "").lower() in ("1", "true", "yes")
