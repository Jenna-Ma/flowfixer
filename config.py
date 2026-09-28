"""Global configuration for the FLOWFIXER reproduction.

All LLM traffic goes through a single OpenAI-compatible endpoint (vveai.com).
Model names follow the paper (Sec. 4.5 "Experiment Setup"):
  - Backbone for FLOWFIXER (spec inference, diagnosis, repair, assessment): GPT-5.2
  - Multi-LLM judges (Sec. 3.4.3): Claude Opus 4.8, Gemini 3.5 Flash,
    DeepSeek V4 Pro, Qwen3.5-35B-A3B

The exact provider-side model id strings may need small tweaks to match the
aggregator's catalogue; override any value below with an environment variable.
"""
from __future__ import annotations

import os


# --------------------------------------------------------------------------- #
# LLM endpoint                                                                 #
# --------------------------------------------------------------------------- #
# The key/base_url were provided by the user; env vars take precedence so the
# secret never has to be edited in code for a different deployment.
API_KEY: str = 'your_api_key_here'  # Replace with your actual
BASE_URL: str = os.environ.get("FLOWFIXER_BASE_URL", "your_base_url_here")  # Replace with your actual base URL

# Backbone model used by every FLOWFIXER reasoning step.
BACKBONE_MODEL: str = os.environ.get("FLOWFIXER_BACKBONE", "gpt-5.2")

# Four independent judges for the Multi-LLM Judge (Sec. 3.4.3).
JUDGE_MODELS = os.environ.get(
    "FLOWFIXER_JUDGES",
    "claude-opus-4.8,gemini-3.5-flash,deepseek-v4-pro,qwen3.5-35b-a3b",
).split(",")

# Model used to simulate workflow execution when the dataset does not ship an
# executable runtime (dynamic verification, Sec. 3.4.3) and to synthesize the
# unseen test inputs for RQ4.
EXECUTOR_MODEL: str = os.environ.get("FLOWFIXER_EXECUTOR", "gpt-5.2")

# --------------------------------------------------------------------------- #
# Sampling / generation                                                       #
# --------------------------------------------------------------------------- #
TEMPERATURE: float = float(os.environ.get("FLOWFIXER_TEMPERATURE", "0.2"))
MAX_TOKENS: int = int(os.environ.get("FLOWFIXER_MAX_TOKENS", "4096"))
REQUEST_TIMEOUT: int = int(os.environ.get("FLOWFIXER_TIMEOUT", "120"))
MAX_RETRIES: int = int(os.environ.get("FLOWFIXER_MAX_RETRIES", "4"))

# --------------------------------------------------------------------------- #
# Pipeline hyper-parameters                                                    #
# --------------------------------------------------------------------------- #
# Retry budget for the diagnosis <-> repair loop (Sec. 3.4.3).
REPAIR_RETRY_BUDGET: int = int(os.environ.get("FLOWFIXER_RETRY_BUDGET", "5"))

# Failure-attribution suspicious score (Sec. 3.3.1):
#   score(n) = ALPHA * violation_rate(n) + (1 - ALPHA) * propagation_impact(n)
# The paper describes the two factors but not the exact weighting; ALPHA is our
# instantiation and is exposed here for tuning / ablation.
ATTRIBUTION_ALPHA: float = float(os.environ.get("FLOWFIXER_ALPHA", "0.6"))

# Offset Rationality (Sec. 3.4.2): acceptable modification magnitude expressed
# as a fraction of the number of nodes in the original workflow.
OFFSET_MIN_RATIO: float = float(os.environ.get("FLOWFIXER_OFFSET_MIN", "0.0"))
OFFSET_MAX_RATIO: float = float(os.environ.get("FLOWFIXER_OFFSET_MAX", "0.6"))
# Absolute floor: at least one edit is always allowed.
OFFSET_MIN_EDITS: int = 1

# Experience-pool retrieval (Sec. 3.5): a candidate experience is reused when
# its similarity to the current failure exceeds this threshold.
EXPERIENCE_SIM_THRESHOLD: float = float(os.environ.get("FLOWFIXER_EXP_SIM", "0.55"))
EXPERIENCE_TOP_K: int = int(os.environ.get("FLOWFIXER_EXP_TOPK", "3"))

# Number of times each case is repaired to average out randomness (Sec. 4.5).
NUM_RUNS: int = int(os.environ.get("FLOWFIXER_RUNS", "3"))

# RQ4: unseen test inputs generated per original workflow (Sec. 4.5).
RQ4_UNSEEN_PER_WORKFLOW: int = int(os.environ.get("FLOWFIXER_RQ4_N", "100"))

# --------------------------------------------------------------------------- #
# Caching                                                                      #
# --------------------------------------------------------------------------- #
# On-disk cache of LLM responses (keyed by model + prompt hash). Keeps repeated
# runs cheap and deterministic; delete the directory to force fresh calls.
CACHE_DIR: str = os.environ.get(
    "FLOWFIXER_CACHE",
    os.path.join(os.path.dirname(__file__), ".llm_cache"),
)
ENABLE_CACHE: bool = os.environ.get("FLOWFIXER_CACHE_ON", "1") == "1"

# Set FLOWFIXER_MOCK=1 to short-circuit every LLM call with a deterministic
# offline stub (useful for wiring/CI; produces structurally valid but trivial
# outputs). Off by default per the user's setup.
MOCK_LLM: bool = os.environ.get("FLOWFIXER_MOCK", "0") == "1"
