"""
config.py — every setting the project reads from .env, in one place.

The API key and model names live in .env (which Git ignores), never in the code.
Other files import them from here instead of reading .env themselves.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from this project folder, so it works no matter where Python is started.
# override=True: the .env file always wins. VS Code also copies .env into a terminal
# when it opens, and without this, editing .env would only work in a NEW terminal.
load_dotenv(Path(__file__).parent / ".env", override=True)

# NVIDIA's API accepts the same requests as OpenAI's, so we use the openai package.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def require(name):
    """Return a setting from .env, or stop the program if it is missing."""
    value = os.getenv(name, "").strip()
    if not value:
        # Name the missing setting, but never print a value: it could be the key.
        sys.exit(f"Missing {name} in .env. Add a line like this to .env: {name}=...")
    return value


NVIDIA_API_KEY = require("NVIDIA_API_KEY")
AGENT_MODEL = require("AGENT_MODEL")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "").strip() or None    # optional until Step 3

# How random the agent's choices are (0 = almost always the same answer). Some randomness
# lets repeated runs differ like real use. It is the SAME in every run and saved with each run.
AGENT_TEMPERATURE = 0.7
