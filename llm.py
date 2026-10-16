"""
llm.py — the ONE door every AI model call in this project goes through.

Why one door? So two jobs happen for every call, in one place:
  1. THROTTLE: at most 30 requests per minute. NVIDIA's free limit is about 40 per
     minute, shared by every model on our key, so we stay safely under it.
  2. RETRY: if NVIDIA says "too many requests" (429) or has a server error (5xx),
     wait and try again, waiting twice as long each time. At most 3 tries in total.

Usage:
    from llm import chat
    reply = chat([{"role": "user", "content": "Hi"}])
    reply.content        # the model's text answer
    reply.tool_calls     # the tools it wants to call, or None
"""

import time

import openai

import config

MAX_REQUESTS_PER_MINUTE = 30
SECONDS_BETWEEN_REQUESTS = 60 / MAX_REQUESTS_PER_MINUTE     # 2.0 seconds
MAX_TRIES = 3

# max_retries=0 turns off the openai package's own hidden retries, so the rules in
# this file are the only ones (otherwise it would retry inside our retries).
# timeout: give up on one request after 2 minutes instead of the default 10.
client = openai.OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY,
                       max_retries=0, timeout=120)

last_request_time = 0.0     # when the last request was sent (seconds, time.monotonic())


def wait_for_turn():
    """Sleep until SECONDS_BETWEEN_REQUESTS have passed since the last request."""
    global last_request_time
    wait = SECONDS_BETWEEN_REQUESTS - (time.monotonic() - last_request_time)
    if wait > 0:
        time.sleep(wait)
    last_request_time = time.monotonic()


def is_worth_retrying(error):
    """429 (too many requests) and 5xx (server trouble) often fix themselves.
    Other errors, like a wrong model name or a bad key, will fail again every time."""
    return isinstance(error, openai.APIStatusError) and (
        error.status_code == 429 or error.status_code >= 500)


def backoff_seconds(attempt):
    """How many seconds to wait after failed try number `attempt` (1, 2, ...)."""
    # Exponential backoff: the wait DOUBLES after every failed try (2 s, 4 s, 8 s ...).
    # If an overloaded server gets instant retries from every program, it stays
    # overloaded. Waiting longer each time gives it room to recover.
    return 2 ** attempt


def send_once(model, messages, tools):
    """One request to NVIDIA, with no throttle or retry. Only sends tools if we have some."""
    if tools:
        return client.chat.completions.create(model=model, messages=messages, tools=tools)
    return client.chat.completions.create(model=model, messages=messages)


def chat(messages, tools=None, model=None):
    """Send messages to a model and return its reply (with .content and .tool_calls)."""
    model = model or config.AGENT_MODEL
    for attempt in range(1, MAX_TRIES + 1):
        wait_for_turn()
        try:
            response = send_once(model, messages, tools)
            return response.choices[0].message
        except openai.APIStatusError as error:
            if not is_worth_retrying(error) or attempt == MAX_TRIES:
                raise                   # give up: let the caller see the error
            wait = backoff_seconds(attempt)
            print(f"[llm] error {error.status_code}; waiting {wait}s, "
                  f"then try {attempt + 1} of {MAX_TRIES}")
            time.sleep(wait)
