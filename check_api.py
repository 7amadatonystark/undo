"""
check_api.py — checks that our NVIDIA key and models work. Uses about 4 API calls.

    python check_api.py

Test 1: AGENT_MODEL and JUDGE_MODEL each answer a simple message.
Test 2: AGENT_MODEL can use a tool. The real agent (Step 1) only works if it can.
"""

import json
from datetime import datetime

import openai

import config
from llm import chat

# A dummy tool, described in the format the API expects. It only reads the clock.
GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time (hours:minutes).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def get_time():
    return datetime.now().strftime("%H:%M")


def test_hello(label, model):
    """Test 1: send one simple message and print the reply."""
    if not model:
        print(f"{label}: skipped ({label} is not set in .env)")
        return
    reply = chat([{"role": "user", "content": "Say hello in one sentence"}], model=model)
    print(f"{label} ({model}): {(reply.content or '(no text)').strip()}")


def test_tool_use():
    """Test 2: the model must ask for get_time PROPERLY, then use the result."""
    messages = [{"role": "user", "content": "What time is it? Use the get_time tool."}]
    reply = chat(messages, tools=[GET_TIME_TOOL])

    # A real tool call arrives in reply.tool_calls. Some models instead write
    # something like {"name": "get_time"} into their text, which our agent can't run.
    if not reply.tool_calls:
        print("FAIL: no tool call. The model answered with text instead:")
        print("   ", reply.content)
        return False
    for call in reply.tool_calls:
        print(f"Tool call: {call.function.name}({call.function.arguments})")
        if call.function.name != "get_time":
            print("FAIL: it called a tool that doesn't exist")
            return False
        try:
            json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            print("FAIL: its tool arguments are not valid JSON")
            return False

    # Send back what the model asked for, plus one answer per tool call.
    # This is the same loop the real agent will use in Step 1.
    messages.append({
        "role": "assistant",
        "content": reply.content or "",
        "tool_calls": [{"id": call.id, "type": "function",
                        "function": {"name": call.function.name,
                                     "arguments": call.function.arguments}}
                       for call in reply.tool_calls],
    })
    now = get_time()
    for call in reply.tool_calls:
        messages.append({"role": "tool", "tool_call_id": call.id, "content": now})

    final = chat(messages, tools=[GET_TIME_TOOL])
    if not final.content:
        print("FAIL: no final answer after getting the tool result")
        return False
    print(f"Tool result we sent: {now}")
    print(f"Final answer: {final.content.strip()}")
    return True


def main():
    print("=== Test 1: say hello ===")
    for label, model in [("AGENT_MODEL", config.AGENT_MODEL), ("JUDGE_MODEL", config.JUDGE_MODEL)]:
        try:
            test_hello(label, model)
        except openai.APIError as error:
            print(f"{label} ({model}): FAIL: {error}")

    print("\n=== Test 2: tool use (AGENT_MODEL) ===")
    try:
        ok = test_tool_use()
    except openai.APIError as error:
        print(f"FAIL: {error}")
        ok = False
    print("PASS: the agent model can use tools" if ok else
          "Tool use FAILED: we should pick a different AGENT_MODEL")


if __name__ == "__main__":
    main()
