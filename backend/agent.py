"""STEP B - The tool-execution loop (prepare_context -> call_model -> run_turn)."""
import json
from datetime import datetime
from pathlib import Path

import httpx

from backend.hr_tools import TOOLS, TOOL_FUNCTIONS, PREDICTIONS

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "llama3.2"
MAX_STEPS = 6          # the model may be called at most 6 times per question
MAX_TURNS = 10         # remember the last 10 question/answer pairs
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "tool_calls.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful, concise HR assistant. Always respond clearly in English. "
    "Use the tools to get predictions. Never invent numbers: always call a tool first. "
    "If the user does not give a required input, ask for it."
)

history = []           # list of {"role": "user"/"assistant", "content": "..."}


# ---------- Step 1: prepare context ----------
def prepare_context(user_message):
    """System prompt + memory block + last 10 turns + the new user message."""
    memory = "Predictions made so far: " + (
        "; ".join(f"{pid} ({p['model']}, {p['inputs']}, result {p['value']})"
                  for pid, p in PREDICTIONS.items()) or "none")
    return (
        [{"role": "system", "content": SYSTEM_PROMPT + "\n" + memory}]
        + history[-MAX_TURNS * 2:]
        + [{"role": "user", "content": user_message}]
    )


# ---------- Step 2: call the model ----------
def call_model(messages):
    """Send the messages + the 5 tool schemas to Llama. Returns Llama's reply message."""
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS, "stream": False}
    try:
        response = httpx.post(OLLAMA_URL, json=payload, timeout=120)
    except httpx.ConnectError:
        raise RuntimeError("Cannot reach Ollama on port 11434. Start it with: ollama serve")
    data = response.json()
    if response.status_code != 200:
        raise RuntimeError(data.get("error", f"Ollama returned {response.status_code}"))
    return data["message"]


# ---------- Helper: run one tool safely ----------
def run_tool(name, arguments):
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        if isinstance(arguments.get("changes"), str):      # model sent changes as text
            arguments["changes"] = json.loads(arguments["changes"])
        return function(**arguments)
    except Exception as e:                                  # wrong/missing arguments etc.
        return {"error": f"Tool failed: {e}"}


def save_log(entry):
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------- Steps 3, 4, 5: handle response, iterate, return ----------
def run_turn(user_message):
    """Returns (final_text, tool_log)."""
    messages = prepare_context(user_message)
    tool_log = []
    final_text = None

    for _ in range(MAX_STEPS):
        reply = call_model(messages)
        messages.append(reply)                      # keep Llama's reply in the context

        tool_calls = reply.get("tool_calls")
        if not tool_calls:                          # Case 1: plain text -> the turn is done
            final_text = reply.get("content", "")
            break

        for call in tool_calls:                     # Case 2: run every tool Llama asked for
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            result = run_tool(name, arguments)
            entry = {"name": name, "args": arguments, "result": result}
            tool_log.append(entry)
            save_log({"time": datetime.now().isoformat(timespec="seconds"), **entry})
            # Step 4: add the tool result to the context, then loop back to call_model
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})

    if final_text is None:
        final_text = "Sorry, I used too many steps and could not finish. Please try again."

    # Step 5: save the conversation and return
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": final_text})
    return final_text, tool_log
