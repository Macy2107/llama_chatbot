"""API layer: HTTP routes only."""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agent import run_turn
router = APIRouter(prefix="/api")

class ChatRequest(BaseModel):
    message: str = ""


def sse(data):
    """Format one Server-Sent Event."""
    return f"data: {json.dumps(data)}\n\n"


@router.post("/chat-stream")
def chat_stream(body: ChatRequest):
    """Receive user input, let the agent work, and send the result back via SSE."""
    if not body.message.strip():
        return {"error": "Message content cannot be empty."}

    def events():
        try:
            text, tool_log = run_turn(body.message)
        except Exception as e:
            yield sse({"error": str(e)})
            return
        for item in tool_log:               # show which tools were used
            yield sse({"tool": item})
        yield sse({"token": text})          # the final answer
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")