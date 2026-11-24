"""FastAPI application: wires the API routes and serves the frontend."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.chat_routes import router

app = FastAPI(title="Llama SSE Chatbot API")

# CORS so the frontend can also be opened directly as a file
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

FRONTEND_FILE = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/")
async def index():
    """Serve the test page, so the browser and the API share one origin."""
    return FileResponse(FRONTEND_FILE)
