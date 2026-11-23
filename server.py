import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import build_agent

load_dotenv()

app = FastAPI(title="Jarvice Python Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

agent = build_agent()


class ChatRequest(BaseModel):
    message: str | None = Field(default=None)
    threadId: str = "default-session"
    documentContent: str | None = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "message": "Jaervice backend is running!"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    if not request.message and not request.documentContent:
        raise HTTPException(status_code=400, detail="Message or document is required")

    final_message = request.message or ""
    if request.documentContent:
        excerpt = request.documentContent[:3000]
        final_message = (
            "I've uploaded a document. Here's my query: "
            f'{request.message or "Please summarize this document"}\n\n'
            f"Document content (first 3000 chars):\n{excerpt}"
        )

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": final_message}]},
            {"configurable": {"thread_id": request.threadId}},
        )
        response_text = result["messages"][-1].content
        return {
            "response": response_text,
            "threadId": request.threadId,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": True,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process message: {exc}") from exc
