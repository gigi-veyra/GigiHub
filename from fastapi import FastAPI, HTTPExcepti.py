from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional, TypedDict
from openai import OpenAI
import os

app = FastAPI(title="Gigi Core")

class Message(BaseModel):
    text: str

@app.get("/")
def root():
    return {"ok": True, "service": "gigi", "env": os.getenv("GENERIC_TIMEZONE", "UTC")}

@app.post("/echo")
def echo(msg: Message):
    return {"reply": msg.text}

client = OpenAI()  # reads OPENAI_API_KEY from environment

class Msg(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    system: Optional[str] = "You are Gigi, a warm and helpful assistant."
    history: List[Msg] = []
    temperature: float = 0.7

@app.post("/chat")
def chat(req: ChatRequest):
    messages: List[Msg] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=req.temperature,
        )
        return {"reply": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")
    # main.py
from __future__ import annotations

import os
from typing import List, Optional, Literal, TypedDict, Deque, Dict
from collections import deque

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Env ----
ENV = os.getenv("ENV", "dev")
GENERIC_TIMEZONE = os.getenv("GENERIC_TIMEZONE", "UTC")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").strip()

# ---- App ----
app = FastAPI(title="Gigi Core", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Types & Schemas ----
class Msg(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    system: Optional[str] = "You are Gigi, a warm and helpful assistant."
    history: List[Msg] = []
    temperature: float = 0.7
    session_id: Optional[str] = "default"

# In-memory convo memory
MAX_TURNS = 12  # pairs of user/assistant turns kept
Memory = Dict[str, Deque[Msg]]
memory: Memory = {}

def _get_session(session_id: Optional[str]) -> Deque[Msg]:
    sid = session_id or "default"
    if sid not in memory:
        # keep roughly 2 messages per turn (user + assistant)
        memory[sid] = deque(maxlen=MAX_TURNS * 2)
    return memory[sid]

# ---- n8n Call ----
async def call_n8n(payload: dict) -> dict:
    if not N8N_WEBHOOK_URL:
        raise HTTPException(status_code=500, detail="N8N_WEBHOOK_URL not set")
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                N8N_WEBHOOK_URL,
                json=payload,
                headers={"content-type": "application/json"},
            )
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                return resp.json()
            return {"text": resp.text}
    except httpx.HTTPStatusError as e:
        # Surface n8n’s HTTP status and body
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"n8n HTTP error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"n8n request error: {type(e).__name__}: {str(e)}",
        )

# ---- Routes ----
@app.get("/health")
async def health():
    return {"ok": True, "env": ENV, "tz": GENERIC_TIMEZONE}

@app.post("/chat")
async def chat(req: ChatRequest):
    session = _get_session(req.session_id)

    # Build message context: system + previous memory + provided history + new user msg
    messages: List[Msg] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.extend(list(session))
    if req.history:
        messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    # Call n8n if configured. Otherwise echo for dev.
    if N8N_WEBHOOK_URL:
        result = await call_n8n(
            {"messages": messages, "temperature": req.temperature}
        )
        assistant_text = (
            result.get("reply")
            or result.get("text")
            or result.get("message")
            or str(result)
        )
    else:
        assistant_text = f"(dev) Echo: {req.message}"

    # Update memory
    session.append({"role": "user", "content": req.message})
    session.append({"role": "assistant", "content": assistant_text})

    return {
        "reply": assistant_text,
        "session_id": req.session_id,
        "turns_kept": MAX_TURNS,
    }

# Optional: allow `python main.py` for quick local runs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
