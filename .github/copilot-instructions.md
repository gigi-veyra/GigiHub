# Copilot Instructions for GigiHub

## Project Overview
GigiHub is a Python project. The main entry point is `main.py`. All dependencies are listed in `requirements.txt`.

## Architecture & Structure
- `main.py`: Central script. All core logic is here. No submodules or packages detected.
- `requirements.txt`: Lists required Python packages. Install with `pip install -r requirements.txt`.
- `__pycache__/`: Contains Python bytecode; ignore for development.

## Developer Workflows
- **Run the project:**
  ```bash
  python main.py
  ```
- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- **No explicit test or build scripts found.**

## Conventions & Patterns
- Single-file architecture. All logic is in `main.py`.
- No custom build, test, or deployment scripts.
- No external service integrations or cross-component communication.
- No project-specific naming conventions detected.

## Integration Points
- Only standard Python dependencies (see `requirements.txt`).
- No API keys, environment variables, or external config files detected.

## Example Workflow
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python main.py`

## Key Files
- `main.py`: All main logic
- `requirements.txt`: Dependency list

---
If you add new files, modules, or workflows, update this document to help future AI agents be productive.
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
pip install fastapi uvicorn httpx pydantic
uvicorn main:app --reload --port 8000

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

pip install fastapi uvicorn httpx pydantic
uvicorn main:app --reload --port 8000
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
app = FastAPI(title="Gigi Core", version="0.3.0")

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
        # ~2 messages per turn (user + assistant)
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
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"n8n HTTP error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"n8n request error: {type(e).__name__}: {str(e)}",
        )

# ---- Routes: health & chat ----
@app.get("/health")
async def health():
    return {"ok": True, "env": ENV, "tz": GENERIC_TIMEZONE}

@app.post("/chat")
async def chat(req: ChatRequest):
    session = _get_session(req.session_id)

    # Build message context
    messages: List[Msg] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.extend(list(session))
    if req.history:
        messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    # Call n8n if configured, else echo
    if N8N_WEBHOOK_URL:
        result = await call_n8n({"messages": messages, "temperature": req.temperature})
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

    return {"reply": assistant_text, "session_id": req.session_id, "turns_kept": MAX_TURNS}

# ---- Routes: memory management ----
@app.get("/sessions")
async def list_sessions():
    """List all session IDs currently in memory."""
    return {"sessions": list(memory.keys()), "count": len(memory)}

@app.get("/memory/{session_id}")
async def get_memory(session_id: str):
    """Return the stored messages for a session."""
    if session_id not in memory:
        return {"session_id": session_id, "messages": [], "turns_kept": MAX_TURNS}
    return {
        "session_id": session_id,
        "messages": list(memory[session_id]),
        "turns_kept": MAX_TURNS,
    }

@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear a session's memory."""
    existed = session_id in memory
    memory.pop(session_id, None)
    return {"session_id": session_id, "cleared": True, "existed": existed}

@app.delete("/sessions")
async def clear_all_sessions():
    """Danger: clear ALL sessions."""
    count = len(memory)
    memory.clear()
    return {"cleared_sessions": count}

# ---- Local dev entrypoint ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

uvicorn main:app --reload --port 8000
# main.py
from __future__ import annotations

import os
import sqlite3
import time
from typing import List, Optional, Literal, TypedDict, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

# ---------------- Env ----------------
ENV = os.getenv("ENV", "dev")
GENERIC_TIMEZONE = os.getenv("GENERIC_TIMEZONE", "UTC")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()  # set this to enable header auth
DB_PATH = os.getenv("DB_PATH", "memory.db").strip()
MAX_TURNS = int(os.getenv("MAX_TURNS", "12"))  # number of user+assistant pairs retained

# ---------------- App ----------------
app = FastAPI(title="Gigi Core", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENV != "prod" else os.getenv("ALLOWED_ORIGINS", "").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO if ENV != "prod" else logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("gigi-core")

# ---------------- Types & Schemas ----------------
class Msg(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    system: Optional[str] = "You are Gigi, a warm and helpful assistant."
    history: List[Msg] = []
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    session_id: Optional[str] = "default"

# ---------------- Persistence (SQLite) ----------------
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with _db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                session_id TEXT NOT NULL,
                ts REAL NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
                content TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session_ts ON messages(session_id, ts);
            """
        )
    log.info("DB initialized at %s", DB_PATH)

def ensure_session(session_id: str) -> None:
    with _db() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions(id) VALUES (?)", (session_id,))

def add_message(session_id: str, role: str, content: str) -> None:
    ts = time.time()
    with _db() as conn:
        conn.execute(
            "INSERT INTO messages(session_id, ts, role, content) VALUES (?,?,?,?)",
            (session_id, ts, role, content),
        )
        # keep only last 2*MAX_TURNS messages (user+assistant) plus an optional system
        cur = conn.execute("SELECT COUNT(*) AS c FROM messages WHERE session_id=?", (session_id,))
        total = cur.fetchone()["c"]
        keep = MAX_TURNS * 2 + 1  # +1 to allow a system msg
        if total > keep:
            to_delete = total - keep
            conn.execute(
                """
                DELETE FROM messages
                WHERE rowid IN (
                    SELECT rowid FROM messages
                    WHERE session_id=?
                    ORDER BY ts ASC
                    LIMIT ?
                )
                """,
                (session_id, to_delete),
            )

def get_messages(session_id: str) -> List[Msg]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY ts ASC",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def clear_session(session_id: str) -> Dict[str, Any]:
    with _db() as conn:
        existed = conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is not None
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    return {"session_id": session_id, "cleared": True, "existed": existed}

def list_sessions() -> List[str]:
    with _db() as conn:
        rows = conn.execute("SELECT id FROM sessions ORDER BY created_at DESC").fetchall()
    return [r["id"] for r in rows]

def clear_all_sessions() -> int:
    with _db() as conn:
        conn.execute("DELETE FROM messages")
        deleted = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
        conn.execute("DELETE FROM sessions")
    return deleted

# ---------------- Simple Auth ----------------
def require_key(x_api_key: Optional[str]) -> None:
    if not API_KEY:
        return  # auth disabled in dev if not set
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ---------------- n8n Call ----------------
async def call_n8n(payload: dict) -> dict:
    if not N8N_WEBHOOK_URL:
        raise HTTPException(status_code=500, detail="N8N_WEBHOOK_URL not set")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                N8N_WEBHOOK_URL, json=payload, headers={"content-type": "application/json"}
            )
            resp.raise_for_status()
            if "application/json" in (resp.headers.get("content-type") or ""):
                return resp.json()
            return {"text": resp.text}
    except httpx.HTTPStatusError as e:
        body = e.response.text[:2000]
        log.warning("n8n HTTP error %s: %s", e.response.status_code, body)
        raise HTTPException(status_code=e.response.status_code, detail=f"n8n HTTP error: {body}")
    except Exception as e:
        log.exception("n8n request error")
        raise HTTPException(status_code=500, detail=f"n8n request error: {type(e).__name__}: {e}")

# ---------------- Startup ----------------
@app.on_event("startup")
def _startup():
    init_db()
    log.info("Startup complete")

# ---------------- Routes ----------------
@app.get("/health")
async def health():
    return {"ok": True, "env": ENV, "tz": GENERIC_TIMEZONE, "db": DB_PATH or ":memory:"}

@app.get("/sessions")
async def sessions_list(x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    return {"sessions": list_sessions(), "count": len(list_sessions())}

@app.delete("/sessions")
async def sessions_clear_all(x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    cleared = clear_all_sessions()
    return {"cleared_sessions": cleared}

@app.get("/memory/{session_id}")
async def memory_get(session_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    msgs = get_messages(session_id)
    return {"session_id": session_id, "messages": msgs, "turns_kept": MAX_TURNS}

@app.delete("/memory/{session_id}")
async def memory_clear(session_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    return clear_session(session_id)

@app.post("/chat")
async def chat(req: ChatRequest, x_api_key: Optional[str] = Header(default=None), request: Request = None):
    require_key(x_api_key)

    session_id = req.session_id or "default"
    ensure_session(session_id)

    # Build context
    context: List[Msg] = []
    if req.system:
        context.append({"role": "system", "content": req.system})

    # add persisted history
    context.extend(get_messages(session_id))

    # add transient history from the request if provided
    if req.history:
        context.extend(req.history)

    # add the new user message
    user_msg: Msg = {"role": "user", "content": req.message}
    context.append(user_msg)

    # route to n8n if configured, else dev echo
    if N8N_WEBHOOK_URL:
        result = await call_n8n({"messages": context, "temperature": req.temperature})
        assistant_text = (
            result.get("reply")
            or result.get("text")
            or result.get("message")
            or str(result)
        )
    else:
        assistant_text = f"(dev) Echo: {req.message}"

    # persist latest turn
    add_message(session_id, "user", req.message)
    add_message(session_id, "assistant", assistant_text)

    # structured response
    return {
        "reply": assistant_text,
        "session_id": session_id,
        "turns_kept": MAX_TURNS,
        "used_n8n": bool(N8N_WEBHOOK_URL),
        "ip": request.client.host if request and request.client else None,
    }

# --------------- Local dev entrypoint ---------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
