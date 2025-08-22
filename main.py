import os
import logging
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="GigiHub API", version="1.0.0")

# Enable CORS for all origins (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH") == "true"
client = OpenAI(api_key=OPENAI_API_KEY)

async def auth(x_api_key: str | None = Header(default=None)):
    if REQUIRE_AUTH:
        if not API_KEY:
            raise HTTPException(status_code=500, detail="Server missing API_KEY")
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
@app.get("/health")
def health():
    return {"ok": True}

class Prompt(BaseModel):
    text: str

@app.get("/")
async def read_root():
    return {"status": "ok", "service": "GigiHub API"}

@app.post("/generate")
async def generate(prompt: Prompt, _=Depends(auth)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set or not loaded from .env")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.text}],
            max_tokens=200,
            temperature=0.2,
        )
        return {"response": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

# Local dev entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
