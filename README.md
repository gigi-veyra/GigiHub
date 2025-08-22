# GigiHub (FastAPI, lean)

GigiHub is a minimal FastAPI service that exposes:
- `/` — status
- `/health` — health check
- `/generate` — calls OpenAI (gpt-4o-mini) for text/code
- `/generate_code` — returns code-only (no fences), useful for snippets

## Prereqs
- Python 3.11+
- An OpenAI API key

## Setup
```bash
# install deps
pip install -r requirements.txt

# copy env example and fill in your real key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-... (REQUIRE_AUTH can stay false)
Run (port 8001)
uvicorn main:app --reload --host 0.0.0.0 --port 8001
VS Code task (one-click restart)
Open the Command Palette → “Run Task” → Restart Gigi (8001)
Test (local)
# health
curl -s http://localhost:8001/health

# generate (text/code mixed)
curl -s -X POST http://localhost:8001/generate \
	-H "Content-Type: application/json" \
	-d '{"text":"Write a Python function that reverses a string."}'

# generate_code (code only)
curl -s -X POST http://localhost:8001/generate_code \
	-H "Content-Type: application/json" \
	-d '{"prompt":"Write a Python function that checks if a number is prime."}'
Make the port public (optional)
Open the PORTS panel in Codespaces → set 8001 to Public. Then use the forwarded .github.dev URL in curl:
curl -s https://<your-8001-url>.github.dev/health
Notes:
Keep .env out of git. Only commit .env.example.
Auth is optional: set REQUIRE_AUTH=true and API_KEY=... to require header x-api-key.
# GigiHub