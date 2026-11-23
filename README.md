# Jarvice Python Backend

A Python implementation of the Jarvice agent backend using LangGraph, Groq, Tavily, Wikipedia, and FastAPI.

## Setup

```powershell
cd python
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item ..\.env.example .env
```

Add `GROQ_API_KEY` and `TAVILY_API_KEY` to `.env`.

## Run CLI

```powershell
python cli.py
```

## Run HTTP server

```powershell
uvicorn server:app --reload --host 0.0.0.0 --port 3001
```

## Endpoints

- `GET /health`
- `POST /api/chat`

Chat request example:

```json
{
  "message": "What is the latest AI news?",
  "threadId": "session-1",
  "documentContent": null
}
```
