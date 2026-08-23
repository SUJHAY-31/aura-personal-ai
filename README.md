# AURA

**Autonomous Unified Reasoning Assistant** — a production-oriented, modular AI personal assistant platform inspired by the proactive, context-aware spirit of JARVIS, built for real workflows rather than generic chat.

AURA is **not** a chatbot clone. It is an **agent platform**: pluggable capabilities, explicit permissions, durable memory, planning, automation, and multi-surface clients that share one backend brain.

---

## Vision

| Principle | Intent |
|-----------|--------|
| **Modular agents** | Separate domains (AI, planner, automation, voice, memory) evolve independently behind stable interfaces. |
| **Local-first intelligence** | Ollama and local Qwen models keep sensitive work on your machine; cloud can be added later without rewriting the core. |
| **Clean architecture** | HTTP/API at the edge; domain logic inward; infrastructure (SQLite, LLM clients) at the boundary. |
| **Incremental delivery** | Ship thin vertical slices (health check → memory → tools → voice) with tests and docs at each step. |

---

## Repository layout

```
AURA/
├── backend/app/          # FastAPI service and Python domain modules
├── frontend/             # React dashboard (planned)
├── desktop/              # Electron shell (planned)
├── mobile/               # Flutter app (planned)
├── docs/                 # Architecture and runbooks
├── scripts/              # Dev and deployment helpers
└── tests/                # Backend and integration tests
```

### Backend modules (placeholders)

| Module | Role |
|--------|------|
| `ai/` | LLM orchestration, prompts, Ollama/Qwen integration |
| `api/` | Routers, request/response schemas, versioning |
| `automation/` | Tasks, triggers, and external tool execution |
| `memory/` | Short- and long-term context, retrieval |
| `permissions/` | Capability gates and policy |
| `planner/` | Goal decomposition and step scheduling |
| `voice/` | Speech I/O pipeline |
| `database/` | Persistence, migrations, SQLite access |
| `models/` | Shared domain types and entities |
| `utils/` | Cross-cutting helpers (logging, config) |

---

## Tech stack

- **Backend:** Python, FastAPI, SQLite (initial)
- **Models:** Ollama, local Qwen
- **Clients:** React (dashboard), Electron (desktop), Flutter (mobile)

---

## Quick start (backend skeleton)

From the repository root, with a virtual environment activated:

```bash
pip install fastapi uvicorn
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Verify:

```bash
curl http://127.0.0.1:8000/
```

Expected JSON:

```json
{
  "assistant": "AURA",
  "status": "Running",
  "version": "0.1.0",
  "message": "Welcome to AURA AI"
}
```

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

> Dependencies will be pinned in `requirements.txt` as modules land. The file is intentionally empty at bootstrap.

---

## Status

**Version 0.1.0** — project scaffold only: structure, health endpoint, no business logic.

---

## License

TBD.
