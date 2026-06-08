# Ticket Triage Agent 

An **AI-powered support ticket triage system** built with Google Gemini / Ollama, FastAPI, and SQLite. The agent reads JSON support tickets, classifies them by category and priority using a ReAct-style loop with few-shot prompting, and persists results to a database and CSV file.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Ticket Triage Agent                      │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  JSON Files │───▶│TicketLoader  │───▶│  ReAct Agent  │  │
│  │(data/sample │    │(src/loader/) │    │ (src/agent/)  │  │
│  │  _tickets/) │    └──────────────┘    └───────┬───────┘  │
│  └─────────────┘                                │          │
│                                                 ▼          │
│                                     ┌───────────────────┐  │
│                                     │  LLM Classifier   │  │
│                                     │(src/classifier/)  │  │
│                                     │  Few-Shot Prompt  │  │
│                                     │ Gemini / Ollama   │  │
│                                     └─────────┬─────────┘  │
│                                               │            │
│                           ┌───────────────────┼──────────┐ │
│                           ▼                   ▼          │ │
│                  ┌──────────────┐   ┌──────────────────┐ │ │
│                  │  CSV Export  │   │  SQLite Database  │ │ │
│                  │(output/      │   │(database/        │ │ │
│                  │ results.csv) │   │ tickets.db)       │ │ │
│                  └──────────────┘   └──────────────────┘ │ │
│                           │                              │ │
│                           └──────────────────────────────┘ │
│                                        ▲                   │
│                               ┌────────┴────────┐          │
│                               │   FastAPI REST  │          │
│                               │  (src/api/)     │          │
│                               │  POST /triage   │          │
│                               │  GET /results   │          │
│                               └─────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### ReAct Agent Loop

```
Thought: "Discover tickets in folder"
  Action: load_tickets(folder)
    Observation: "Found 5 ticket(s): [T001, T002, ...]"

Thought: "Classify ticket T001"
  Action: classifier.classify(T001)   # gemini or ollama, based on LLM_PROVIDER
    Observation: "→ Bug / P2 High. Reasoning: ..."

... (repeat for each ticket)

Thought: "All tickets processed. Triage complete."
  Action: return AgentRun(results)
    Observation: "Processed 5/5 tickets (0 errors)."
```

---

## Project Structure

```
ticket-triage-agent/
├── src/
│   ├── models/          # Pydantic data models (Ticket, TriageResult, AgentRun)
│   ├── loader/          # JSON ticket file loader
│   ├── classifier/      # LLM classifier (Gemini/Ollama) with few-shot prompting
│   ├── agent/           # ReAct-style agent loop
│   ├── export/          # CSV exporter
│   ├── database/        # SQLite persistence layer
│   └── api/             # FastAPI app + route handlers
├── tests/               # pytest unit tests (all mocked — no API calls)
├── data/
│   └── sample_tickets/  # Sample JSON ticket files (T001–T005)
├── database/            # SQLite DB (auto-created, git-ignored)
├── output/              # CSV output (auto-created, git-ignored)
├── resumes/             # Team member resumes (PDF)
│   ├── member1.pdf
│   ├── member2.pdf
│   ├── member3.pdf
│   └── member4.pdf
├── video/               # Demo video reference
│   └── demo_video_link.txt
├── docs/
│   └── AI_USAGE.md      # AI usage documentation
├── main.py              # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone & enter the repo

```bash
git clone https://github.com/jjeganvijay/ticket-triage-agent.git
cd ticket-triage-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set configuration variables (see .env.example)
```

By default, the agent uses Google Gemini. To configure:
* **Gemini**: Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env`. Get a key at [AI Studio](https://aistudio.google.com/app/apikey).
* **Ollama**: Start your local Ollama server, pull the model (`ollama pull llama3.2`), and configure `.env`:
  ```ini
  LLM_PROVIDER=ollama
  OLLAMA_MODEL=llama3.2
  OLLAMA_BASE_URL=http://localhost:11434
  ```

---

## Running

### CLI (end-to-end triage)

```bash
python main.py
```

Options:

```bash
python main.py --folder data/sample_tickets --csv output/results.csv --db database/tickets.db --quiet
```

### FastAPI server

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

#### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/v1/triage` | Run triage on a tickets folder |
| `GET` | `/api/v1/results` | List all stored triage results |
| `GET` | `/api/v1/results/{ticket_id}` | Get result for a specific ticket |
| `GET` | `/api/v1/results/{ticket_id}/status` | Lightweight category + priority only |

---

## Running Tests

```bash
pytest tests/ -v
```

All tests are fully mocked — no real LLM API calls are made.

---

## Ticket Format

Place JSON files in `data/sample_tickets/`. Each file must have:

```json
{
  "ticket_id": "T001",
  "description": "I was charged twice for my subscription."
}
```

An optional `metadata` field is also supported.

---

## Output

### CSV (`output/results.csv`)

```
ticket_id,description,category,priority,reasoning,processed_at
T001,I was charged twice...,Billing,P2 High,Duplicate billing...,2024-01-01T12:00:00
```

### Database (`database/tickets.db`)

Table: `triage_results`

| Column | Type |
|---|---|
| id | INTEGER PK |
| ticket_id | TEXT UNIQUE |
| description | TEXT |
| category | TEXT |
| priority | TEXT |
| reasoning | TEXT |
| processed_at | TEXT (ISO 8601) |

---

## Classification Schema

| Category | When used |
|---|---|
| Bug | Software defects, crashes, data corruption |
| Feature Request | New features, UI enhancements |
| Billing | Payment, subscription, refund issues |
| Other | General questions, how-to inquiries |

| Priority | When used |
|---|---|
| P1 Critical | Production outage, data loss, security breach |
| P2 High | Major feature broken, no workaround |
| P3 Medium | Minor bug, workaround available |
| P4 Low | Cosmetic, nice-to-have, general questions |

---

## Assumptions & Limitations

### Assumptions
* **Input tickets** are valid JSON files and properly formatted.
* **Categories** are strictly limited to `Bug`, `Feature Request`, `Billing`, and `Other`.
* **Priorities** are strictly limited to `P1`, `P2`, `P3`, and `P4`.
* **Ollama Usage:** When using the Ollama provider, the Ollama server must be installed and actively running on the host machine.

### Limitations
* **Classification Quality:** Classification accuracy heavily depends on the underlying LLM's capabilities and reasoning constraints.
* **Hardware Constraints:** Local Ollama execution performance is tied directly to available system resources (CPU/RAM/GPU).
* **Environment:** The project is intended for mock/demo workloads and not enterprise-scale transaction throughput without queuing systems.
* **Consistency:** Results and reasoning verbosity may vary between Gemini and Ollama models due to different architectures.

---

## Demo Video

**Demo Video:**

- 🎬 **Loom** (recommended): [Watch the Demo](https://www.loom.com/share/4240b436a4f34cfab02ab91fbd2ff32d)
- 📁 **Google Drive** (if Loom is unavailable): [Watch on Drive](https://drive.google.com/file/d/1U91WTnQRVKzadkqucm5n88GFlhgmaKCm/view?usp=sharing)

---

## AI Usage

See [docs/AI_USAGE.md](docs/AI_USAGE.md) for full details on the Gemini/Ollama integration, few-shot prompting strategy, and limitations.

---

## License

MIT
