# AI Usage Note

## Tool Used
- **Model**: Google Gemini 2.5 Flash (`gemini-2.5-flash`) or local Ollama model (default: `llama3.2`) — selected via `LLM_PROVIDER` env variable
- **SDK**: `google-genai` Python SDK (Gemini) and `httpx` (Ollama REST API)
- **Access**: Via Gemini API (key stored in `.env`, never committed) or local Ollama server (default: `http://localhost:11434`)

## What AI Helped With

The model (resolved dynamically via `LLM_PROVIDER` environment variable) is used exclusively for **ticket classification**. For every support ticket, the agent sends a structured prompt to the selected LLM and expects a JSON response containing:

| Field | Values |
|---|---|
| `category` | Bug, Feature Request, Billing, Other |
| `priority` | P1 Critical, P2 High, P3 Medium, P4 Low |
| `reasoning` | Short 1–2 sentence explanation |

AI coding assistants also helped accelerate development by:
- Scaffolding the initial project structure and Pydantic models
- Suggesting the ReAct agent loop design pattern
- Drafting the few-shot prompt examples and output parsing logic
- Writing boilerplate FastAPI routes and pytest fixtures

## What AI Got Wrong

Not all AI-generated suggestions were usable without correction. Key issues encountered:

1. **Outdated SDK method names**: The initial AI suggestion used the older `google.generativeai` SDK. The import path and client initialization had to be manually corrected to match the current `google-genai` library API.

2. **Incorrect JSON parsing approach**: The AI initially suggested using a simple `response.text` extraction. In practice, the model sometimes wraps output in markdown code fences (` ```json ... ``` `), requiring a regex post-processor to strip those fences before calling `json.loads()`.

3. **Missing Pydantic validation step**: AI-generated code did not include validation of the model response against the `Category` and `Priority` enums. Without this, invalid model outputs would pass silently. A validation layer using `Category(result["category"])` was added manually.

4. **Ollama retry logic**: The first draft of `OllamaClassifier` had no error handling for connection timeouts. Exponential backoff retry logic had to be added manually after observing failures when the local Ollama server was slow to respond.

5. **Test fixtures needed manual tuning**: AI-generated `pytest` mocks did not correctly patch the module path for the Gemini and Ollama clients, causing import-time failures. All fixtures were debugged and corrected by hand.

> **Note**: All AI-generated code was reviewed, tested, and validated before inclusion. No code was committed without passing all 26 unit tests.

## Prompting Strategy

### Key Development Prompts

The following prompts were critical during development:

**Classifier System Prompt** (used in production):
```
You are a support ticket classifier. Classify the following ticket and return ONLY a raw JSON object.

Categories: Bug, Feature Request, Billing, Other
Priorities: P1 Critical, P2 High, P3 Medium, P4 Low

Format:
{"category": "...", "priority": "...", "reasoning": "..."}
```

**Few-Shot Prompting** — Six labelled examples prepended to every prompt:
1. Double billing → Billing / P2 High
2. iOS crash → Bug / P2 High
3. Dark mode request → Feature Request / P4 Low
4. Account deletion → Bug / P1 Critical
5. PDF export question → Other / P4 Low
6. Payment gateway 500 → Bug / P1 Critical

**Development Scaffolding Prompt**:
```
Build a ReAct-style agent loop in Python that: reads JSON tickets from a folder,
classifies each using an LLM, persists results to SQLite and CSV, and exposes
results via a FastAPI REST API. Use Pydantic for all data models.
```

### Output Parsing
The model is instructed to return **only** a raw JSON object. A regex post-processor strips any accidental markdown code fences before `json.loads()` is called.

## ReAct Agent Loop

The agent wraps LLM calls inside a **Thought → Action → Observation** loop:
- **Thought**: Reason about the next ticket to process
- **Action**: Call `gemini_classifier.classify(ticket)` or `ollama_classifier.classify(ticket)` (resolved dynamically based on provider)
- **Observation**: Record the result or error

This trace is persisted in the `AgentRun` object and returned by the API.

## Limitations & Mitigations

| Limitation | Mitigation |
|---|---|
| Model may hallucinate categories | Strict JSON schema + post-validation via Pydantic enums |
| API rate limits | Sequential processing; retries can be added via `tenacity` |
| Non-determinism | Temperature is left at default; future work: set `temperature=0` |
| API key exposure | Key loaded from `.env`, excluded from git via `.gitignore` |
