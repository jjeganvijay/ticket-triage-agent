# AI Usage Note

## Tool Used
- **Model**: Google Gemini 2.5 Flash (`gemini-2.5-flash`)
- **SDK**: `google-generativeai` Python SDK
- **Access**: Via Gemini API (key stored in `.env`, never committed)

## How AI Is Used

The Gemini model is used exclusively for **ticket classification**. For every support ticket, the agent sends a structured prompt to Gemini and expects a JSON response containing:

| Field | Values |
|---|---|
| `category` | Bug, Feature Request, Billing, Other |
| `priority` | P1 Critical, P2 High, P3 Medium, P4 Low |
| `reasoning` | Short 1–2 sentence explanation |

## Prompting Strategy

### Few-Shot Prompting
Six labelled examples are prepended to every prompt to anchor the model's output format and calibration:

1. Double billing → Billing / P2 High
2. iOS crash → Bug / P2 High
3. Dark mode request → Feature Request / P4 Low
4. Account deletion → Bug / P1 Critical
5. PDF export question → Other / P4 Low
6. Payment gateway 500 → Bug / P1 Critical

### Output Parsing
The model is instructed to return **only** a raw JSON object. A regex post-processor strips any accidental markdown code fences before `json.loads()` is called.

## ReAct Agent Loop

The agent wraps Gemini calls inside a **Thought → Action → Observation** loop:
- **Thought**: Reason about the next ticket to process
- **Action**: Call `gemini_classifier.classify(ticket)`
- **Observation**: Record the result or error

This trace is persisted in the `AgentRun` object and returned by the API.

## Limitations & Mitigations

| Limitation | Mitigation |
|---|---|
| Model may hallucinate categories | Strict JSON schema + post-validation via Pydantic enums |
| API rate limits | Sequential processing; retries can be added via `tenacity` |
| Non-determinism | Temperature is left at default; future work: set `temperature=0` |
| API key exposure | Key loaded from `.env`, excluded from git via `.gitignore` |
