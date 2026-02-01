# AI Agents Email Crash Course - Project Memory

## Course Structure
- 7-day crash course on building AI agents from DataTalks.Club
- Lesson plans are in `/home/highview/AIagentsEmailCrashCourse/Day{N}/` as .docx files
- All code lives in `/home/highview/AIagentsEmailCrashCourse/Day1/aihero/project/`
- Python venv at `.venv/`, use `uv` for package management (no pip installed)

## Completed Days

### Day 1: Data Ingestion
- `ingest.py` - Downloads and parses markdown files from GitHub repos using `read_repo_data()`
- Downloaded FastAPI docs (1,042 files) saved to `fastapi_docs.json`

### Day 2: Chunking
- `chunk.py` - Two chunking methods:
  - `sliding_window()` / `chunk_docs_sliding_window()` - fixed size with overlap
  - `split_markdown_by_level()` / `chunk_docs_by_sections()` - splits by markdown headers
- Output: `fastapi_chunks_sliding.json` (6,332 chunks), `fastapi_chunks_sections.json`

### Day 3: Search
- `search.py` - Three search types:
  - `text_search()` - lexical search using `minsearch.Index`
  - `vector_search()` - semantic search using `sentence-transformers` (`multi-qa-distilbert-cos-v1`)
  - `hybrid_search()` - combines both, deduplicates results
- Embeddings cached in `fastapi_embeddings.npy`
- Key dependencies: `minsearch`, `sentence-transformers`

### Day 4: Agents and Tools
- `agent.py` - Pydantic AI agent for answering FastAPI documentation questions
  - Uses `google-gla:gemini-2.0-flash` as the LLM (switched from OpenAI due to quota issues)
  - Two tools: `search_fastapi_docs()` (hybrid search) and `search_text_only()` (keyword search)
  - System prompt instructs agent to always search docs before answering
  - Interactive conversation loop with message history
  - Loads `.env` file via `python-dotenv` for API keys
- `.env` file contains `OPENAI_API_KEY` and `GEMINI_API_KEY`
- Added dependencies: `pydantic-ai`, `openai`, `google-genai`, `python-dotenv`
- Note: `result.response.parts[0].content` is how to extract text from Pydantic AI AgentRunResult (not `result.data`)

### Day 5: Evaluation (COMPLETE)
- `evaluate.py` - Full evaluation pipeline:
  - Logging system: `log_entry()` records agent interactions to `eval_logs.json`
  - LLM-as-a-judge: `eval_agent` evaluates responses on 7 criteria using `EvaluationChecklist` Pydantic model
    - Criteria: instructions_follow, instructions_avoid, answer_relevant, answer_clear, answer_citations, completeness, tool_call_search
  - Test data generation: `question_generator` agent creates 5 FastAPI questions
  - Metrics: computes pass rates per criterion and overall score
  - Output files: `eval_logs.json`, `eval_results.json`
  - Has `retry_with_backoff()` for exponential backoff on rate limits (10s, 20s, 40s)
  - Configurable: NUM_QUESTIONS, DELAY_BETWEEN_CALLS, MAX_RETRIES at top of file
  - Estimated ~11 API calls total (fits within Gemini free tier ~20/day)
- Run: `uv run python evaluate.py`
- **Evaluation results:**
  - Run 1: OVERALL 53.6% (citations 0%, relevance 75%, completeness 75%)
  - Run 2: OVERALL 65.7% (citations 40%, relevance 100%, completeness 100%)
  - Consistent 100%: answer_clear, instructions_avoid (no hallucinations)
  - Weak: instructions_follow (~20-25%), tool_call_search (0%)
  - tool_call_search always 0% — eval limitation: judge can't see internal tool calls in response text
  - Key improvement area: citations (improved with better system prompt emphasis)

### Day 6: Publish Your Agent (COMPLETE)
- Refactored code into modular `app/` package:
  - `app/ingest.py` — data download, parsing, chunking (clean `index_data()` function)
  - `app/search_tools.py` — `SearchTool` class encapsulating text + vector + hybrid search
  - `app/search_agent.py` — `create_agent()` factory, configurable model, updated system prompt with citation emphasis
  - `app/logs.py` — per-interaction JSON logging to `logs/` directory
  - `app/streamlit_app.py` — Streamlit web UI with chat interface, cached resources, message history
- Added dependency: `streamlit`
- Run locally: `uv run streamlit run app/streamlit_app.py` (serves on http://localhost:8501)
- Deploy: `uv export --no-dev > requirements.txt`, push to GitHub, deploy on share.streamlit.io

### Day 7: Share Results (COMPLETE)
- Created comprehensive `README.md` with project overview, installation, usage, structure, evaluation results, and tech stack
- Organized evaluation into `eval/` directory with updated import paths
- Created `.env.example` for onboarding
- Updated `.gitignore` to exclude data files, .env, and logs
- Exported `requirements.txt` for Streamlit Cloud deployment
- **COURSE COMPLETE**

## API Keys & Rate Limits
- OpenAI key in `.env` — account has no credits (insufficient_quota)
- Gemini key in `.env` — billing enabled but subject to free tier limits
- **Gemini free tier (late 2025/2026):** ~20-25 requests/day, 10-15 RPM. Resets at midnight Pacific.
- Agent (agent.py) uses `google-gla:gemini-2.0-flash`
- Evaluation (evaluate.py) uses `google-gla:gemini-2.0-flash-lite`
- If rate limited, wait for daily reset or switch to paid tier / alternative provider (Groq is free with generous limits)

## Technical Reference
- All code lives in `/home/highview/AIagentsEmailCrashCourse/Day1/aihero/project/`
- Python 3.12+, use `uv` for package management (NOT pip)
- Run commands: `uv add <pkg>`, `uv run python <script>.py`
- Virtual env at `.venv/`
- Data flow: ingest.py → chunk.py → search.py → agent.py → evaluate.py
- Modular app: `app/` package (ingest, search_tools, search_agent, logs, streamlit_app)
- Run agent (CLI): `uv run python agent.py`
- Run agent (web): `uv run streamlit run app/streamlit_app.py`
- Run evaluation: `uv run python evaluate.py`
- **Pydantic AI v1.50.0 API notes:**
  - Plain text agent: `result.response.parts[0].content` to get response text
  - Structured output agent (with `output_type`): `result.output` to get parsed Pydantic model
  - Message history: `result.all_messages()`

## Status
- ALL 7 DAYS COMPLETE
- Optional next steps: deploy to Streamlit Cloud, create demo video, share on social media
