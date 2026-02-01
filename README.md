# FastAPI Documentation Agent

An AI-powered agent that answers questions about the FastAPI web framework by searching the official documentation. Built with Pydantic AI, sentence-transformers, and Streamlit.

## Overview

This project creates an intelligent assistant that can answer developer questions about FastAPI by:

1. **Ingesting** 1,042 markdown files from the FastAPI GitHub repository
2. **Chunking** documents into 6,332 searchable segments using a sliding window approach
3. **Indexing** with both text (keyword) and vector (semantic) search
4. **Answering** questions using an LLM agent that searches the docs before responding
5. **Evaluating** agent quality with an automated LLM-as-a-judge pipeline

The agent uses hybrid search (combining keyword and semantic search) to find relevant documentation, then generates accurate answers grounded in the actual docs.

## Installation

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/) package manager

```bash
# Clone the repository
git clone <your-repo-url>
cd project

# Install dependencies
uv sync

# Set up API keys
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### API Keys

The agent uses Google Gemini. Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```
GEMINI_API_KEY=your-key-here
```

## Usage

### Prepare the data (first time only)

```bash
# 1. Download FastAPI docs
uv run python ingest.py

# 2. Chunk the documents
uv run python chunk.py

# 3. Build search indexes and embeddings
uv run python search.py
```

### Run the agent (CLI)

```bash
uv run python agent.py
```

```
You: How do I add authentication to FastAPI?
Agent: FastAPI provides built-in tools for authentication including OAuth2...
```

### Run the agent (Web UI)

```bash
uv run streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

### Run the evaluation

```bash
uv run python eval/evaluate.py
```

## Project Structure

```
project/
├── ingest.py                  # Day 1: Download docs from GitHub
├── chunk.py                   # Day 2: Chunk documents
├── search.py                  # Day 3: Text, vector, and hybrid search
├── agent.py                   # Day 4: Interactive CLI agent
├── evaluate.py                # Day 5: Evaluation pipeline (original)
├── app/                       # Day 6: Modular app package
│   ├── ingest.py              #   Data ingestion & chunking
│   ├── search_tools.py        #   SearchTool class
│   ├── search_agent.py        #   Agent factory
│   ├── logs.py                #   Interaction logging
│   └── streamlit_app.py       #   Streamlit web UI
├── eval/                      # Day 7: Organized evaluation
│   ├── evaluate.py            #   Evaluation pipeline
│   ├── eval_logs.json         #   Agent interaction logs
│   └── eval_results.json      #   Evaluation results
├── fastapi_docs.json          # Raw downloaded docs (1,042 files)
├── fastapi_chunks_sliding.json # Chunked docs (6,332 chunks)
├── fastapi_embeddings.npy     # Cached vector embeddings
├── .env                       # API keys (not committed)
└── pyproject.toml             # Dependencies
```

## Features

- **Hybrid search** combining keyword (minsearch) and semantic (sentence-transformers) search
- **Two search tools** available to the agent: hybrid search and text-only search
- **Conversation memory** across multiple turns
- **Streamlit web UI** with chat interface and cached resource loading
- **Automated evaluation** using LLM-as-a-judge with 7 quality criteria
- **Rate-limit resilient** with exponential backoff retry logic

## Evaluation

The evaluation pipeline generates test questions, runs them through the agent, and evaluates responses on 7 criteria:

| Criterion | Description | Pass Rate |
|-----------|-------------|-----------|
| answer_clear | Answer is clear and well-structured | 100% |
| instructions_avoid | No hallucination or made-up info | 100% |
| answer_relevant | Response addresses the question | 100% |
| completeness | Covers key aspects of the question | 100% |
| answer_citations | References specific documentation | 40% |
| instructions_follow | Followed search-first instructions | 20% |
| **OVERALL** | | **65.7%** |

Key findings:
- The agent produces clear, relevant, and complete answers with no hallucination
- Citation of specific doc files is an area for improvement
- The system prompt can be further tuned to improve citation behavior

## Tech Stack

- **Python 3.12** with [uv](https://docs.astral.sh/uv/) for dependency management
- **Pydantic AI** for agent framework and tool calling
- **Google Gemini** (gemini-2.0-flash) as the LLM
- **sentence-transformers** (multi-qa-distilbert-cos-v1) for semantic embeddings
- **minsearch** for text and vector indexing
- **Streamlit** for web UI
- **NumPy** for embedding storage

## Credits

Built as part of the [AI Agents Crash Course](https://alexeygrigorev.com/aihero/) by DataTalks.Club.
