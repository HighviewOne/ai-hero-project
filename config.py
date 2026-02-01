# ── Project Configuration ──
REPO_OWNER = "fastapi"
REPO_NAME = "fastapi"
REPO_BRANCH = "master"
PROJECT_NAME = "FastAPI"  # Display name for UI and prompts

# ── Models ──
LLM_MODEL = "google-gla:gemini-2.0-flash"
EVAL_MODEL = "google-gla:gemini-2.0-flash-lite"
EMBEDDING_MODEL = "multi-qa-distilbert-cos-v1"

# ── Derived file paths (auto-generated from REPO_NAME) ──
DOCS_FILE = f"{REPO_NAME}_docs.json"
CHUNKS_FILE = f"{REPO_NAME}_chunks_sliding.json"
CHUNKS_SECTIONS_FILE = f"{REPO_NAME}_chunks_sections.json"
EMBEDDINGS_FILE = f"{REPO_NAME}_embeddings.npy"
