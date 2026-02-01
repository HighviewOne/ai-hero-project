import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# Load API keys from Streamlit secrets (for cloud deployment) or .env (local)
if hasattr(st, 'secrets'):
    for key in ['GEMINI_API_KEY', 'OPENAI_API_KEY']:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]

# Add project root to path so app imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_NAME, CHUNKS_FILE, EMBEDDINGS_FILE
from app.search_tools import load_search_tool
from app.search_agent import create_agent
from app.logs import log_interaction


# ── Page config ──────────────────────────────────────────────────

st.set_page_config(
    page_title=f"{PROJECT_NAME} Docs Agent",
    page_icon="⚡",
    layout="centered",
)

st.title(f"{PROJECT_NAME} Documentation Agent")
st.caption(f"Ask questions about {PROJECT_NAME} and get answers from the official docs.")


# ── Initialize resources (cached) ───────────────────────────────

@st.cache_resource
def init_search_tool():
    """Load search indexes once and cache across reruns."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_path = os.path.join(project_dir, CHUNKS_FILE)
    embeddings_path = os.path.join(project_dir, EMBEDDINGS_FILE)
    return load_search_tool(chunks_path, embeddings_path)


@st.cache_resource
def init_agent(_search_tool):
    """Create the agent once and cache it."""
    return create_agent(_search_tool)


with st.spinner("Loading search indexes and model..."):
    search_tool = init_search_tool()
    agent = init_agent(search_tool)


# ── Chat history ─────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "message_history" not in st.session_state:
    st.session_state.message_history = None

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ───────────────────────────────────────────────────

if prompt := st.chat_input(f"Ask about {PROJECT_NAME}..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Searching docs and thinking..."):
            result = asyncio.run(
                agent.run(prompt, message_history=st.session_state.message_history)
            )
            response_text = result.response.parts[0].content
            st.session_state.message_history = result.all_messages()

        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Log interaction
    try:
        log_interaction(agent, result, prompt)
    except Exception:
        pass  # Don't break the UI if logging fails
