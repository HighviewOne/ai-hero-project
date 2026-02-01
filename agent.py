import asyncio
import json

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic_ai import Agent, RunContext

from search import (
    load_chunks,
    build_text_index,
    build_vector_index,
    text_search,
    vector_search,
    hybrid_search,
    print_results,
)

# --- Load data and build indexes at startup ---

print("Loading chunks...")
chunks = load_chunks('fastapi_chunks_sliding.json')
print(f"Loaded {len(chunks)} chunks")

print("Building text index...")
text_idx = build_text_index(chunks)

print("Loading embedding model...")
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')

print("Loading cached embeddings...")
embeddings = np.load('fastapi_embeddings.npy')

from minsearch import VectorSearch
vec_idx = VectorSearch()
vec_idx.fit(embeddings, chunks)

print("Indexes ready!\n")


# --- Define the Pydantic AI agent ---

SYSTEM_PROMPT = """\
You are a helpful FastAPI documentation assistant. Your job is to answer \
questions about the FastAPI web framework accurately and thoroughly.

When a user asks a question:
1. Always search the documentation first before answering. Do not rely on \
your own knowledge alone.
2. If the first search doesn't return useful results, try rephrasing the \
query or searching with different keywords.
3. You may perform multiple searches to gather comprehensive information.
4. Base your answers on the search results. Cite specific details from the docs.
5. If the documentation doesn't cover the topic, say so honestly.

Keep answers concise but complete. Use code examples from the docs when relevant.\
"""

agent = Agent(
    'google-gla:gemini-2.0-flash',
    system_prompt=SYSTEM_PROMPT,
)


@agent.tool_plain
def search_fastapi_docs(query: str) -> str:
    """Search the FastAPI documentation using hybrid search (text + vector).

    Args:
        query: The search query about FastAPI.

    Returns:
        Matching documentation excerpts.
    """
    results = hybrid_search(query, text_idx, vec_idx, embedding_model, num_results=5)
    if not results:
        return "No results found."

    output = []
    for i, r in enumerate(results):
        preview = r['chunk'][:1500]
        output.append(f"[{i+1}] Source: {r['filename']}\n{preview}")
    return "\n\n---\n\n".join(output)


@agent.tool_plain
def search_text_only(query: str) -> str:
    """Search FastAPI docs using keyword/text search. Good for exact terms.

    Args:
        query: The keyword search query.

    Returns:
        Matching documentation excerpts.
    """
    results = text_search(query, text_idx, num_results=5)
    if not results:
        return "No results found."

    output = []
    for i, r in enumerate(results):
        preview = r['chunk'][:1500]
        output.append(f"[{i+1}] Source: {r['filename']}\n{preview}")
    return "\n\n---\n\n".join(output)


# --- Interactive loop ---

async def main():
    print("=" * 60)
    print("FastAPI Documentation Agent (Pydantic AI)")
    print("Type your questions about FastAPI. Type 'quit' to exit.")
    print("=" * 60)

    message_history = None

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        result = await agent.run(user_input, message_history=message_history)
        message_history = result.all_messages()

        print(f"\nAgent: {result.response.parts[0].content}")


if __name__ == "__main__":
    asyncio.run(main())
