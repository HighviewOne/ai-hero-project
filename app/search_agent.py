from pydantic_ai import Agent

from config import LLM_MODEL, PROJECT_NAME
from app.search_tools import SearchTool


SYSTEM_PROMPT = f"""\
You are a helpful {PROJECT_NAME} documentation assistant. Your job is to answer \
questions about the {PROJECT_NAME} web framework accurately and thoroughly.

When a user asks a question:
1. Always search the documentation first before answering. Do not rely on \
your own knowledge alone.
2. If the first search doesn't return useful results, try rephrasing the \
query or searching with different keywords.
3. You may perform multiple searches to gather comprehensive information.
4. Base your answers on the search results. Cite the source filenames when possible.
5. If the documentation doesn't cover the topic, say so honestly.

Keep answers concise but complete. Use code examples from the docs when relevant. \
Always mention which documentation file the information comes from.\
"""


def create_agent(search_tool: SearchTool, model: str = LLM_MODEL):
    """Create a Pydantic AI agent with the search tool."""
    agent = Agent(
        model,
        system_prompt=SYSTEM_PROMPT,
        name='docs_agent',
    )

    @agent.tool_plain
    def search_docs(query: str) -> str:
        f"""Search the {PROJECT_NAME} documentation using hybrid search (text + vector).

        Args:
            query: The search query about {PROJECT_NAME}.

        Returns:
            Matching documentation excerpts.
        """
        return search_tool.search(query, num_results=5)

    @agent.tool_plain
    def search_text_only(query: str) -> str:
        f"""Search {PROJECT_NAME} docs using keyword/text search. Good for exact terms.

        Args:
            query: The keyword search query.

        Returns:
            Matching documentation excerpts.
        """
        results = search_tool.text_search(query, num_results=5)
        if not results:
            return "No results found."
        output = []
        for i, r in enumerate(results):
            preview = r['chunk'][:1500]
            output.append(f"[{i+1}] Source: {r['filename']}\n{preview}")
        return "\n\n---\n\n".join(output)

    return agent
