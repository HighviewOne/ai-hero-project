import asyncio
import json
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from sentence_transformers import SentenceTransformer
from minsearch import VectorSearch

from search import (
    load_chunks,
    build_text_index,
    text_search,
    hybrid_search,
)

# ── Config ───────────────────────────────────────────────────────

NUM_QUESTIONS = 5          # Keep low to stay within Gemini free tier (~20 req/day)
DELAY_BETWEEN_CALLS = 5    # Seconds between API calls (respect RPM limits)
MAX_RETRIES = 3            # Retry on rate limit errors
MODEL = 'google-gla:gemini-2.0-flash-lite'


# ── Retry helper ─────────────────────────────────────────────────

async def retry_with_backoff(coro_fn, *args, max_retries=MAX_RETRIES, **kwargs):
    """Call an async function with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if '429' in err_str or 'resource_exhausted' in err_str or 'rate' in err_str:
                wait = (2 ** attempt) * 10  # 10s, 20s, 40s
                print(f"    Rate limited (attempt {attempt+1}/{max_retries}). Waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
    # Final attempt without catching
    return await coro_fn(*args, **kwargs)


# ── 1. Load data and build indexes ──────────────────────────────

print("Loading chunks...")
chunks = load_chunks('fastapi_chunks_sliding.json')
print(f"Loaded {len(chunks)} chunks")

print("Building text index...")
text_idx = build_text_index(chunks)

print("Loading embedding model...")
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')

print("Loading cached embeddings...")
embeddings = np.load('fastapi_embeddings.npy')
vec_idx = VectorSearch()
vec_idx.fit(embeddings, chunks)

print("Indexes ready!\n")


# ── 2. FastAPI docs agent (same as agent.py) ────────────────────

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

docs_agent = Agent(
    MODEL,
    system_prompt=SYSTEM_PROMPT,
    name='fastapi_docs_agent',
)


@docs_agent.tool_plain
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


@docs_agent.tool_plain
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


# ── 3. Logging ──────────────────────────────────────────────────

def log_entry(agent, result, user_query, source="generated"):
    """Create a log entry for an agent interaction."""
    messages = result.all_messages()
    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

    response_text = result.response.parts[0].content

    return {
        "timestamp": datetime.now().isoformat(),
        "agent_name": agent.name,
        "system_prompt": SYSTEM_PROMPT,
        "model": MODEL,
        "user_query": user_query,
        "response": response_text,
        "messages": dict_messages,
        "source": source,
    }


def save_logs(logs, filepath="eval_logs.json"):
    """Save logs to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, default=str)
    print(f"Saved {len(logs)} logs to {filepath}")


# ── 4. LLM-as-a-Judge evaluation agent ─────────────────────────

class EvaluationCheck(BaseModel):
    check_name: str
    justification: str
    check_pass: bool


class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck]
    summary: str


EVAL_PROMPT = """\
You are an evaluation agent. You will be given a user question and an agent's \
response about FastAPI documentation.

Evaluate the response on EACH of these criteria. For each, provide a \
justification and whether it passes (true/false):

1. instructions_follow: The agent followed its instructions to search docs before answering.
2. instructions_avoid: The agent did NOT make up information or hallucinate.
3. answer_relevant: The response directly addresses the user's question.
4. answer_clear: The answer is clear, well-structured, and easy to understand.
5. answer_citations: The response references specific parts of the documentation.
6. completeness: The response covers the key aspects of the question.
7. tool_call_search: The agent used a search tool to find information.

Be strict but fair. Base your evaluation on the content provided.\
"""

eval_agent = Agent(
    MODEL,
    system_prompt=EVAL_PROMPT,
    name='eval_agent',
    output_type=EvaluationChecklist,
)


async def evaluate_interaction(user_query, agent_response, messages):
    """Run the evaluation agent on a single interaction."""
    tools_called = []
    for msg in messages:
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'tool_name'):
                    tools_called.append(part.tool_name)

    eval_input = f"""## User Question
{user_query}

## Agent Response
{agent_response}

## Tools Called
{', '.join(tools_called) if tools_called else 'None'}
"""
    result = await retry_with_backoff(eval_agent.run, eval_input)
    return result.output


# ── 5. Test question generation ─────────────────────────────────

class QuestionsList(BaseModel):
    questions: list[str]


QUESTION_GEN_PROMPT = f"""\
You are a test data generator. Generate realistic questions that a developer \
might ask about the FastAPI web framework. Questions should cover a variety \
of topics: routing, dependencies, authentication, middleware, testing, \
deployment, databases, WebSockets, background tasks, etc.

Generate exactly {NUM_QUESTIONS} diverse questions. Make them specific and practical.\
"""

question_generator = Agent(
    MODEL,
    system_prompt=QUESTION_GEN_PROMPT,
    name='question_generator',
    output_type=QuestionsList,
)


async def generate_test_questions():
    """Generate test questions using the LLM."""
    print(f"Generating {NUM_QUESTIONS} test questions...")
    result = await retry_with_backoff(
        question_generator.run,
        f"Generate {NUM_QUESTIONS} FastAPI questions."
    )
    questions = result.output.questions
    print(f"Generated {len(questions)} questions")
    for i, q in enumerate(questions):
        print(f"  {i+1}. {q}")
    return questions


# ── 6. Run full evaluation pipeline ────────────────────────────

async def run_evaluation():
    print(f"Configuration: {NUM_QUESTIONS} questions, {DELAY_BETWEEN_CALLS}s delay, model={MODEL}")
    print(f"Estimated API calls: ~{1 + NUM_QUESTIONS + NUM_QUESTIONS} (within Gemini free tier)\n")

    # Step 1: Generate test questions
    questions = await generate_test_questions()
    await asyncio.sleep(DELAY_BETWEEN_CALLS)

    # Step 2: Run each question through the docs agent and log
    logs = []
    print("\nRunning questions through the agent...")
    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Q: {question}")
        try:
            result = await retry_with_backoff(docs_agent.run, question)
            response_text = result.response.parts[0].content
            print(f"  A: {response_text[:150]}...")

            log = log_entry(docs_agent, result, question, source="generated")
            logs.append(log)
        except Exception as e:
            print(f"  ERROR: {e}")
            logs.append({
                "timestamp": datetime.now().isoformat(),
                "user_query": question,
                "error": str(e),
                "source": "generated",
            })
        await asyncio.sleep(DELAY_BETWEEN_CALLS)

    save_logs(logs)

    # Step 3: Evaluate each interaction
    print("\nEvaluating agent responses...")
    evaluations = []
    for i, log in enumerate(logs):
        if "error" in log:
            print(f"  [{i+1}] Skipping (had error)")
            continue

        print(f"  [{i+1}/{len(logs)}] Evaluating: {log['user_query'][:60]}...")
        try:
            checklist = await evaluate_interaction(
                log['user_query'],
                log['response'],
                log.get('messages', []),
            )
            evaluations.append({
                "question": log['user_query'],
                "checklist": [
                    {
                        "check_name": c.check_name,
                        "justification": c.justification,
                        "check_pass": c.check_pass,
                    }
                    for c in checklist.checklist
                ],
                "summary": checklist.summary,
            })
        except Exception as e:
            print(f"    ERROR evaluating: {e}")
        await asyncio.sleep(DELAY_BETWEEN_CALLS)

    # Save evaluations
    with open("eval_results.json", 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, indent=2)
    print(f"\nSaved {len(evaluations)} evaluations to eval_results.json")

    # Step 4: Compute metrics
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    if not evaluations:
        print("No evaluations to analyze.")
        return

    check_stats = {}
    for ev in evaluations:
        for check in ev['checklist']:
            name = check['check_name']
            if name not in check_stats:
                check_stats[name] = {'pass': 0, 'total': 0}
            check_stats[name]['total'] += 1
            if check['check_pass']:
                check_stats[name]['pass'] += 1

    print(f"\nResults across {len(evaluations)} evaluated interactions:\n")
    print(f"  {'Check':<25} {'Pass Rate':>10} {'Passed':>8} {'Total':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*8}")

    total_pass = 0
    total_checks = 0
    for name, stats in sorted(check_stats.items()):
        rate = stats['pass'] / stats['total'] * 100
        total_pass += stats['pass']
        total_checks += stats['total']
        print(f"  {name:<25} {rate:>9.1f}% {stats['pass']:>8} {stats['total']:>8}")

    overall = total_pass / total_checks * 100 if total_checks > 0 else 0
    print(f"\n  {'OVERALL':<25} {overall:>9.1f}% {total_pass:>8} {total_checks:>8}")
    print()


if __name__ == "__main__":
    asyncio.run(run_evaluation())
