import json
import os
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter

LOG_DIR = Path(os.getenv('LOGS_DIRECTORY', 'logs'))


def log_interaction(agent, result, user_query, source="user"):
    """Log an agent interaction to a JSON file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    messages = result.all_messages()
    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

    response_text = result.response.parts[0].content

    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent_name": agent.name,
        "user_query": user_query,
        "response": response_text,
        "messages": dict_messages,
        "source": source,
    }

    filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = LOG_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(entry, f, indent=2, default=str)

    return entry
