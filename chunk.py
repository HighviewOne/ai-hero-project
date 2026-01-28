import json
import re


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")
    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i:i + size]
        result.append({'start': i, 'chunk': chunk})
        if i + size >= n:
            break
    return result


def split_markdown_by_level(text, level=2):
    header_pattern = r'^(#{' + str(level) + r'} )(.+)$'
    pattern = re.compile(header_pattern, re.MULTILINE)
    parts = pattern.split(text)

    sections = []
    for i in range(1, len(parts), 3):
        header = parts[i] + parts[i + 1]
        header = header.strip()
        content = ""
        if i + 2 < len(parts):
            content = parts[i + 2].strip()
        if content:
            section = f'{header}\n\n{content}'
        else:
            section = header
        sections.append(section)

    return sections


def chunk_docs_sliding_window(docs, size=2000, step=1000):
    chunks = []
    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content', '')
        if not doc_content:
            continue
        windows = sliding_window(doc_content, size, step)
        for window in windows:
            window.update(doc_copy)
        chunks.extend(windows)
    return chunks


def chunk_docs_by_sections(docs, level=2):
    chunks = []
    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content', '')
        if not doc_content:
            continue
        sections = split_markdown_by_level(doc_content, level=level)
        if not sections:
            # No headers found at this level, keep whole document
            section_doc = doc_copy.copy()
            section_doc['section'] = doc_content
            chunks.append(section_doc)
            continue
        for section in sections:
            section_doc = doc_copy.copy()
            section_doc['section'] = section
            chunks.append(section_doc)
    return chunks


if __name__ == "__main__":
    # Load FastAPI docs from Day 1
    with open('fastapi_docs.json', 'r', encoding='utf-8') as f:
        fastapi_docs = json.load(f)

    print(f"Total documents loaded: {len(fastapi_docs)}")

    # --- Method 1: Sliding Window ---
    sw_chunks = chunk_docs_sliding_window(fastapi_docs, size=2000, step=1000)
    sw_chunks_count = len(sw_chunks)
    print(f"\n--- Method 1: Sliding Window (size=2000, step=1000) ---")
    print(f"Chunks: {sw_chunks_count}")
    print(f"Sample chunk (first 200 chars):")
    print(f"  File: {sw_chunks[0].get('filename', 'N/A')}")
    print(f"  Start: {sw_chunks[0].get('start', 'N/A')}")
    print(f"  Text: {sw_chunks[0]['chunk'][:200]}...")

    # --- Method 2: Section-based ---
    sec_chunks = chunk_docs_by_sections(fastapi_docs, level=2)
    sec_count = len(sec_chunks)
    print(f"\n--- Method 2: Section-based (level 2 headers) ---")
    print(f"Chunks: {sec_count}")
    print(f"Sample chunk (first 200 chars):")
    print(f"  File: {sec_chunks[0].get('filename', 'N/A')}")
    print(f"  Text: {sec_chunks[0]['section'][:200]}...")

    # Save both results
    with open('fastapi_chunks_sliding.json', 'w', encoding='utf-8') as f:
        json.dump(sw_chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved sliding window chunks to fastapi_chunks_sliding.json")

    with open('fastapi_chunks_sections.json', 'w', encoding='utf-8') as f:
        json.dump(sec_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved section chunks to fastapi_chunks_sections.json")

    # --- Comparison ---
    content_lengths = [len(doc.get('content', '')) for doc in fastapi_docs]
    avg_doc = sum(content_lengths) / len(content_lengths) if content_lengths else 0
    sw_lengths = [len(c['chunk']) for c in sw_chunks]
    sec_lengths = [len(c['section']) for c in sec_chunks]

    print(f"\n--- Comparison ---")
    print(f"{'Metric':<30} {'Original':<15} {'Sliding':<15} {'Sections':<15}")
    print(f"{'Count':<30} {len(fastapi_docs):<15} {len(sw_chunks):<15} {len(sec_chunks):<15}")
    print(f"{'Avg length (chars)':<30} {avg_doc:<15.0f} {sum(sw_lengths)/len(sw_lengths):<15.0f} {sum(sec_lengths)/len(sec_lengths):<15.0f}")
    print(f"{'Max length (chars)':<30} {max(content_lengths):<15} {max(sw_lengths):<15} {max(sec_lengths):<15}")
    print(f"{'Min length (chars)':<30} {min(content_lengths):<15} {min(sw_lengths):<15} {min(sec_lengths):<15}")
