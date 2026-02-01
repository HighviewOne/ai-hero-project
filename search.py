import json

import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from minsearch import Index, VectorSearch


def load_chunks(filepath='fastapi_chunks_sliding.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_text_index(chunks):
    index = Index(
        text_fields=["chunk", "filename"],
        keyword_fields=[]
    )
    index.fit(chunks)
    return index


def build_vector_index(chunks, embedding_model):
    embeddings = []
    for d in tqdm(chunks, desc="Building embeddings"):
        v = embedding_model.encode(d['chunk'])
        embeddings.append(v)
    embeddings = np.array(embeddings)

    vindex = VectorSearch()
    vindex.fit(embeddings, chunks)
    return vindex, embeddings


def text_search(query, index, num_results=5):
    return index.search(query, num_results=num_results)


def vector_search(query, vindex, embedding_model, num_results=5):
    q = embedding_model.encode(query)
    return vindex.search(q, num_results=num_results)


def hybrid_search(query, index, vindex, embedding_model, num_results=5):
    text_results = text_search(query, index, num_results=num_results)
    vec_results = vector_search(query, vindex, embedding_model, num_results=num_results)

    seen_ids = set()
    combined = []
    for result in text_results + vec_results:
        key = (result['filename'], result.get('start'))
        if key not in seen_ids:
            seen_ids.add(key)
            combined.append(result)
    return combined


def print_results(results, label, max_text=200):
    print(f"\n--- {label} ---")
    for i, r in enumerate(results):
        chunk_preview = r['chunk'][:max_text].replace('\n', ' ')
        print(f"  {i+1}. [{r['filename']}] {chunk_preview}...")
    print()


if __name__ == "__main__":
    print("Loading chunks...")
    chunks = load_chunks('fastapi_chunks_sliding.json')
    print(f"Loaded {len(chunks)} chunks")

    # 1. Text search
    print("\nBuilding text index...")
    text_idx = build_text_index(chunks)

    query = "How do I add authentication to FastAPI?"
    results = text_search(query, text_idx)
    print_results(results, f"Text Search: '{query}'")

    # 2. Vector search
    print("Loading embedding model...")
    model = SentenceTransformer('multi-qa-distilbert-cos-v1')

    print("Building vector index (this may take a while)...")
    vec_idx, embeddings = build_vector_index(chunks, model)

    results = vector_search(query, vec_idx, model)
    print_results(results, f"Vector Search: '{query}'")

    # 3. Hybrid search
    results = hybrid_search(query, text_idx, vec_idx, model)
    print_results(results, f"Hybrid Search: '{query}'")

    # Save embeddings for reuse
    np.save('fastapi_embeddings.npy', embeddings)
    print("Saved embeddings to fastapi_embeddings.npy")
