import json

import numpy as np
from sentence_transformers import SentenceTransformer
from minsearch import Index, VectorSearch
from tqdm.auto import tqdm


class SearchTool:
    """Encapsulates text and vector search over chunked documents."""

    def __init__(self, chunks, embedding_model_name='multi-qa-distilbert-cos-v1',
                 embeddings_path=None):
        self.chunks = chunks
        self.model = SentenceTransformer(embedding_model_name)

        # Build text index
        print("Building text index...")
        self.text_index = Index(
            text_fields=["chunk", "filename"],
            keyword_fields=[]
        )
        self.text_index.fit(chunks)

        # Build vector index
        if embeddings_path:
            print(f"Loading cached embeddings from {embeddings_path}...")
            embeddings = np.load(embeddings_path)
        else:
            print("Computing embeddings...")
            embeddings = []
            for d in tqdm(chunks, desc="Embedding"):
                embeddings.append(self.model.encode(d['chunk']))
            embeddings = np.array(embeddings)

        self.vec_index = VectorSearch()
        self.vec_index.fit(embeddings, chunks)
        self.embeddings = embeddings
        print(f"Search tool ready ({len(chunks)} chunks indexed)")

    def text_search(self, query: str, num_results: int = 5) -> list:
        """Keyword-based text search."""
        return self.text_index.search(query, num_results=num_results)

    def vector_search(self, query: str, num_results: int = 5) -> list:
        """Semantic vector search."""
        q = self.model.encode(query)
        return self.vec_index.search(q, num_results=num_results)

    def hybrid_search(self, query: str, num_results: int = 5) -> list:
        """Combine text and vector search, deduplicated."""
        text_results = self.text_search(query, num_results=num_results)
        vec_results = self.vector_search(query, num_results=num_results)

        seen = set()
        combined = []
        for r in text_results + vec_results:
            key = (r['filename'], r.get('start'))
            if key not in seen:
                seen.add(key)
                combined.append(r)
        return combined

    def search(self, query: str, num_results: int = 5) -> str:
        """Search and return formatted results string (for agent tool use)."""
        results = self.hybrid_search(query, num_results=num_results)
        if not results:
            return "No results found."

        output = []
        for i, r in enumerate(results):
            preview = r['chunk'][:1500]
            output.append(f"[{i+1}] Source: {r['filename']}\n{preview}")
        return "\n\n---\n\n".join(output)


def load_search_tool(chunks_path='fastapi_chunks_sliding.json',
                     embeddings_path='fastapi_embeddings.npy'):
    """Load chunks from file and create a SearchTool."""
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return SearchTool(chunks, embeddings_path=embeddings_path)
