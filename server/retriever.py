import json
from pathlib import Path

# In a real scenario, this would load rank_bm25, but the hackathon rules dictate pure Python BM25 or pip install rank_bm25.
# We specified rank_bm25==0.2.2 in the requirements.
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

class BM25Retriever:
    def __init__(self, evidence_path: str = "data/evidence.json"):
        self.evidence_path = Path(evidence_path)
        self.snippets = []
        self.bm25 = None
        self._load_data()

    def _load_data(self):
        if not self.evidence_path.exists():
            return

        with open(self.evidence_path, "r", encoding="utf-8") as f:
            self.snippets = json.load(f)

        if BM25Okapi is not None and self.snippets:
            # Tokenize by simple whitespace for BM25
            tokenized_corpus = [doc["text"].lower().split(" ") for doc in self.snippets]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.snippets or not self.bm25:
            return []

        tokenized_query = query.lower().split(" ")
        # Get top k documents
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Sort by score
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            snippet = self.snippets[idx].copy()
            snippet["relevance_score"] = float(doc_scores[idx])
            results.append(snippet)

        return results
