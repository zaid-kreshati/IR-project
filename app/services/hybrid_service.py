### app/services/hybrid_service.py
from collections import defaultdict
from typing import List
from app.services.bm25_service import BM25Service
from app.services.embeddings_service import EmbeddingSearcher
from app.services.preprocess import preprocess_text, preprocess_text_embeddings
import asyncio


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60):
    """
    Perform Reciprocal Rank Fusion (RRF) on a list of rankings.
    Each ranking is a list of doc_ids sorted by relevance (most relevant first).
    """
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridSearchService:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.bm25_service = BM25Service()
        self.embed_service = EmbeddingSearcher()

    async def load_models(self):
        print("Loading models for collection:", self.collection_name)
        await asyncio.to_thread(self.bm25_service.load, self.collection_name)
        await asyncio.to_thread(self.embed_service.load, self.collection_name)

    async def search(self, query: str, top_k: int = 10):
        bm25_raw = await asyncio.to_thread(self.bm25_service.search, query, 500000)
        embed_raw = await asyncio.to_thread(self.embed_service.search, query, self.collection_name, 500000)



        # Extract ranked doc_ids from both sources
        # bm25_ranking = [doc['doc_id'] for doc in bm25_raw]  # Extracting doc_id from each dictionary
        bm25_ranking = [doc['doc_id'] for doc in bm25_raw['results']]  # Extracting from 'results'

        # bm25_ranking = [doc[0][0] for doc in bm25_raw]  # (doc_id, body), score
        print(" bm25_ranking:", len(bm25_ranking))
        embed_ranking = [res["doc_id"] for res in embed_raw["results"]]
        print("embed_ranking:",len(embed_ranking))


        # Apply Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion([bm25_ranking, embed_ranking], k=60)

        return {
             "count": len(fused_results),
            "results": [
                {"doc_id": doc_id, "fused_score": round(score, 4)}
                for doc_id, score in fused_results[:top_k]
            ]
        }
