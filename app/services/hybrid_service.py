### app/services/hybrid_service.py
from collections import defaultdict
from typing import List
import time
from app.services.bm25_service import BM25Service
from app.services.embeddings_service import EmbeddingSearcher
from app.services.preprocess_service import preprocess_text, preprocess_text_embeddings
import asyncio


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60):
    print("rankings:", len(rankings))
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridSearchService:
    def __init__(self, collection_name: str):
        self.bm25_service = BM25Service(collection_name)
        self.embed_service = EmbeddingSearcher(collection_name=collection_name)
        self.collection_name = collection_name

    async def load_models(self):
        print("Loading models for collection:", self.collection_name)
        await asyncio.to_thread(self.bm25_service.load)
        await asyncio.to_thread(self.embed_service.load)

    async def search(self, query: str, top_k: int = 10):
        print("searching....")
        start_time = time.time()
        
        bm25_raw = await asyncio.to_thread(self.bm25_service.search, query, 500000)
        embed_raw = await asyncio.to_thread(self.embed_service.search, query, 500000)

        # bm25_ranking = [doc['doc_id'] for doc in bm25_raw]  # Extracting doc_id from each dictionary
        bm25_ranking = [doc['doc_id'] for doc in bm25_raw['results']]  # Extracting from 'results'

        # bm25_ranking = [doc[0][0] for doc in bm25_raw]  # (doc_id, body), score
        print(" bm25_ranking:", len(bm25_ranking))
        embed_ranking = [res["doc_id"] for res in embed_raw["results"]]
        print("embed_ranking:",len(embed_ranking))
        # Apply Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion([bm25_ranking, embed_ranking], k=60)

        execution_time = time.time() - start_time
        
        return {
             "matched_count": len(fused_results),
             "execution_time": round(execution_time, 4),
             "results": [
                {"doc_id": doc_id, "score": round(score, 4)}
                for doc_id, score in fused_results[:top_k]
            ]
        }

    async def search_with_Index(self, query: str, top_k: int = 10):
        print("searching with index....")
        start_time = time.time()
        
        bm25_raw = await asyncio.to_thread(self.bm25_service.search_with_inverted_index, query, 500000)
        embed_raw = await asyncio.to_thread(self.embed_service.search_vector_index, query, self.collection_name, 500000)
        bm25_ranking = [doc['doc_id'] for doc in bm25_raw['results']]  # Extracting from 'results'
        embed_ranking = [res["doc_id"] for res in embed_raw["results"]]

        # Apply Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion([bm25_ranking, embed_ranking], k=60)

        execution_time = time.time() - start_time

        return {
             "count": len(fused_results),
             "execution_time": round(execution_time, 4),
             "results": [
                {"doc_id": doc_id, "fused_score": round(score, 4)}
                for doc_id, score in fused_results[:top_k]
            ]
        }
