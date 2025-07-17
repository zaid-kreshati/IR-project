### app/services/hybrid_service.py
from collections import defaultdict
from typing import List
import time
from app.services.bm25_service import BM25Service
from app.services.embeddings_service import EmbeddingSearcher
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

    async def search(self, query: str, top_k: int = 10, threshold=0.0):
        print("searching....")
        start_time = time.time()
        
        # Get top_k results from both services
        bm25_raw = await asyncio.to_thread(self.bm25_service.search, query, top_k=top_k, threshold=threshold)
        embed_raw = await asyncio.to_thread(self.embed_service.search, query, top_k=top_k, threshold=threshold)

        print(f"bm25 :{bm25_raw['matched_count']}")
        print(f"embedding :{embed_raw['matched_count']}")

        # Create mappings of doc_id to body and scores
        bm25_ranking = [doc['doc_id'] for doc in bm25_raw['results']]  # Extracting from 'results'
        embed_ranking = [res["doc_id"] for res in embed_raw["results"]]

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

    async def search_with_Index(self, query: str, top_k: int = 10, threshold=0.0):
        start_time = time.time()
        
        # Get top_k results from both services
        bm25_raw = await asyncio.to_thread(self.bm25_service.search_with_inverted_index, query, top_k=top_k, threshold=threshold)
        embed_raw = await asyncio.to_thread(self.embed_service.search_vector_index, query, top_k=top_k, threshold=threshold)
        
        # Create mappings of doc_id to body and scores
        bm25_docs = {doc['doc_id']: {'body': doc['body'], 'score': doc['score']} for doc in bm25_raw['results']}
        embed_docs = {res['doc_id']: {'body': res['body'], 'score': res['score']} for res in embed_raw['results']}

        # Merge results while preserving scores
        merged_results = []
        
        # Add BM25 results
        for doc_id, data in bm25_docs.items():
            merged_results.append({
                'doc_id': doc_id,
                'body': data['body'],
                'score': data['score'],
                'source': 'bm25'
            })
            
        # Add embedding results
        for doc_id, data in embed_docs.items():
            if doc_id not in bm25_docs:  # Avoid duplicates
                merged_results.append({
                    'doc_id': doc_id,
                    'body': data['body'],
                    'score': data['score'],
                    'source': 'embedding'
                })

        # Sort merged results by score
        merged_results.sort(key=lambda x: x['score'], reverse=True)

        execution_time = time.time() - start_time

        return {
            "matched_count": len(merged_results),
            "execution_time": round(execution_time, 4),
            "results": merged_results[:top_k]
        }
