from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import torch
import joblib
import os

from utility.text_processing_helper import preprocess_text_embeddings

app = FastAPI()

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["ir_project"]


class DocumentRequest(BaseModel):
    doc_id: str
    text: str


class QueryRequest(BaseModel):
    query: str
    similarity_threshold: float = 0.001


class EmbeddingSearcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = self.check_gpu_compatibility()
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.embeddings_by_dataset = {}

    def check_gpu_compatibility(self):
        if torch.backends.mps.is_available():
            print("MPS backend is available.")
            return torch.device("mps")
        else:
            print("MPS backend is not available. Using CPU.")
            return torch.device("cpu")

    def build_embeddings_for_dataset(self, dataset_name: str):
        collection = db[dataset_name]
        docs = list(collection.find({}, {"_id": 0, "text": 1}))
        texts = [doc["text"] for doc in docs]

        if not texts:
            raise ValueError("No documents to build embeddings.")

        embeddings = self.model.encode(texts)
        self.embeddings_by_dataset[dataset_name] = embeddings
        joblib.dump(embeddings, f"{dataset_name}_embeddings.joblib")

    def search_dataset(self, dataset_name: str, query: str, similarity_threshold=0.001):
        collection = db[dataset_name]

        # Load embeddings if not already loaded
        if dataset_name not in self.embeddings_by_dataset:
            emb_file = f"{dataset_name}_embeddings.joblib"
            if not os.path.exists(emb_file):
                raise FileNotFoundError(f"Embeddings not found for dataset '{dataset_name}'.")
            self.embeddings_by_dataset[dataset_name] = joblib.load(emb_file)

        embeddings = self.embeddings_by_dataset[dataset_name]

        query_processed = preprocess_text_embeddings(query)
        query_embedding = self.model.encode(query_processed)

        similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings).flatten()
        docs = list(collection.find({}, {"_id": 0, "doc_id": 1}))
        doc_ids = [doc["doc_id"] for doc in docs]

        document_ranking = {
            doc_id: score for doc_id, score in zip(doc_ids, similarities)
        }

        filtered = {
            doc_id: score for doc_id, score in document_ranking.items()
            if score >= similarity_threshold
        }

        sorted_docs = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs


searcher = EmbeddingSearcher()


@app.post("/build_embeddings/{dataset_name}")
def build_embeddings(dataset_name: str = Path(...)):
    try:
        searcher.build_embeddings_for_dataset(dataset_name)
        return {"message": f"Embeddings built and saved for dataset '{dataset_name}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/{dataset_name}")
def search(dataset_name: str, request: QueryRequest):
    try:
        results = searcher.search_dataset(dataset_name, request.query, request.similarity_threshold)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
