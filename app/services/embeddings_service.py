from codecs import raw_unicode_escape_decode
from sys import setdlopenflags
from typing_extensions import Collection
import joblib
from pymongo import MongoClient
import sentence_transformers
from joblib import dump, load
from sklearn.metrics.pairwise import cosine_similarity
import torch
from app.services.preprocess_service import preprocess_text_embeddings
import ir_datasets
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import numpy as np
from annoy import AnnoyIndex
import time
class EmbeddingSearcher:
    def __init__(self, model_name='all-MiniLM-L6-v2', preprocessor=preprocess_text_embeddings,collection_name:str ="documents"):
        self.collection_name = collection_name.replace("/", "-").replace("\\", "_").strip()
        self.documents = []
        self.document_embeddings = None
        self.device = self.check_gpu_compatibility()
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)

        # self.model = SentenceTransformer(model_name, device=self.device)
        self.preprocessor = preprocessor
        self.vectorizer = None
        self.matrix = None
       
        # Assuming embedding dimensions from model
        self.embedding_dimension = 384  # For 'all-MiniLM-L6-v2' model, typical embedding dimension is 768
        self.index = AnnoyIndex(self.embedding_dimension, 'angular')  # Using 'angular' distance (cosine similarity)
    


    def check_gpu_compatibility(self):
        if torch.backends.mps.is_available():
            print("MPS backend is available.")
            return torch.device("mps")
        else:
            print("MPS backend is not available. Using CPU.")
            return torch.device("cpu")
   
    def build_documents_embeddings(self):
        print("building documents embeddings....")
        client = MongoClient("mongodb://localhost:27017/")
        db = client["ir_project"]
        collection = db[self.collection_name]

        raw_docs = [
            (doc["doc_id"], doc["body"])
            for doc in collection.find({}, {"_id": 0, "doc_id": 1, "body": 1})
            if isinstance(doc.get("body"), str) and len(doc["body"]) < 50000
        ]


        if not raw_docs:
            raise ValueError("No valid documents found.")

        self.documents = raw_docs
        if not self.documents:
            print("⚠️ No documents with 'text' field found!")
            return

        # Preprocess and encode texts
        document_texts = [self.preprocessor(text) for _, text in self.documents]


        self.document_embeddings = self.model.encode(document_texts)
    
        print("building vector index....")
        for i, embedding in enumerate(self.document_embeddings):
            self.index.add_item(i, embedding)
        self.index.build(50)  # 10 trees for faster search (can be adjusted for speed vs. accuracy)
  
        
    def search(self, query, top_k=500000, threshold=0.1):
        start_time = time.time()
        
        query_embedding = self.model.encode(query).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.document_embeddings).flatten()
        # Map doc_id to similarity score and body
        document_ranking = {
            doc_id: {"score": float(score), "body": body}
            for (doc_id, body), score in zip(self.documents, similarities)
        }
        # Filter documents by threshold
        filtered_documents = {
            doc_id: doc_info
            for doc_id, doc_info in document_ranking.items()
            if doc_info["score"] > threshold
        }
        # Sort by similarity descending
        sorted_dict = sorted(filtered_documents.items(), key=lambda item: item[1]["score"], reverse=True)
        matching_count = len(filtered_documents)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "matched_count": matching_count,
            "results": [
                {"doc_id": doc_id, "score": doc_info["score"], "body": doc_info["body"]}
                for doc_id, doc_info in sorted_dict[:top_k]
            ],
            "execution_time": execution_time
        }


    def get_model_info(self):
        print('\n🔍 Embedding Info:')
        print("Type:", type(self.document_embeddings))
        shape = getattr(self.document_embeddings, 'shape', None)
        sample_vector = []
        if isinstance(self.document_embeddings, np.ndarray):
            if self.document_embeddings.ndim == 2:
                # Expected: (n_documents, embedding_dim)
                sample_vector = self.document_embeddings[0][:10].tolist()
            elif self.document_embeddings.ndim == 1:
                # Unusual case: 1D vector
                sample_vector = self.document_embeddings[:10].tolist()
            else:
                sample_vector = [float(self.document_embeddings[0])]
        
        return {
            "embedded_docs_type": str(type(self.document_embeddings)),
            "embedded_docs_shape": shape if shape else "Unknown shape",
            "sample_embedding_vector": sample_vector,
            "index_type": str(type(self.index))
        }


   
   
    def save(self):
        print("saving embeddings documents....")
        documents_file=f"models/Embeddings/documents_{self.collection_name}.joblib"
        embeddings_file=f"models/Embeddings/document_embeddings_{self.collection_name}.joblib"  
        index_file= f"models/Embeddings/vector_Index{self.collection_name}.ann" 
        print("💾 Saving embeddings documents ")
        os.makedirs("models/Embeddings", exist_ok=True)
        dump(self.documents, documents_file)
        dump(self.document_embeddings, embeddings_file)
        self.index.save(index_file)  # Save the Annoy index    

    def load(self):
        print("🔍 Loading embeddings documents")
        documents_file=f"models/Embeddings/documents_{self.collection_name}.joblib"
        embeddings_file=f"models/Embeddings/document_embeddings_{self.collection_name}.joblib"  
        index = AnnoyIndex(self.embedding_dimension, 'angular')
        index.load(f"models/Embeddings/vector_Index{self.collection_name}.ann")
        data = joblib.load(documents_file)
        embeddings = joblib.load(embeddings_file)  
        self.document_embeddings = embeddings
        self.documents = data
        self.index.load(f"models/Embeddings/vector_Index{self.collection_name}.ann")  # Load the Annoy index
        return self


    def search_vector_index(self, query, top_k=500000, threshold=0.0):
        start_time = time.time()
        
        query_embedding = self.model.encode(query).reshape(1, -1)
        indices = self.index.get_nns_by_vector(query_embedding.flatten(), top_k)
        results = []
        for idx in indices:
            doc_id, body = self.documents[idx]
            similarity = float(1.0 - cosine_similarity(query_embedding, self.document_embeddings[idx].reshape(1, -1))[0][0])
            if similarity > threshold:
                results.append({"doc_id": doc_id, "score": similarity, "body": body})
    
        matching_count = len(results)  # Count only filtered results
        
        end_time = time.time()
        execution_time = end_time - start_time
    
        return {
            "execution_time": execution_time,
            "matched_count": matching_count,
            "results": results,
        }
