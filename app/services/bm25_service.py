### app/services/bm25_service.py
import joblib
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from app.services.preprocess_service import preprocess_bm25
import time
from collections import defaultdict

import os
from pymongo import MongoClient

class BM25Service:
    def __init__(self, name):
        self.tokenized_corpus = []
        self.raw_documents = []
        self.bm25 = None
        self.preprocessor = preprocess_bm25
        self.inverted_index = None  # Reference to InvertedIndexService
        self.query_inverted_index = None  # Reference for the query inverted index
        self.collection_name = name.replace("/", "-").replace("\\", "_").strip()


    def build(self):
        print(f"Building BM25 model for collection: {self.collection_name}")
        client = MongoClient("mongodb://localhost:27017/")
        db = client["ir_project"]
        collection = db[self.collection_name]

        docs_cursor = collection.find({}, {"_id": 0, "doc_id": 1, "body": 1})
        docs = [(doc["doc_id"], doc["body"]) for doc in docs_cursor if isinstance(doc.get("body"), str)]

        if not docs:
            raise ValueError("No valid documents found in the collection.")

        self.raw_documents = docs
        self.tokenized_corpus = [self.preprocessor(doc[1]).split() for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Build the inverted index for the corpus
        self.inverted_index = self.build_inverted_index_bm25()
        print(f"✅ BM25 model built and saved for collection: {self.collection_name}")


    def save(self):
        print(f"⚠️ Saving BM25 model for collection: {self.collection_name}")
        os.makedirs("models/BM25", exist_ok=True)
        joblib.dump(self.raw_documents, f"models/BM25/bm25_raw_docs_{self.collection_name}.joblib")
        joblib.dump(self.tokenized_corpus, f"models/BM25/bm25_tokenized_{self.collection_name}.joblib")
        joblib.dump(self.inverted_index, f"models/BM25/bm25_inverted_index_{self.collection_name}.joblib")

    def load(self):
        print(f"⚠️ Loading BM25 model for collection: {self.collection_name}")
        self.raw_documents = joblib.load(f"models/BM25/bm25_raw_docs_{self.collection_name}.joblib")
        self.tokenized_corpus = joblib.load(f"models/BM25/bm25_tokenized_{self.collection_name}.joblib")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.inverted_index = joblib.load(f"models/BM25/bm25_inverted_index_{self.collection_name}.joblib")
        return {
            "collection": self.collection_name,
            "total_documents": len(self.raw_documents),
            "sample_doc_id": self.raw_documents[0][0] if self.raw_documents else None,
            "sample_tokens": self.tokenized_corpus[0][:10] if self.tokenized_corpus else [],
            "inverted_index_size": len(self.inverted_index) if self.inverted_index else None,
        }

    def search(self, query, top_k=10):
        print(f"🔍 Searching BM25 model for collection: {self.collection_name}")
        start_time = time.time()
        
        if not self.bm25:
            raise ValueError("BM25 model not built or loaded. Call `build()` or `load()` first.")
    
        tokenized_query = self.preprocessor(query).split()
        scores = self.bm25.get_scores(tokenized_query)
    
        # Normalize scores between 0 and 1
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform([[s] for s in scores]).flatten()

        # Rank documents based on normalized scores
        ranked = list(zip(self.raw_documents, normalized))
        sorted_ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

        filtered_ranked = [doc for doc in sorted_ranked if doc[1] > 0]
        
        execution_time = time.time() - start_time
        
        return {
            "matched_count": len(filtered_ranked),
            "results": [
                {"doc_id": doc[0], "score": score} for doc, score in filtered_ranked[:top_k]
            ],
            "execution_time": execution_time
        }

        
    def build_inverted_index_bm25(self):
        inverted_index = defaultdict(list)
        if self.tokenized_corpus:
            for doc_idx, tokens in enumerate(self.tokenized_corpus):
                for token in set(tokens):  # Avoid duplicates of the same term in one document
                    inverted_index[token].append(doc_idx)

        self.inverted_index = inverted_index
        return inverted_index

    def build_query_inverted_index(self, query):
        # Tokenize and preprocess the query
        tokenized_query = self.preprocessor(query).split()

        query_inverted_index = defaultdict(list)
        for idx, token in enumerate(tokenized_query):
            query_inverted_index[token].append(idx)  # Store token positions in the query
        
        self.query_inverted_index = query_inverted_index
        return query_inverted_index

    def search_with_inverted_index(self, query, top_k=10):
        print(f"🔍 Searching BM25 model with inverted index for collection: {self.collection_name}")
        start_time = time.time()
        
        if not self.bm25:
            raise ValueError("BM25 model not built or loaded. Call `build()` or `load()` first.")

        # Build the inverted index for the query
        query_inverted_index = self.build_query_inverted_index(query)
        
        # Retrieve documents and calculate BM25 scores based on the query inverted index
        tokenized_query = self.preprocessor(query_inverted_index).split()
        scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores between 0 and 1
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform([[s] for s in scores]).flatten()

        # Rank documents based on scores
        ranked = list(zip(self.raw_documents, normalized))
        sorted_ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

        filtered_ranked = [doc for doc in sorted_ranked if doc[1] > 0]
        
        execution_time = time.time() - start_time
        
        return {
            "matching_count": len(filtered_ranked),
            "results": [
                {"doc_id": doc[0], "score": score} for doc, score in filtered_ranked[:top_k]
            ],
            "execution_time": execution_time
        }
