### app/services/bm25_service.py
import joblib
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from app.services.preprocess import preprocess_bm25
from app.services.InvertedIndexService import InvertedIndexService

import os
from pymongo import MongoClient

class BM25Service:
    def __init__(self):
        # self.collection_name = collection_name
        self.tokenized_corpus = []
        self.raw_documents = []
        self.bm25 = None
        self.preprocessor = preprocess_bm25
        self.inverted_index_service = None  # Reference to InvertedIndexService


    def build(self, collection_name):
        client = MongoClient("mongodb://localhost:27017/")
        db = client["ir_project"]
        collection = db[collection_name]

        # Fetch and preprocess documents
        docs_cursor = collection.find({}, {"_id": 0, "doc_id": 1, "body": 1})
        docs = [(doc["doc_id"], doc["body"]) for doc in docs_cursor if isinstance(doc.get("body"), str)]

        if not docs:
            raise ValueError("No valid documents found in the collection.")

        self.raw_documents = docs
        self.tokenized_corpus = [self.preprocessor(doc[1]).split() for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"✅ BM25 model built and saved for collection: {collection_name}")

         # Initialize InvertedIndexService
        self.inverted_index_service = InvertedIndexService(self.tokenized_corpus)
        self.inverted_index_service.build_inverted_index()  # Build the inverted index

    def save(self, collection_name):
        """
        Save the raw documents and tokenized corpus to disk.
        """
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.raw_documents, f"models/bm25_raw_docs_{collection_name}.joblib")
        joblib.dump(self.tokenized_corpus, f"models/bm25_tokenized_{collection_name}.joblib")

    def load(self, collection_name):
        """
        Load the tokenized corpus and rebuild the BM25 model.
        """
        print("here is hte error ")
        self.raw_documents = joblib.load(f"models/bm25_raw_docs_{collection_name}.joblib")
        self.tokenized_corpus = joblib.load(f"models/bm25_tokenized_{collection_name}.joblib")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"🔄 BM25 model loaded for collection: {collection_name}")
        print(f"📄 Total documents: {len(self.raw_documents)}")
        print(f"🧾 Sample doc_id: {self.raw_documents[0][0] if self.raw_documents else 'None'}")
        print(f"📝 Sample tokens: {self.tokenized_corpus[0][:10] if self.tokenized_corpus else 'None'}")
        return {
            "collection": collection_name,
            "total_documents": len(self.raw_documents),
            "sample_doc_id": self.raw_documents[0][0] if self.raw_documents else None,
            "sample_tokens": self.tokenized_corpus[0][:10] if self.tokenized_corpus else []
        }

    def search(self, query, top_k=10):
        """
        Search for the top_k most relevant documents for the given query.
        Returns list of dicts with doc_id and similarity score.
        """
        if not self.bm25:
            raise ValueError("BM25 model not built or loaded. Call `build()` or `load()` first.")
    
        tokenized_query = self.preprocessor(query).split()
        scores = self.bm25.get_scores(tokenized_query)
    
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform([[s] for s in scores]).flatten()

        ranked = list(zip(self.raw_documents, normalized))
        sorted_ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

        filtered_ranked = [doc for doc in sorted_ranked if doc[1] > 0]

        
        return {
            "matching_count": len(filtered_ranked),
            "results": [
                {"doc_id": doc[0], "score": score} for doc, score in filtered_ranked[:top_k]
            ],
        }

        return [{"doc_id": doc[0], "score": score} for doc, score in filtered_ranked[:top_k]]

        return sorted_ranked[:top_k]

        



