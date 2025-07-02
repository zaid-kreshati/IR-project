from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
from app.services.preprocess import preprocess_text
from spellchecker import SpellChecker
import numpy as np
import os
import re



class VectorSpaceModel:
    @staticmethod
    def spell_checker():
        return SpellChecker()

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=preprocess_text,
            # max_df=0.7,
            # min_df=5,              # 🔥 ignore very rare words
            max_features=50000,    # 🔥 cap vocab size
            norm="l2",
            use_idf=True,
            token_pattern=None  # to suppress tokenizer warning

        )
        self.documents = []
        self.tfidf_matrix = None
        self.feature_names = None
   

    def search_tfidf(self, query: str, top_k: int = 10, name: str = "documents"):
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            self.load(name)  # make sure they are loaded once


        query_vec = self.tfidf_vectorizer.transform([query])
        self.debug_query_terms( query_vec)
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        # matching_docs = [
        #     {"doc_id": int(i), "score": float(sim)}
        #     for i, sim in enumerate(similarities) if sim > 0
        # ]

        matching_docs = [
            {"doc_id": int(i), "score": float(similarities[i])}
            for i in top_indices
        ]
        matching_count = sum(1 for score in similarities if score > 0)
        
        return {
            "matched_count": matching_count,
            "results": matching_docs,
        }

    def debug_query_terms(self, query_vec):
         # ✅ Extract TF-IDF terms used in the query
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        nonzero_indices = query_vec.nonzero()[1]  # column indices with TF-IDF > 0

        # Build full term-weight mapping
        query_term_weights = {
            feature_names[i]: query_vec[0, i]
            for i in nonzero_indices
        }
        print("\n🔍 Sampled TF-IDF Terms (1 per 10):")
        sorted_items = list(query_term_weights.items())

        for i in range(0, len(sorted_items), 10):
            term, weight = sorted_items[i]
            print(f"{i:5d}: {term:30s} → {weight:.4f}")




        if not query_term_weights:
            print("⚠️ No query terms found in TF-IDF vocabulary.")
            return
        
        print("\n🔍 All Query TF-IDF Terms & Weights:")
        for term, weight in query_term_weights.items():
            print(f"{term}: {weight:.4f}")


    def get_model_info(self):
        # ✅ 1. Clean and print word indexes
        print('\n🔍 Word Indexes (cleaned):')
        clean_vocab = {term: int(index) for term, index in self.tfidf_vectorizer.vocabulary_.items()}
        print(clean_vocab)

        # ✅ 2. Display sparse TF-IDF matrix (optional: convert to dense for small matrices)
        print(self.tfidf_matrix)  # Or: print(self.tfidf_matrix.toarray()) for dense (if small)

        # ✅ 3. Return metadata
        return {
            "vectorizer_type": str(type(self.tfidf_vectorizer)),
            "matrix_shape": self.tfidf_matrix.shape if self.tfidf_matrix is not None else None,
            "vocab_sample": list(clean_vocab.items())[:10],  # first 10 clean (term, index) pairs
        }
    

    def build_vsm_tfidf(self, collection_name):
        client = MongoClient("mongodb://localhost:27017/")
        db = client["ir_project"]
        collection = db[collection_name]

        print("📥 Loading documents from MongoDB...")

        raw_docs = [
            (doc["doc_id"], doc["body"])
            for doc in collection.find({}, {"_id": 0, "doc_id": 1, "body": 1})
            if isinstance(doc.get("body"), str) and len(doc["body"]) < 50000
        ]

        if not raw_docs:
            raise ValueError("No valid documents found.")

        self.documents = raw_docs
        texts = [body for _, body in raw_docs]

        print("⚙️ Building TF-IDF matrix...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"✅ TF-IDF model built: {len(self.documents)} documents, {len(self.feature_names)} features.")
    

    def save(self, collection_name):
        vectorizer_file=f"models/tfidf_vectorizer{collection_name}.joblib"
        matrix_file=f"models/tfidf_matrix{collection_name}.joblib"  
        print("💾 Saving TF-IDF vectorizer and matrix...")
        os.makedirs("models", exist_ok=True)
        dump(self.tfidf_vectorizer, vectorizer_file)
        dump(self.tfidf_matrix, matrix_file)
        print("✅ Model saved to:", vectorizer_file, "and", matrix_file)


    def load(self, collection_name):
        vectorizer_file=f"models/tfidf_vectorizer{collection_name}.joblib"
        matrix_file=f"models/tfidf_matrix{collection_name}.joblib"
        print("🔍 Loading TF-IDF vectorizer and matrix...")
        self.tfidf_vectorizer = load(vectorizer_file)
        self.tfidf_matrix = load(matrix_file)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

