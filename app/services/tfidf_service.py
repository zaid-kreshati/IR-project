from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
from app.services.preprocess_service import preprocess_text
from spellchecker import SpellChecker

from collections import defaultdict
import numpy as np
import os
import time

class VectorSpaceModel:
    @staticmethod
    def spell_checker():
        return SpellChecker()

    def __init__(self, name):
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=preprocess_text,
            # max_df=0.7,
            # min_df=5,              # 🔥 ignore very rare words
            max_features=50000,    # 🔥 cap vocab size
            norm="l2",
            use_idf=True,
            token_pattern=None  # to suppress tokenizer warning
        )
        self.collection_name = name.replace("/", "-").replace("\\", "_").strip()
        self.documents = []
        self.doc_ids = []  # Store document IDs separately
        self.doc_bodies = []  # Store document bodies separately
        self.tfidf_matrix = None
        self.feature_names = None

    def search_tfidf(self, query: str, top_k: int = 10, threshold=0.0):
        # print("🔍 Searching TF-IDF model...")
        start_time = time.time()
        
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            self.load()

        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        matching_docs = [
            {
                "doc_id": self.doc_ids[i],
                "score": float(similarities[i]),
                "body": self.doc_bodies[i]
            }
            for i in top_indices if similarities[i] > threshold
        ]
        matching_count = sum(1 for score in similarities if score > threshold)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "matched_count": matching_count,
            "results": matching_docs,
            "execution_time": execution_time
        }

    def build_vsm_tfidf(self):
        print(f"🔄 Building TF-IDF model for collection: {self.collection_name}")
        client = MongoClient("mongodb://localhost:27017/")
        db = client["ir_project"]
        collection = db[self.collection_name]

        batch_size = 1000
        docs = []
        
        cursor = collection.find({}, {"_id": 0, "doc_id": 1, "body": 1})
        for doc in cursor:
            if isinstance(doc.get("body"), str) and len(doc["body"]) < 50000:
                # Truncate extremely long document IDs if needed
                doc_id = str(doc["doc_id"])[:1000] if len(str(doc["doc_id"])) > 1000 else doc["doc_id"]
                docs.append((doc_id, doc["body"]))
            

        if not docs:
            raise ValueError("No valid documents found.")

        # Separate document IDs and texts
        self.doc_ids, texts = zip(*docs)
        self.doc_bodies = list(texts)  # Store document bodies
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        # Build Inverted Index
        self.build_inverted_index_tfidf()


    def search_with_inverted_index(self, query: str, top_k: int = 10, threshold:float =0.0):
        print("🔍 Searching with inverted index...")
        start_time = time.time()
        
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
    
        matched_docs = set()
        self.load()

        query_vec = self.tfidf_vectorizer.transform([query])

        if query_vec.shape[0] == 0:
            raise ValueError("Query vector is empty. Please provide a valid query.")

        query_inverted_index = self.build_query_inverted_index(query_vec)

        for term in query_inverted_index:
            if term in self.inverted_index:
                matched_docs.update(self.inverted_index[term])
            else:
                print(f"⚠️ Term '{term}' not found in inverted index.")

        result = {"matched_count": 0, "results": [], "execution_time": 0}
        
        if matched_docs:
            sub_matrix = self.tfidf_matrix[list(matched_docs)]
            similarities = cosine_similarity(query_vec, sub_matrix).flatten()
            top_indices = similarities.argsort()[::-1][:top_k]

            matching_docs = [
                {
                    "doc_id": self.doc_ids[list(matched_docs)[i]],
                    "score": float(similarities[i]),
                    "body": self.doc_bodies[list(matched_docs)[i]]
                }
                for i in top_indices if similarities[i] > threshold
            ]
            matching_count = sum(1 for score in similarities if score > threshold)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            result = {
                "matched_count": matching_count,
                "results": matching_docs,
                "execution_time": execution_time
            }
        
        return result

    def build_inverted_index_tfidf(self):
        print("⏳ Building inverted index...")
        inverted_index = defaultdict(list)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        for doc_idx, doc in enumerate(self.tfidf_matrix):
            for word_idx in doc.nonzero()[1]:
                term = feature_names[word_idx]
                inverted_index[term].append(doc_idx)

        self.inverted_index = inverted_index
        return inverted_index

    def build_query_inverted_index(self, query_vec):
        print("🔍 Building query inverted index...")
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        nonzero_indices = query_vec.nonzero()[1]

        query_inverted_index = {}
        for idx in nonzero_indices:
            term = feature_names[idx]
            query_inverted_index[term] = [0]
        return query_inverted_index

    def debug_query_terms(self, query_inverted_index):
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        nonzero_indices = query_inverted_index.nonzero()[1]

        print(f"Inverted Index for Query: {query_inverted_index}")

        query_term_weights = {
            feature_names[i]: query_inverted_index[0, i]
            for i in nonzero_indices
        }
        sorted_items = list(query_term_weights.items())

        for i in range(0, len(sorted_items), 10):
            term, weight = sorted_items[i]
            print(f"{i:5d}: {term:30s} → {weight:.4f}")

        if not query_term_weights:
            print("⚠️ No query terms found in TF-IDF vocabulary.")
        return

    def get_model_info(self):
        clean_vocab = {term: int(index) for term, index in self.tfidf_vectorizer.vocabulary_.items()}
        return {
            "vectorizer_type": str(type(self.tfidf_vectorizer)),
            "matrix_shape": self.tfidf_matrix.shape if self.tfidf_matrix is not None else None,
            "inverted_index_size": len(self.inverted_index) if self.inverted_index is not None else None,
            "vocab_sample": list(clean_vocab.items())[:10],
        }

    def save(self):
        print("💾 Saving TF-IDF vectorizer and matrix...")
        os.makedirs("models/TF-IDF", exist_ok=True)

        vectorizer_file=f"models/TF-IDF/tfidf_vectorizer{self.collection_name}.joblib"
        matrix_file=f"models/TF-IDF/tfidf_matrix{self.collection_name}.joblib"  
        inverted_index_file = f"models/TF-IDF/inverted_index_{self.collection_name}.joblib"
        doc_ids_file = f"models/TF-IDF/doc_ids_{self.collection_name}.joblib"
        doc_bodies_file = f"models/TF-IDF/doc_bodies_{self.collection_name}.joblib"

        dump(self.inverted_index, inverted_index_file)
        dump(self.tfidf_vectorizer, vectorizer_file)
        dump(self.tfidf_matrix, matrix_file)
        dump(self.doc_ids, doc_ids_file)
        dump(self.doc_bodies, doc_bodies_file)
        print("✅ Model saved to:", vectorizer_file, "and", matrix_file, "inverted index" , inverted_index_file)

    def load(self):
        print("🔍 Loading TF-IDF vectorizer, matrix, inverted index...")
        vectorizer_file=f"models/TF-IDF/tfidf_vectorizer{self.collection_name}.joblib"
        matrix_file=f"models/TF-IDF/tfidf_matrix{self.collection_name}.joblib"
        inverted_index_file = f"models/TF-IDF/inverted_index_{self.collection_name}.joblib"
        doc_ids_file = f"models/TF-IDF/doc_ids_{self.collection_name}.joblib"
        doc_bodies_file = f"models/TF-IDF/doc_bodies_{self.collection_name}.joblib"

        self.tfidf_vectorizer = load(vectorizer_file)
        self.tfidf_matrix = load(matrix_file)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.inverted_index = load(inverted_index_file)
        self.doc_ids = load(doc_ids_file)
        self.doc_bodies = load(doc_bodies_file)