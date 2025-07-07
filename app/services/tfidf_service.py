from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
from app.services.preprocess import preprocess_text
from spellchecker import SpellChecker
from collections import defaultdict
import numpy as np
import os

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
        self.collection_name= name.replace("/", "-").replace("\\", "_").strip()
        self.documents = []
        self.tfidf_matrix = None
        self.feature_names = None
        self.inverted_index = None

   

    def search_tfidf(self, query: str, top_k: int = 10):
        print("🔍 Searching TF-IDF model...")
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            self.load() 

        query_vec = self.tfidf_vectorizer.transform([query])
        # self.debug_query_terms( query_vec)
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        matching_docs = [
            {"doc_id": int(i), "score": float(similarities[i])}
            for i in top_indices if similarities[i] > 0
        ]
        matching_count = sum(1 for score in similarities if score > 0)
        
        return {
            "matched_count": matching_count,
            "results": matching_docs,
        }


    def build_vsm_tfidf(self):
        print(f"🔄 Building TF-IDF model for collection: {self.collection_name}")
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
        texts = [body for _, body in raw_docs]

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
    
        # Build Inverted Index
        self.build_inverted_index_tfidf()

        
    def search_with_inverted_index(self, query: str, top_k: int = 10):
        print("🔍 Searching with inverted index...")
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
    
        matched_docs = set()  # Use a set to store unique document IDs
        self.load()

        query_vec = self.tfidf_vectorizer.transform([query])
        # self.debug_query_terms( query_vec)

        # Check if the query vector is empty
        if query_vec.shape[0] == 0:  # Empty query vector
            raise ValueError("Query vector is empty. Please provide a valid query.")

        query_inverted_index = self.build_query_inverted_index(query_vec)

        # Use the query's inverted index to find matching documents
        for term in query_inverted_index:
            if term in self.inverted_index:  # Check the inverted index for documents containing the term
                matched_docs.update(self.inverted_index[term])  # Add document IDs to matched_docs
            else:
                print(f"⚠️ Term '{term}' not found in inverted index.")

        # Now calculate the cosine similarity for these matched documents
        if matched_docs:
            sub_matrix = self.tfidf_matrix[list(matched_docs)]
            similarities = cosine_similarity(query_vec, sub_matrix).flatten()
            top_indices = similarities.argsort()[::-1][:top_k]

            matching_docs = [
                {"doc_id": int(i), "score": float(similarities[i])}
                for i in top_indices if similarities[i] > 0
            ]
            matching_count = sum(1 for score in similarities if score > 0)

            return {
                "matched_count": matching_count,
                "results": matching_docs,
            }
        else:
            return {"matched_count": 0, "results": []}


    def build_inverted_index_tfidf(self):
        print("⏳ Building inverted index...")
        inverted_index = defaultdict(list)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        # Iterate through the TF-IDF matrix and map terms to document IDs
        for doc_idx, doc in enumerate(self.tfidf_matrix):
            for word_idx in doc.nonzero()[1]:  # Get non-zero elements (terms with non-zero weight)
                term = feature_names[word_idx]
                inverted_index[term].append(doc_idx)

        self.inverted_index = inverted_index
        return inverted_index
        

    def build_query_inverted_index(self, query_vec):
        print("🔍 Building query inverted index...")
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        nonzero_indices = query_vec.nonzero()[1]  # Get column indices with non-zero TF-IDF values

        query_inverted_index = {}
        for idx in nonzero_indices:
            term = feature_names[idx]
            query_inverted_index[term] = [0]  # Assign dummy document id as 0 for query terms
        return query_inverted_index

    def debug_query_terms(self, query_inverted_index):
        # ✅ Extract TF-IDF terms used in the query
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        nonzero_indices = query_inverted_index.nonzero()[1]  # column indices with TF-IDF > 0

        print(f"Inverted Index for Query: {query_inverted_index}")

        # Build full term-weight mapping
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
            "vocab_sample": list(clean_vocab.items())[:10],  # first 10 clean (term, index) pairs
        }

    
    def save(self):
        
        print("💾 Saving TF-IDF vectorizer and matrix...")
        os.makedirs("models/TF-IDF", exist_ok=True)

        vectorizer_file=f"models/TF-IDF/tfidf_vectorizer{self.collection_name}.joblib"
        matrix_file=f"models/TF-IDF/tfidf_matrix{self.collection_name}.joblib"  
        inverted_index_file = f"models/TF-IDF/inverted_index_{self.collection_name}.joblib"

        dump(self.inverted_index, inverted_index_file)
        dump(self.tfidf_vectorizer, vectorizer_file)
        dump(self.tfidf_matrix, matrix_file)
        print("✅ Model saved to:", vectorizer_file, "and", matrix_file, "inverted index" , inverted_index_file)


    def load(self):
        print("🔍 Loading TF-IDF vectorizer, matrix, inverted index...")
        vectorizer_file=f"models/TF-IDF/tfidf_vectorizer{self.collection_name}.joblib"
        matrix_file=f"models/TF-IDF/tfidf_matrix{self.collection_name}.joblib"
        inverted_index_file = f"models/TF-IDF/inverted_index_{self.collection_name}.joblib"

        self.tfidf_vectorizer = load(vectorizer_file)
        self.tfidf_matrix = load(matrix_file)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.inverted_index = load(inverted_index_file)




    