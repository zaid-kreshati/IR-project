from typing_extensions import Collection
import joblib
from pymongo import MongoClient
import sentence_transformers
from joblib import dump, load
from sklearn.metrics.pairwise import cosine_similarity
import torch
from app.services.preprocess import preprocess_text_embeddings
import ir_datasets
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import numpy as np




class EmbeddingSearcher:
    def __init__(self, model_name='all-MiniLM-L6-v2', preprocessor=preprocess_text_embeddings):
        self.documents = []
        self.document_embeddings = None
        self.device = self.check_gpu_compatibility()
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)

        # self.model = SentenceTransformer(model_name, device=self.device)
        self.preprocessor = preprocessor
        self.vectorizer = None
        self.matrix = None
        # self.collection_name = collection_name


    def check_gpu_compatibility(self):
        if torch.backends.mps.is_available():
            print("MPS backend is available.")
            return torch.device("mps")
        else:
            print("MPS backend is not available. Using CPU.")
            return torch.device("cpu")
   
    def build_documents_embeddings(self, collection_name):
        client = MongoClient("mongodb://localhost:27017/")
        db = client["ir_project"]
        collection = db[collection_name]


        raw_docs = [
            (doc["doc_id"], doc["body"])
            for doc in collection.find({}, {"_id": 0, "doc_id": 1, "body": 1})
            if isinstance(doc.get("body"), str) and len(doc["body"]) < 50000
        ]

        if not raw_docs:
            raise ValueError("No valid documents found.")

        self.documents = raw_docs
        # Fetch all documents
        # documents_cursor = collection.find({})
        # print(documents_cursor)     
        # self.documents = [(doc["_id"], doc["text"]) for doc in documents_cursor if "text" in doc]

        # self.documents = [(doc["_id"], doc["text"]) for doc in documents_cursor if "text" in doc]

        print(f"📄 Total documents loaded: {len(self.documents)}")

        if not self.documents:
            print("⚠️ No documents with 'text' field found!")
            return

        # Preprocess and encode texts
        document_texts = [self.preprocessor(text) for _, text in self.documents]
        print(f"🧼 First preprocessed doc: {document_texts[0] if document_texts else 'None'}")


        self.document_embeddings = self.model.encode(document_texts)
        print(f"✅ Embeddings shape: {self.document_embeddings.shape}")

    

    def search(self, query, similarity_threshold=0.001, collection_name="documents", top_k=500000):
        # if self.document_embeddings is None:
        # self.load(self.collection_name)  # make sure they are loaded once

        # query_embedding = self.document_embeddings.encode([query])
        query_embedding = self.model.encode(query).reshape(1, -1)
        # query_embedding = self.model.encode((query))
        similarities = cosine_similarity(query_embedding, self.document_embeddings).flatten()
        # similarities = cosine_similarity(query_embedding.reshape(
        #     1, -1), self.document_embeddings).flatten()
        doc = self.documents
        # document_ranking = dict(zip(doc, similarities))
        document_ranking = {
            doc_id: float(score)
            for (doc_id, _), score in zip(self.documents, similarities)
        }
        filtered_documents = {key: float(value) for key, value in document_ranking.items(
        ) }

        filtered_documents = {doc_id: score for doc_id, score in filtered_documents.items() if score > 0}


        sorted_dict = sorted(filtered_documents.items(),
                             key=lambda item: item[1], reverse=True)
        matching_count = sum(1 for score in similarities if score > 0)

        # Return structured results with doc_id and embedding similarity
        # return {
        #     "matching_count": matching_count,
        #     "results": [
        #         {
        #             "doc_id": doc_id,
        #             "similarity": similarity,
        #         }
        #         for i, ((doc_id, _), similarity) in enumerate(sorted_dict[:top_k])
        #     ],
        # }

        return {
            "matching_count": matching_count,
            "results": [
                {"doc_id": doc_id, "similarity": score}
                for doc_id, score in sorted_dict[:top_k]
            ],
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
            "sample_embedding_vector": sample_vector
        }


   
   
    def save(self, collection_name):
        documents_file=f"models/documents_{collection_name}.joblib"
        embeddings_file=f"models/document_embeddings_{collection_name}.joblib"  
        print("💾 Saving embeddings documents ")
        os.makedirs("models", exist_ok=True)
        dump(self.documents, documents_file)
        dump(self.document_embeddings, embeddings_file)
        print("✅ Model saved to:", documents_file, "and", embeddings_file)
    

    def load(self, collection_name):
        print("🔍 Loading embeddings documents")
        print(collection_name)
        documents_file=f"models/documents_{collection_name}.joblib"
        embeddings_file=f"models/document_embeddings_{collection_name}.joblib"  
        print(documents_file)
        print(embeddings_file)
        data = joblib.load(documents_file)
        embeddings = joblib.load(embeddings_file)  
        self.document_embeddings = embeddings
        self.documents = data
        return self


