from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import pathlib
from spellchecker import SpellChecker
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Import preprocessing
from app.services.preprocess_service import preprocess_text
from joblib import dump, load

def clustering(collection_name: str, n_clusters: int = 5):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # Use f-string for proper string formatting
    tfidf_matrix = load(f'models/TF-IDF/tfidf_matrix{collection_name}.joblib')
    tfidf_vectorizer=load(f'models/TF-IDF/tfidf_vectorizer{collection_name}.joblib')
        
    if tfidf_matrix is None or not hasattr(tfidf_vectorizer, 'get_feature_names_out'):
        raise ValueError("Invalid TF-IDF matrix or vectorizer")
            
    kmeans.fit(tfidf_matrix)
    labels = kmeans.labels_

    # Get top terms per cluster
    terms = tfidf_vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    top_terms = {
        f"Cluster {i}": [terms[ind] for ind in order_centroids[i, :10]]
        for i in range(n_clusters)
    }

    return {
        "n_clusters": n_clusters,
        "labels": labels.tolist(),
        "top_terms_per_cluster": top_terms,
    }