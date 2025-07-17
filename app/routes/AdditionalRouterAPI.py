from fastapi import FastAPI, Query, APIRouter, HTTPException
from app.services.query_refiner_service import get_query_suggestions
from app.services.clustering_service import clustering
from fastapi.responses import JSONResponse
from app.services.tfidf_service import VectorSpaceModel

import joblib
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = FastAPI()
router = APIRouter()


@router.get("/")
def read_basic_root():
    return {"message": "IR2 project is running Basic Router 🚀✅"}



@router.get("/refine")
def refine_query(query: str = Query(..., description="User query")):
    suggestions = get_query_suggestions(query)
    return { "results": suggestions}



@router.get("/clustering/tfidf")
def cluster_documents(collection_name: str, n_clusters: int = 5):
    try:
        results= clustering(collection_name,n_clusters)
        return results
       
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")


@router.get("/clustering/tfidf/elbow")
def elbow_method_plot(k_min: int = 2, k_max: int = 10):
    distortions = []
    K = range(k_min, k_max + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tfidf_matrix)
        distortions.append(kmeans.inertia_)

    # Plot Elbow
    fig, ax = plt.subplots()
    ax.plot(K, distortions, "bo-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Distortion)")
    ax.set_title("Elbow Method For Optimal k")

    # Encode plot as base64 image
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    encoded_plot = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "k_range": list(K),
        "distortions": distortions,
        "elbow_plot_base64": encoded_plot,
    }


@router.post("/clustering/tfidf/silhouette")
def silhouette_scores(k_min: int = 2, k_max: int = 10):
    scores = {}
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels)
        scores[k] = round(score, 4)

    return {"silhouette_scores": scores}


app.include_router(router)
