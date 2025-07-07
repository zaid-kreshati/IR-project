# app/routes/BasicRouter.py
from fastapi import FastAPI, Query, APIRouter, HTTPException
from pydantic import BaseModel

from app.services.tfidf_service import VectorSpaceModel
from app.services.embeddings_service import EmbeddingSearcher  # Assumes this exists
from app.services.bm25_service import BM25Service
from app.services.hybrid_service import HybridSearchService  # ⬅️ Import this at the top

app = FastAPI(
    title="Information Retrieval System",
    description="TF-IDF powered IR API",
    version="1.0.0"
)

router = APIRouter(tags=["TF-IDF"])

# -------------------------------
# 📦 Models
# -------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    collection_name: str = "documents"

# -------------------------------
# 🔍 TF-IDF Routes
# -------------------------------

@router.post("/search/tfidf")
async def tfidf_search(request: QueryRequest):
    try:
        vsm = VectorSpaceModel(request.collection_name)
        results = vsm.search_tfidf(request.query, top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/tfidf-inverted")
async def tfidf_search_with_inverted_index(request: QueryRequest):
    try:
        vsm = VectorSpaceModel(request.collection_name)
        results = vsm.search_with_inverted_index(request.query, top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/build-tfidf-model")
def build_tfidf(collection_name: str = "documents"):
    try:
        vsm = VectorSpaceModel(collection_name)
        vsm.build_vsm_tfidf()
        vsm.save()
        return {"message": "✅ TF-IDF model built and saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load-tfidf")
def load_tfidf(collection_name: str = "documents"):
    try:
        vsm = VectorSpaceModel(collection_name)
        vsm.load()
        return vsm.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 🤖 Embedding Routes
# -------------------------------

@router.get("/build-embedded")
def build_embedded(collection_name: str = "documents"):
    try:
        embeddings = EmbeddingSearcher(collection_name=collection_name)
        embeddings.build_documents_embeddings()
        embeddings.save()
        return {"message": "✅ Embeddings built and saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load-embedded")
def load_embedded(collection_name: str = "documents"):
    try:
        embeddings = EmbeddingSearcher(collection_name=collection_name)
        embeddings.load()
        return embeddings.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search_embedded")
async def embedded_search(request: QueryRequest):
    try:
        embeddings = EmbeddingSearcher(collection_name=request.collection_name)
        embeddings.load()
        results = embeddings.search(request.query, top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search_embedded/vector_index")
async def embedded_search(request: QueryRequest):
    try:
        embeddings = EmbeddingSearcher(collection_name=request.collection_name)
        embeddings.load()
        results = embeddings.search_vector_index(request.query, top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------
# BM25
# ------------------
### app/routes/bm25_routes.py

@router.get("/build-bm25")
def build_bm25_index(collection_name: str = "documents"):
    try:
        bm25 = BM25Service(collection_name)
        bm25.build()
        bm25.save()
        return {"message": f"✅ BM25 index built for collection '{collection_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/BM25")
def search_bm25(request: QueryRequest):
    try:
        bm25 = BM25Service(request.collection_name)
        bm25.load()
        results = bm25.search(request.query, top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/BM25/inverted_index")
def search_bm25(request: QueryRequest):
    try:
        bm25 = BM25Service(request.collection_name)
        bm25.load()
        results = bm25.search(request.query, top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load/BM25")
def load_bm25_info(collection_name: str = Query(...)):
    try:
        bm25 = BM25Service(collection_name)
        info = bm25.load()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------
#       hybrid
# ------------------

@router.post("/hybrid/rrf")
async def hybrid_rrf_search(request: QueryRequest):
    try:
        service = HybridSearchService(request.collection_name)
        print("Loading models for collection1:", request.collection_name)
        await service.load_models()
        print("Loaded models for collection1:", request.collection_name)
        results = await service.search(query=request.query, top_k=request.top_k)
        return {
            "query": request.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid/rrf/with_Index")
async def hybrid_rrf_search(request: QueryRequest):
    try:
        service = HybridSearchService(request.collection_name)
        await service.load_models()
        print("Loaded models for collection1:", request.collection_name)
        results = await service.search(query=request.query, top_k=request.top_k)
        return {
            "query": request.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# -------------------------------
# 🌐 Health Check
# -------------------------------

@router.get("/")
def read_basic_root():
    return {"message": "IR project is running Basic Router 🚀✅"}

# -------------------------------
# 🔗 Register Routes
# -------------------------------

app.include_router(router)
