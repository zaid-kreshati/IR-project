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
# 🔧 Utility
# -------------------------------

def sanitize_collection_name(name: str) -> str:
    return name.replace("/", "-").replace("\\", "_").strip()

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
        name = sanitize_collection_name(request.collection_name)
        vsm = VectorSpaceModel()
        results = vsm.search_tfidf(request.query, top_k=request.top_k, name=name)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/tfidf-inverted")
async def tfidf_search_with_inverted_index(request: QueryRequest):
    try:
        name = sanitize_collection_name(request.collection_name)
        vsm = VectorSpaceModel()
        results = vsm.search_with_inverted_index(request.query, top_k=request.top_k, name=name)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/build-inverted")
def build_inverted_index(collection_name: str = "documents"):
    try:
        name = sanitize_collection_name(collection_name)
        vsm = VectorSpaceModel()
        vsm.load(name)
        print("debug here")
        return {"message": "✅ Inverted index built."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/build-tfidf-model")
def build_tfidf(collection_name: str = "documents"):
    try:
        name = sanitize_collection_name(collection_name)
        vsm = VectorSpaceModel()
        vsm.build_vsm_tfidf(name)
        vsm.save(name)
        return {"message": "✅ TF-IDF model built and saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load-tfidf")
def load_tfidf(collection_name: str = "documents"):
    try:
        name = sanitize_collection_name(collection_name)
        vsm = VectorSpaceModel()
        vsm.load(name)
        return vsm.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 🤖 Embedding Routes
# -------------------------------

@router.get("/build-embedded")
def build_embedded(collection_name: str = "documents"):
    try:
        name = sanitize_collection_name(collection_name)
        embeddings = EmbeddingSearcher()
        embeddings.build_documents_embeddings(name)
        embeddings.save(collection_name=name)
        return {"message": "✅ Embeddings built and saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load-embedded")
def load_embedded(collection_name: str = "documents"):
    try:
        name = sanitize_collection_name(collection_name)
        embeddings = EmbeddingSearcher()
        embeddings.load(name)
        return embeddings.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search_embedded")
async def embedded_search(request: QueryRequest):
    try:
        name = sanitize_collection_name(request.collection_name)
        embeddings = EmbeddingSearcher()
        embeddings.load(name)
        results = embeddings.search(request.query, top_k=request.top_k, collection_name=name)
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
        name = sanitize_collection_name(collection_name)
        bm25 = BM25Service()
        bm25.build(name)
        bm25.save(name)
        return {"message": f"✅ BM25 index built for collection '{name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/BM25")
def search_bm25(request: QueryRequest):
    try:
        bm25 = BM25Service()
        bm25.load(request.collection_name)
        results = bm25.search(request.query, top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load/BM25")
def load_bm25_info(collection_name: str = Query(...)):
    try:
        bm25 = BM25Service()
        info = bm25.load(collection_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------
#       hybrid
# ------------------

@router.post("/hybrid/rrf")
async def hybrid_rrf_search(request: QueryRequest):
    try:
        name = sanitize_collection_name(request.collection_name)
        service = HybridSearchService(name)
        await service.load_models()
        print("Loaded models for collection1:", name)
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
