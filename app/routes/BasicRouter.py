# app/routes/BasicRouter.py

from fastapi import FastAPI, Query, APIRouter, HTTPException
from pydantic import BaseModel

from app.services.tfidf_service import VectorSpaceModel
# from app.services.embedding_service import EmbeddingSearcher  # Assumes this exists

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
        embeddings.build_embeddings(name)
        embeddings.save_embeddings(name)
        return {"message": "✅ Embeddings built and saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load-embedded")
def load_embedded(collection_name: str = "documents"):
    try:
        name = sanitize_collection_name(collection_name)
        embeddings = EmbeddingSearcher()
        embeddings.load_embeddings(name)
        return {"message": "✅ Embeddings loaded."}
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
