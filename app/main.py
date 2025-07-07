from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from app.services.load_data import insert_documents
from app.routes.BasicRouterAPI import router as BasicRouterAPI
from app.routes.BasicRouterWEB import router as BasicRouterWEB

app = FastAPI(
    title="Information Retrieval System",
    version="1.0.0",
    description="Custom IR system with TF-IDF and BERT pipelines"
)

# Include feature-specific routers
app.include_router(BasicRouterAPI, prefix="/Basic/API", tags=["Document_Representation"])
app.include_router(BasicRouterWEB, prefix="/Basic/WEB", tags=["Document_Representation"])

# -------------------------
# General Routes
# -------------------------

@app.get("/")
def read_root():
    return {"message": "IR project is running 🚀"}

# -------------------------
# Dataset Loader Route
# -------------------------

router = APIRouter()

class DatasetRequest(BaseModel):
    dataset_name: str

@router.post("/load", tags=["Dataset Loader"])
def load_dataset(req: DatasetRequest):
    try:
        result = insert_documents(req.dataset_name)
        return {
            "message": f"✅ Dataset '{req.dataset_name}' loaded into MongoDB",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Register the dataset loader router
app.include_router(router)