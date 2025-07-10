# app/routes/BasicRouter.py
from fastapi import FastAPI, APIRouter
from pymongo import MongoClient
from typing import List


app = FastAPI(
    title="Information Retrieval System",
    description="TF-IDF powered IR API",
    version="1.0.0"
)

router = APIRouter(tags=["TF-IDF"])



# -------------------------------
# 🌐 Health Check
# -------------------------------

@router.get("/")
def read_basic_root():
    return {"message": "IR project is running Basic Router 🚀✅"}


@app.get("/home")
def returnview():
    return ValuesView({'frontend-ui/src/pages/Home.tsx'})


@router.get("/get-datasets", tags=["Dataset"])
def get_datasets():
    try:
        # Connect to MongoDB (adjust connection string if needed)
        client = MongoClient("mongodb://localhost:27017/")
        db = client["ir_project"]  # Replace with your DB name
        datasets = db.list_collection_names()
        print(datasets)
        return {"datasets": datasets}
    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# 🔗 Register Routes
# -------------------------------

app.include_router(router)





