from fastapi import FastAPI, Query, APIRouter, HTTPException
from app.services.query_refiner_service import get_query_suggestions
from fastapi.responses import JSONResponse

app = FastAPI()
router = APIRouter()


@router.get("/")
def read_basic_root():
    return {"message": "IR2 project is running Basic Router 🚀✅"}



@router.get("/refine")
def refine_query(query: str = Query(..., description="User query")):
    suggestions = get_query_suggestions(query)
    return { "results": suggestions}


app.include_router(router)
