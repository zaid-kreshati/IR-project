from fastapi import FastAPI, APIRouter
from app.routes.BasicRouterAPI import router as BasicRouterAPI
from app.routes.BasicRouterWEB import router as BasicRouterWEB
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Information Retrieval System",
    version="1.0.0",
    description="Custom IR system with TF-IDF and BERT pipelines"
)

# Allow CORS (frontend requests from React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 👈 React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include feature-specific routers
app.include_router(BasicRouterAPI, prefix="/Basic/API", tags=["Document_Representation"])
app.include_router(BasicRouterWEB, prefix="/Basic/WEB", tags=["Document_Representation"])

@app.get("/")
def read_basic_root():
    return {"message": "IR project is running Basic Router 🚀✅"}
