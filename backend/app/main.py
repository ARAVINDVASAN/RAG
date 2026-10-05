from fastapi import FastAPI,APIRouter
from app.api.v1.endpoints.rag import router
app = FastAPI()


app.include_router(router,tags=["rag"])


@app.get("/health")
async def root():
    return {"message": "all running okay"}