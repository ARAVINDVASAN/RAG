from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
from app.services.async_rag_service import run_rag
router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing"
        )

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save file: {str(e)}"
        )

    finally:
        file.file.close()

    return {
        "filename": file.filename,
        "path": str(file_path)
    }
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel



class QuestionRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask_question(request: QuestionRequest):
    return StreamingResponse(
        run_rag(request.question),
        media_type="text/plain; charset=utf-8",
    )


