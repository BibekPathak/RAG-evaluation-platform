from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import Optional, List
import tempfile
import os
import uuid
from app.models.database import get_db
from app.models.models import Dataset
from app.models.schemas import (
    DatasetCreate,
    DatasetResponse,
    DatasetUploadResponse,
    QuestionAnswer
)
from app.services.dataset_generator import DatasetGeneratorService

router = APIRouter()


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    source_type: str = Form("pdf"),
    num_questions: int = Form(10),
    db = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        generator = DatasetGeneratorService()
        text, chunks = generator.extract_and_chunk(tmp_path, source_type)
        
        if not text:
            raise HTTPException(status_code=400, detail="Failed to extract text from file")
        
        questions = generator.generate_qa_pairs(text, num_pairs=num_questions, source=file.filename)
        
        dataset = Dataset(
            id=str(uuid.uuid4()),
            name=name,
            source_type=source_type,
            source_path=file.filename,
            question_count=len(questions),
            questions=questions,
            created_at=__import__('datetime').datetime.utcnow(),
            updated_at=__import__('datetime').datetime.utcnow()
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        return DatasetUploadResponse(
            dataset=DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                source_type=dataset.source_type,
                source_path=dataset.source_path,
                question_count=dataset.question_count,
                questions=[QuestionAnswer(**q) for q in dataset.questions],
                created_at=dataset.created_at,
                updated_at=dataset.updated_at
            ),
            chunks_created=len(chunks),
            questions_generated=len(questions)
        )
    finally:
        os.unlink(tmp_path)


@router.get("", response_model=List[DatasetResponse])
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    db = Depends(get_db)
):
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).offset(skip).limit(limit).all()
    return [
        DatasetResponse(
            id=d.id,
            name=d.name,
            source_type=d.source_type,
            source_path=d.source_path,
            question_count=d.question_count,
            questions=[QuestionAnswer(**q) for q in d.questions],
            created_at=d.created_at,
            updated_at=d.updated_at
        )
        for d in datasets
    ]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        source_type=dataset.source_type,
        source_path=dataset.source_path,
        question_count=dataset.question_count,
        questions=[QuestionAnswer(**q) for q in dataset.questions],
        created_at=dataset.created_at,
        updated_at=dataset.updated_at
    )


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    db = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    db.delete(dataset)
    db.commit()
    
    return {"message": "Dataset deleted successfully"}


@router.post("/{dataset_id}/regenerate")
async def regenerate_questions(
    dataset_id: str,
    num_questions: int = 10,
    db = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset.source_path and os.path.exists(dataset.source_path):
        generator = DatasetGeneratorService()
        text, _ = generator.extract_and_chunk(dataset.source_path, dataset.source_type)
    else:
        text = " ".join([q.get("answer", "") for q in dataset.questions])
    
    generator = DatasetGeneratorService()
    questions = generator.generate_qa_pairs(text, num_pairs=num_questions, source=dataset.source_path or "document")
    
    dataset.questions = questions
    dataset.question_count = len(questions)
    dataset.updated_at = __import__('datetime').datetime.utcnow()
    
    db.commit()
    db.refresh(dataset)
    
    return {
        "dataset_id": dataset.id,
        "questions_generated": len(questions)
    }
