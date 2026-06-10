from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from app.models.database import get_db
from app.models.models import Experiment
from app.models.schemas import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentUpdate,
    CompareRequest,
    CompareResponse,
    MetricsSummary
)
from app.services.experiment_tracker import ExperimentTracker

router = APIRouter()


@router.get("", response_model=List[ExperimentResponse])
async def list_experiments(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    embedding_model: Optional[str] = None,
    llm: Optional[str] = None,
    retriever: Optional[str] = None,
    db = Depends(get_db)
):
    tracker = ExperimentTracker(db)
    experiments = tracker.list_experiments(
        skip=skip,
        limit=limit,
        embedding_model=embedding_model,
        llm=llm,
        retriever=retriever
    )
    
    return [ExperimentResponse.from_orm_with_metrics(exp) for exp in experiments]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    db = Depends(get_db)
):
    tracker = ExperimentTracker(db)
    experiment = tracker.get_experiment(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return ExperimentResponse.from_orm_with_metrics(experiment)


@router.post("", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreate,
    db = Depends(get_db)
):
    tracker = ExperimentTracker(db)
    experiment = tracker.create_experiment(request)
    
    return ExperimentResponse.from_orm_with_metrics(experiment)


@router.patch("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: str,
    request: ExperimentUpdate,
    db = Depends(get_db)
):
    tracker = ExperimentTracker(db)
    experiment = tracker.get_experiment(experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if request.name is not None:
        experiment.name = request.name
    if request.description is not None:
        experiment.description = request.description
    
    db.commit()
    db.refresh(experiment)
    
    return ExperimentResponse.from_orm_with_metrics(experiment)


@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    db = Depends(get_db)
):
    tracker = ExperimentTracker(db)
    success = tracker.delete_experiment(experiment_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return {"message": "Experiment deleted successfully"}


@router.get("/{experiment_id}/history")
async def get_experiment_history(
    experiment_id: str,
    metric: str = "faithfulness",
    limit: int = Query(50, ge=1, le=200),
    db = Depends(get_db)
):
    tracker = ExperimentTracker(db)
    history = tracker.get_metric_history(experiment_id, metric, limit)
    
    return {"experiment_id": experiment_id, "metric": metric, "history": history}


@router.post("/compare", response_model=CompareResponse)
async def compare_experiments(
    request: CompareRequest,
    db = Depends(get_db)
):
    tracker = ExperimentTracker(db)
    result = tracker.compare_experiments(request.experiment_ids)
    
    return CompareResponse(
        experiments=[ExperimentResponse.from_orm_with_metrics(exp) for exp in result["experiments"]],
        winning_metrics=result["winning_metrics"]
    )


@router.get("/best/{metric}")
async def get_best_experiment(
    metric: str = "faithfulness",
    db = Depends(get_db)
):
    tracker = ExperimentTracker(db)
    experiment = tracker.get_best_experiment(metric)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="No experiments found with this metric")
    
    return ExperimentResponse.from_orm_with_metrics(experiment)
