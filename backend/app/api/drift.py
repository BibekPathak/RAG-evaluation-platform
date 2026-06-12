from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.schemas import (
    DriftAlertResponse,
    DriftSummary,
    DriftStats,
    VersionComparison,
    DatasetVersionResponse
)
from app.services.drift_detector import DriftDetector

router = APIRouter()


@router.get("/stats", response_model=DriftStats)
async def get_drift_stats(db: Session = Depends(get_db)):
    detector = DriftDetector(db)
    stats = detector.get_drift_stats()

    return DriftStats(
        total_datasets=stats["total_datasets"],
        datasets_with_drift=stats["datasets_with_drift"],
        avg_drift_score=stats["avg_drift_score"],
        critical_alerts=stats["critical_alerts"],
        warning_alerts=stats["warning_alerts"],
        total_active_alerts=stats["total_active_alerts"],
        recent_alerts=[DriftAlertResponse(**a) for a in stats["recent_alerts"]],
        drift_history=stats["drift_history"]
    )


@router.get("/alerts", response_model=List[DriftAlertResponse])
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by status: active, acknowledged, resolved"),
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    detector = DriftDetector(db)
    alerts = detector.get_all_alerts(status=status, dataset_id=dataset_id, limit=limit)

    return [DriftAlertResponse(**a) for a in alerts]


@router.post("/alerts/{alert_id}/acknowledge", response_model=DriftAlertResponse)
async def acknowledge_alert(
    alert_id: str,
    db: Session = Depends(get_db)
):
    detector = DriftDetector(db)
    alert = detector.acknowledge_alert(alert_id)

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return DriftAlertResponse(
        id=alert.id,
        dataset_id=alert.dataset_id,
        dataset_version_id=alert.dataset_version_id,
        metric=alert.metric,
        old_value=alert.old_value,
        new_value=alert.new_value,
        change_pct=alert.change_pct,
        psi=alert.psi,
        severity=alert.severity,
        status=alert.status,
        created_at=alert.created_at,
        acknowledged_at=alert.acknowledged_at,
        resolved_at=alert.resolved_at
    )


@router.post("/alerts/{alert_id}/resolve", response_model=DriftAlertResponse)
async def resolve_alert(
    alert_id: str,
    db: Session = Depends(get_db)
):
    detector = DriftDetector(db)
    alert = detector.resolve_alert(alert_id)

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return DriftAlertResponse(
        id=alert.id,
        dataset_id=alert.dataset_id,
        dataset_version_id=alert.dataset_version_id,
        metric=alert.metric,
        old_value=alert.old_value,
        new_value=alert.new_value,
        change_pct=alert.change_pct,
        psi=alert.psi,
        severity=alert.severity,
        status=alert.status,
        created_at=alert.created_at,
        acknowledged_at=alert.acknowledged_at,
        resolved_at=alert.resolved_at
    )


@router.post("/alerts/acknowledge-all")
async def acknowledge_all_alerts(
    dataset_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    detector = DriftDetector(db)
    count = detector.acknowledge_all_alerts(dataset_id=dataset_id)

    return {"message": f"Acknowledged {count} alerts", "count": count}


@router.get("/datasets/{dataset_id}/versions", response_model=List[DatasetVersionResponse])
async def get_dataset_versions(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    detector = DriftDetector(db)
    versions = detector.get_dataset_versions(dataset_id)

    return [
        DatasetVersionResponse(
            id=v.id,
            dataset_id=v.dataset_id,
            version_number=v.version_number,
            version_hash=v.version_hash,
            question_count=v.question_count,
            drift_score=v.drift_score,
            drift_details=v.drift_details,
            created_at=v.created_at
        )
        for v in versions
    ]


@router.get("/datasets/{dataset_id}/drift-summary", response_model=DriftSummary)
async def get_drift_summary(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    detector = DriftDetector(db)
    summary = detector.get_drift_summary(dataset_id)

    if not summary:
        raise HTTPException(status_code=404, detail="Dataset not found or has no versions")

    return DriftSummary(
        dataset_id=summary["dataset_id"],
        dataset_name=summary["dataset_name"],
        current_version=summary["current_version"],
        total_versions=summary["total_versions"],
        current_drift_score=summary["current_drift_score"],
        drift_detected=summary["drift_detected"],
        active_alerts=summary["active_alerts"],
        acknowledged_alerts=summary["acknowledged_alerts"],
        resolved_alerts=summary["resolved_alerts"],
        recent_versions=[
            DatasetVersionResponse(**v) for v in summary["recent_versions"]
        ],
        metrics_comparison=summary["metrics_comparison"]
    )


@router.get("/datasets/{dataset_id}/compare", response_model=VersionComparison)
async def compare_versions(
    dataset_id: str,
    version_a: int = Query(..., description="First version number"),
    version_b: int = Query(..., description="Second version number"),
    db: Session = Depends(get_db)
):
    detector = DriftDetector(db)
    comparison = detector.compare_versions(dataset_id, version_a, version_b)

    if not comparison:
        raise HTTPException(status_code=404, detail="One or both versions not found")

    return VersionComparison(
        dataset_id=comparison["dataset_id"],
        dataset_name=comparison["dataset_name"],
        version_a=DatasetVersionResponse(**comparison["version_a"]),
        version_b=DatasetVersionResponse(**comparison["version_b"]),
        metrics_a=comparison["metrics_a"],
        metrics_b=comparison["metrics_b"],
        metric_changes=comparison["metric_changes"],
        drift_result=comparison["drift_result"]
    )


@router.post("/datasets/{dataset_id}/create-version")
async def create_version(
    dataset_id: str,
    questions: List[dict],
    metrics_summary: dict,
    change_summary: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    detector = DriftDetector(db)

    try:
        version, drift_result = detector.create_version(
            dataset_id=dataset_id,
            questions=questions,
            metrics_summary=metrics_summary,
            change_summary=change_summary
        )

        return {
            "version_id": version.id,
            "version_number": version.version_number,
            "drift_score": version.drift_score,
            "drift_detected": drift_result.drift_detected if drift_result else False,
            "alerts_created": len(drift_result.alerts) if drift_result else 0,
            "severity": drift_result.severity if drift_result else None
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create version: {str(e)}")
