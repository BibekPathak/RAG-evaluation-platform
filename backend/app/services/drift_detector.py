import math
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.models import Dataset, DatasetVersion, DriftAlert
from app.models.schemas import DriftResult, MetricChange, DriftAlertResponse

logger = logging.getLogger(__name__)


DRIFT_THRESHOLDS = {
    "psi_low": 0.1,
    "psi_medium": 0.2,
    "psi_high": 0.3,
}

METRIC_THRESHOLDS = {
    "context_recall": {"min": 0.85, "degradation_threshold": 0.05},
    "context_precision": {"min": 0.85, "degradation_threshold": 0.05},
    "faithfulness": {"min": 0.90, "degradation_threshold": 0.03},
    "answer_relevancy": {"min": 0.85, "degradation_threshold": 0.05},
    "hallucination_rate": {"max": 0.05, "degradation_threshold": 0.02},
    "mrr": {"min": 0.80, "degradation_threshold": 0.05},
    "ndcg": {"min": 0.80, "degradation_threshold": 0.05},
    "hit_rate": {"min": 0.85, "degradation_threshold": 0.05},
}


class DriftDetector:
    def __init__(self, db: Session):
        self.db = db

    def compute_hash(self, questions: List[Any]) -> str:
        questions_str = json.dumps(questions, sort_keys=True)
        return hashlib.sha256(questions_str.encode()).hexdigest()

    def calculate_psi(self, old_value: float, new_value: float) -> float:
        if old_value <= 0:
            old_value = 0.001
        if new_value <= 0:
            new_value = 0.001

        ratio = new_value / old_value if old_value > 0 else 0

        if ratio <= 0:
            return 1.0

        psi = (new_value - old_value) * math.log(ratio)

        return abs(psi)

    def determine_severity(self, psi: float, change_pct: float, metric: str) -> str:
        if psi >= DRIFT_THRESHOLDS["psi_high"] or abs(change_pct) >= 10:
            return "critical"
        elif psi >= DRIFT_THRESHOLDS["psi_medium"] or abs(change_pct) >= 5:
            return "high"
        elif psi >= DRIFT_THRESHOLDS["psi_low"] or abs(change_pct) >= 3:
            return "medium"
        else:
            return "low"

    def compute_drift_score(
        self,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float]
    ) -> DriftResult:
        metric_changes = {}
        alerts = []
        total_psi = 0.0
        metrics_with_change = 0

        all_metrics = set(old_metrics.keys()) | set(new_metrics.keys())

        for metric in all_metrics:
            old_val = old_metrics.get(metric, 0.0)
            new_val = new_metrics.get(metric, 0.0)

            if old_val == 0 and new_val == 0:
                continue

            change_pct = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
            psi = self.calculate_psi(old_val, new_val)

            severity = self.determine_severity(psi, change_pct, metric)

            metric_changes[metric] = MetricChange(
                old=old_val,
                new=new_val,
                change_pct=change_pct,
                psi=psi
            )

            total_psi += psi
            metrics_with_change += 1

            if metric in METRIC_THRESHOLDS:
                threshold = METRIC_THRESHOLDS[metric]

                if "min" in threshold and new_val < threshold["min"]:
                    alerts.append({
                        "metric": metric,
                        "severity": "critical" if new_val < threshold["min"] * 0.9 else "high",
                        "message": f"{metric} below minimum threshold ({new_val:.2%} < {threshold['min']:.0%})",
                        "old_value": old_val,
                        "new_value": new_val,
                        "change_pct": change_pct,
                        "psi": psi
                    })

                if "max" in threshold and new_val > threshold["max"]:
                    alerts.append({
                        "metric": metric,
                        "severity": "critical" if new_val > threshold["max"] * 1.5 else "high",
                        "message": f"{metric} above maximum threshold ({new_val:.2%} > {threshold['max']:.0%})",
                        "old_value": old_val,
                        "new_value": new_val,
                        "change_pct": change_pct,
                        "psi": psi
                    })

            if abs(change_pct) >= 5:
                direction = "dropped" if change_pct < 0 else "increased"
                alerts.append({
                    "metric": metric,
                    "severity": severity,
                    "message": f"{metric} {direction} {abs(change_pct):.1f}% ({old_val:.2%} → {new_val:.2%})",
                    "old_value": old_val,
                    "new_value": new_val,
                    "change_pct": change_pct,
                    "psi": psi
                })

        avg_psi = total_psi / metrics_with_change if metrics_with_change > 0 else 0.0

        drift_detected = avg_psi >= DRIFT_THRESHOLDS["psi_low"] or len(alerts) > 0

        severity = "critical" if avg_psi >= DRIFT_THRESHOLDS["psi_high"] or any(
            a["severity"] == "critical" for a in alerts
        ) else "high" if avg_psi >= DRIFT_THRESHOLDS["psi_medium"] or any(
            a["severity"] == "high" for a in alerts
        ) else "medium" if avg_psi >= DRIFT_THRESHOLDS["psi_low"] or any(
            a["severity"] == "medium" for a in alerts
        ) else "low"

        return DriftResult(
            psi=avg_psi,
            drift_detected=drift_detected,
            severity=severity,
            metric_changes=metric_changes,
            alerts=alerts
        )

    def create_version(
        self,
        dataset_id: str,
        questions: List[Any],
        metrics_summary: Dict[str, float],
        change_summary: Optional[str] = None
    ) -> Tuple[DatasetVersion, Optional[DriftResult]]:
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        version_hash = self.compute_hash(questions)

        existing_versions = self.db.query(DatasetVersion).filter(
            DatasetVersion.dataset_id == dataset_id
        ).order_by(DatasetVersion.version_number.desc()).all()

        previous_version = existing_versions[0] if existing_versions else None
        previous_metrics = previous_version.metrics_summary if previous_version else {}

        drift_result = None
        if previous_version:
            drift_result = self.compute_drift_score(
                previous_metrics,
                metrics_summary
            )

        new_version_number = (previous_version.version_number + 1) if previous_version else 1

        new_version = DatasetVersion(
            id=hashlib.md5(f"{dataset_id}_{new_version_number}_{version_hash}".encode()).hexdigest()[:36],
            dataset_id=dataset_id,
            version_number=new_version_number,
            version_hash=version_hash,
            question_count=len(questions),
            questions_snapshot=questions,
            metrics_summary=metrics_summary,
            drift_score=drift_result.psi if drift_result else None,
            drift_details=drift_result.model_dump() if drift_result else None,
            created_at=datetime.utcnow()
        )

        self.db.add(new_version)

        dataset.version_count = new_version_number
        dataset.current_version_id = new_version.id
        dataset.questions = questions
        dataset.question_count = len(questions)
        dataset.updated_at = datetime.utcnow()

        self.db.flush()

        if drift_result and drift_result.alerts:
            for alert_data in drift_result.alerts:
                alert = DriftAlert(
                    id=hashlib.md5(f"{dataset_id}_{new_version_number}_{alert_data['metric']}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:36],
                    dataset_id=dataset_id,
                    dataset_version_id=new_version.id,
                    metric=alert_data["metric"],
                    old_value=alert_data["old_value"],
                    new_value=alert_data["new_value"],
                    change_pct=alert_data["change_pct"],
                    psi=alert_data["psi"],
                    severity=alert_data["severity"],
                    status="active",
                    created_at=datetime.utcnow()
                )
                self.db.add(alert)

        self.db.commit()
        self.db.refresh(new_version)

        return new_version, drift_result

    def get_dataset_versions(self, dataset_id: str) -> List[DatasetVersion]:
        return self.db.query(DatasetVersion).filter(
            DatasetVersion.dataset_id == dataset_id
        ).order_by(DatasetVersion.version_number.desc()).all()

    def get_drift_summary(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            return None

        versions = self.get_dataset_versions(dataset_id)

        if not versions:
            return None

        current_version = versions[0]

        active_alerts = self.db.query(DriftAlert).filter(
            DriftAlert.dataset_id == dataset_id,
            DriftAlert.status == "active"
        ).count()

        acknowledged_alerts = self.db.query(DriftAlert).filter(
            DriftAlert.dataset_id == dataset_id,
            DriftAlert.status == "acknowledged"
        ).count()

        resolved_alerts = self.db.query(DriftAlert).filter(
            DriftAlert.dataset_id == dataset_id,
            DriftAlert.status == "resolved"
        ).count()

        metrics_comparison = None
        if len(versions) >= 2:
            metrics_comparison = {}
            old_metrics = versions[1].metrics_summary
            new_metrics = versions[0].metrics_summary

            all_metrics = set(old_metrics.keys()) | set(new_metrics.keys())
            for metric in all_metrics:
                old_val = old_metrics.get(metric, 0.0)
                new_val = new_metrics.get(metric, 0.0)
                change_pct = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
                psi = self.calculate_psi(old_val, new_val)

                metrics_comparison[metric] = MetricChange(
                    old=old_val,
                    new=new_val,
                    change_pct=change_pct,
                    psi=psi
                )

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "current_version": current_version.version_number,
            "total_versions": len(versions),
            "current_drift_score": current_version.drift_score,
            "drift_detected": current_version.drift_score >= DRIFT_THRESHOLDS["psi_low"] if current_version.drift_score else False,
            "active_alerts": active_alerts,
            "acknowledged_alerts": acknowledged_alerts,
            "resolved_alerts": resolved_alerts,
            "recent_versions": [
                {
                    "id": v.id,
                    "dataset_id": v.dataset_id,
                    "version_number": v.version_number,
                    "version_hash": v.version_hash,
                    "question_count": v.question_count,
                    "drift_score": v.drift_score,
                    "drift_details": v.drift_details,
                    "created_at": v.created_at
                }
                for v in versions[:10]
            ],
            "metrics_comparison": metrics_comparison
        }

    def compare_versions(
        self,
        dataset_id: str,
        version_a: int,
        version_b: int
    ) -> Optional[Dict[str, Any]]:
        ver_a = self.db.query(DatasetVersion).filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.version_number == version_a
        ).first()

        ver_b = self.db.query(DatasetVersion).filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.version_number == version_b
        ).first()

        if not ver_a or not ver_b:
            return None

        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()

        drift_result = self.compute_drift_score(
            ver_a.metrics_summary,
            ver_b.metrics_summary
        )

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name if dataset else "",
            "version_a": {
                "id": ver_a.id,
                "dataset_id": ver_a.dataset_id,
                "version_number": ver_a.version_number,
                "version_hash": ver_a.version_hash,
                "question_count": ver_a.question_count,
                "drift_score": ver_a.drift_score,
                "created_at": ver_a.created_at
            },
            "version_b": {
                "id": ver_b.id,
                "dataset_id": ver_b.dataset_id,
                "version_number": ver_b.version_number,
                "version_hash": ver_b.version_hash,
                "question_count": ver_b.question_count,
                "drift_score": ver_b.drift_score,
                "created_at": ver_b.created_at
            },
            "metrics_a": ver_a.metrics_summary,
            "metrics_b": ver_b.metrics_summary,
            "metric_changes": drift_result.metric_changes,
            "drift_result": drift_result
        }

    def get_all_alerts(
        self,
        status: Optional[str] = None,
        dataset_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        query = self.db.query(DriftAlert)

        if status:
            query = query.filter(DriftAlert.status == status)

        if dataset_id:
            query = query.filter(DriftAlert.dataset_id == dataset_id)

        alerts = query.order_by(DriftAlert.created_at.desc()).limit(limit).all()

        result = []
        for alert in alerts:
            dataset = self.db.query(Dataset).filter(Dataset.id == alert.dataset_id).first()
            result.append({
                "id": alert.id,
                "dataset_id": alert.dataset_id,
                "dataset_version_id": alert.dataset_version_id,
                "dataset_name": dataset.name if dataset else "",
                "metric": alert.metric,
                "old_value": alert.old_value,
                "new_value": alert.new_value,
                "change_pct": alert.change_pct,
                "psi": alert.psi,
                "severity": alert.severity,
                "status": alert.status,
                "created_at": alert.created_at,
                "acknowledged_at": alert.acknowledged_at,
                "resolved_at": alert.resolved_at
            })

        return result

    def get_drift_stats(self) -> Dict[str, Any]:
        total_datasets = self.db.query(Dataset).count()
        datasets_with_versions = self.db.query(Dataset).filter(
            Dataset.version_count > 0
        ).count()

        all_alerts = self.db.query(DriftAlert).all()

        active_alerts = [a for a in all_alerts if a.status == "active"]
        critical_alerts = [a for a in active_alerts if a.severity == "critical"]
        warning_alerts = [a for a in active_alerts if a.severity in ("high", "medium")]

        total_psi = sum(
            v.drift_score for v in self.db.query(DatasetVersion).all()
            if v.drift_score is not None
        )
        versions_with_drift = self.db.query(DatasetVersion).filter(
            DatasetVersion.drift_score >= DRIFT_THRESHOLDS["psi_low"]
        ).count()

        avg_drift = total_psi / versions_with_drift if versions_with_drift > 0 else 0.0

        dataset_ids_with_drift = set(
            v.dataset_id for v in self.db.query(DatasetVersion).filter(
                DatasetVersion.drift_score >= DRIFT_THRESHOLDS["psi_low"]
            ).all()
        )

        return {
            "total_datasets": total_datasets,
            "datasets_with_drift": len(dataset_ids_with_drift),
            "avg_drift_score": avg_drift,
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "total_active_alerts": len(active_alerts),
            "recent_alerts": self.get_all_alerts(status="active", limit=20),
            "drift_history": self._get_drift_history()
        }

    def _get_drift_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        versions = self.db.query(DatasetVersion).filter(
            DatasetVersion.drift_score.isnot(None)
        ).order_by(DatasetVersion.created_at.desc()).limit(limit).all()

        history = []
        for v in versions:
            dataset = self.db.query(Dataset).filter(Dataset.id == v.dataset_id).first()
            history.append({
                "dataset_id": v.dataset_id,
                "dataset_name": dataset.name if dataset else "",
                "version_number": v.version_number,
                "drift_score": v.drift_score,
                "timestamp": v.created_at.isoformat()
            })

        return history

    def acknowledge_alert(self, alert_id: str) -> Optional[DriftAlert]:
        alert = self.db.query(DriftAlert).filter(DriftAlert.id == alert_id).first()

        if not alert:
            return None

        alert.status = "acknowledged"
        alert.acknowledged_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(alert)

        return alert

    def resolve_alert(self, alert_id: str) -> Optional[DriftAlert]:
        alert = self.db.query(DriftAlert).filter(DriftAlert.id == alert_id).first()

        if not alert:
            return None

        alert.status = "resolved"
        alert.resolved_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(alert)

        return alert

    def acknowledge_all_alerts(self, dataset_id: Optional[str] = None) -> int:
        query = self.db.query(DriftAlert).filter(DriftAlert.status == "active")

        if dataset_id:
            query = query.filter(DriftAlert.dataset_id == dataset_id)

        alerts = query.all()
        count = 0

        for alert in alerts:
            alert.status = "acknowledged"
            alert.acknowledged_at = datetime.utcnow()
            count += 1

        self.db.commit()

        return count
