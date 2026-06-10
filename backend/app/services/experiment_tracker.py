from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from app.models.database import SessionLocal
from app.models.models import Experiment, Dataset, EvaluationResult
from app.models.schemas import ExperimentCreate, MetricsSummary
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, db: Session):
        self.db = db

    def create_experiment(self, experiment_data: ExperimentCreate) -> Experiment:
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=experiment_data.name,
            description=experiment_data.description,
            embedding_model=experiment_data.embedding_model,
            llm=experiment_data.llm,
            retriever=experiment_data.retriever,
            vector_db_config=experiment_data.vector_db_config or {},
            llm_config=experiment_data.llm_config or {},
            dataset_id=experiment_data.dataset_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        
        return experiment

    def update_metrics(
        self,
        experiment_id: str,
        metrics: MetricsSummary,
        latency_ms: Optional[float] = None,
        cost_usd: Optional[float] = None
    ) -> Optional[Experiment]:
        experiment = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        
        if not experiment:
            return None
        
        if metrics.context_recall is not None:
            experiment.context_recall = metrics.context_recall
        if metrics.context_precision is not None:
            experiment.context_precision = metrics.context_precision
        if metrics.faithfulness is not None:
            experiment.faithfulness = metrics.faithfulness
        if metrics.answer_relevancy is not None:
            experiment.answer_relevancy = metrics.answer_relevancy
        if metrics.hallucination_rate is not None:
            experiment.hallucination_rate = metrics.hallucination_rate
        if metrics.mrr is not None:
            experiment.mrr = metrics.mrr
        if metrics.ndcg is not None:
            experiment.ndcg = metrics.ndcg
        if metrics.hit_rate is not None:
            experiment.hit_rate = metrics.hit_rate
        
        if latency_ms is not None:
            experiment.latency_ms = latency_ms
        if cost_usd is not None:
            experiment.cost_usd = cost_usd
        
        experiment.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(experiment)
        
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        return self.db.query(Experiment).filter(Experiment.id == experiment_id).first()

    def list_experiments(
        self,
        skip: int = 0,
        limit: int = 100,
        embedding_model: Optional[str] = None,
        llm: Optional[str] = None,
        retriever: Optional[str] = None
    ) -> List[Experiment]:
        query = self.db.query(Experiment)
        
        if embedding_model:
            query = query.filter(Experiment.embedding_model == embedding_model)
        if llm:
            query = query.filter(Experiment.llm == llm)
        if retriever:
            query = query.filter(Experiment.retriever == retriever)
        
        return query.order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()

    def delete_experiment(self, experiment_id: str) -> bool:
        experiment = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        
        if not experiment:
            return False
        
        self.db.delete(experiment)
        self.db.commit()
        
        return True

    def get_metric_history(
        self,
        experiment_id: str,
        metric_name: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        experiment = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        
        if not experiment:
            return []
        
        metric_map = {
            "context_recall": experiment.context_recall,
            "context_precision": experiment.context_precision,
            "faithfulness": experiment.faithfulness,
            "answer_relevancy": experiment.answer_relevancy,
            "hallucination_rate": experiment.hallucination_rate,
            "mrr": experiment.mrr,
            "ndcg": experiment.ndcg,
            "hit_rate": experiment.hit_rate,
        }
        
        return [{
            "timestamp": experiment.created_at.isoformat(),
            "value": metric_map.get(metric_name)
        }]

    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        experiments = self.db.query(Experiment).filter(
            Experiment.id.in_(experiment_ids)
        ).all()
        
        if not experiments:
            return {"experiments": [], "winning_metrics": {}}
        
        metrics = ["context_recall", "context_precision", "faithfulness", "answer_relevancy", "mrr", "ndcg"]
        
        winning_metrics = {}
        
        for metric in metrics:
            best_exp = None
            best_value = -1
            
            for exp in experiments:
                value = getattr(exp, metric, None)
                if value is not None and value > best_value:
                    best_value = value
                    best_exp = exp
            
            if best_exp:
                winning_metrics[metric] = best_exp.id
        
        return {
            "experiments": experiments,
            "winning_metrics": winning_metrics
        }

    def get_best_experiment(self, metric: str = "faithfulness") -> Optional[Experiment]:
        return self.db.query(Experiment).filter(
            getattr(Experiment, metric) != None
        ).order_by(
            getattr(Experiment, metric).desc()
        ).first()


import uuid
