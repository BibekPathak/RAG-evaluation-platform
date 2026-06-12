import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Float, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.models.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    embedding_model = Column(String(100), nullable=False)
    llm = Column(String(100), nullable=False)
    retriever = Column(String(100), nullable=False)
    vector_db_config = Column(JSON, default=dict)
    llm_config = Column(JSON, default=dict)
    context_recall = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    hallucination_rate = Column(Float, nullable=True)
    mrr = Column(Float, nullable=True)
    ndcg = Column(Float, nullable=True)
    hit_rate = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    cost_usd = Column(Float, nullable=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True)
    dataset_version_id = Column(String(36), ForeignKey("dataset_versions.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="experiments")
    evaluation_results = relationship("EvaluationResult", back_populates="experiment")
    dataset_version = relationship("DatasetVersion", back_populates="experiments")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    source_type = Column(String(50), nullable=False)
    source_path = Column(String(500), nullable=True)
    question_count = Column(Integer, default=0)
    questions = Column(JSON, default=list)
    current_version_id = Column(String(36), ForeignKey("dataset_versions.id"), nullable=True)
    version_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    experiments = relationship("Experiment", back_populates="dataset")
    versions = relationship("DatasetVersion", back_populates="dataset", order_by="desc(DatasetVersion.version_number)")
    current_version = relationship("DatasetVersion", foreign_keys=[current_version_id], post_update=True)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    version_hash = Column(String(64), nullable=False)
    question_count = Column(Integer, default=0)
    questions_snapshot = Column(JSON, default=list)
    metrics_summary = Column(JSON, default=dict)
    drift_score = Column(Float, nullable=True)
    drift_details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="versions")
    experiments = relationship("Experiment", back_populates="dataset_version")
    alerts = relationship("DriftAlert", back_populates="dataset_version", order_by="desc(DriftAlert.created_at)")


class DriftAlert(Base):
    __tablename__ = "drift_alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    dataset_version_id = Column(String(36), ForeignKey("dataset_versions.id"), nullable=False)
    metric = Column(String(50), nullable=False)
    old_value = Column(Float, nullable=False)
    new_value = Column(Float, nullable=False)
    change_pct = Column(Float, nullable=False)
    psi = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    dataset = relationship("Dataset")
    dataset_version = relationship("DatasetVersion", back_populates="alerts")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    question = Column(Text, nullable=False)
    retrieved_contexts = Column(JSON, default=list)
    generated_answer = Column(Text, nullable=True)
    ground_truth_answer = Column(Text, nullable=True)
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_utilization = Column(Float, nullable=True)
    hallucination_score = Column(Float, nullable=True)
    judge_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    experiment = relationship("Experiment", back_populates="evaluation_results")
