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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    dataset = relationship("Dataset", back_populates="experiments")
    evaluation_results = relationship("EvaluationResult", back_populates="experiment")


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    source_type = Column(String(50), nullable=False)
    source_path = Column(String(500), nullable=True)
    question_count = Column(Integer, default=0)
    questions = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    experiments = relationship("Experiment", back_populates="dataset")


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
