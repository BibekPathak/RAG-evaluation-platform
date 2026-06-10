from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    source: str
    context: List[str] = []


class DatasetBase(BaseModel):
    name: str
    source_type: str = "pdf"


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    id: str
    source_path: Optional[str] = None
    question_count: int = 0
    questions: List[QuestionAnswer] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetUploadResponse(BaseModel):
    dataset: DatasetResponse
    chunks_created: int
    questions_generated: int


class RetrievalMetrics(BaseModel):
    context_recall: float = Field(ge=0, le=1, description="Context recall score")
    context_precision: float = Field(ge=0, le=1, description="Context precision score")
    mrr: float = Field(ge=0, le=1, description="Mean reciprocal rank")
    ndcg: float = Field(ge=0, le=1, description="Normalized discounted cumulative gain")
    hit_rate: float = Field(ge=0, le=1, description="Hit rate at k")


class GenerationMetrics(BaseModel):
    faithfulness: float = Field(ge=0, le=1, description="Faithfulness score")
    answer_relevancy: float = Field(ge=0, le=1, description="Answer relevancy score")
    context_utilization: float = Field(ge=0, le=1, description="Context utilization score")
    hallucination_rate: float = Field(ge=0, le=1, description="Hallucination rate")


class MetricsSummary(BaseModel):
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    hallucination_rate: Optional[float] = None
    mrr: Optional[float] = None
    ndcg: Optional[float] = None
    hit_rate: Optional[float] = None


class ExperimentBase(BaseModel):
    name: str
    description: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    llm: str = "gpt-4o"
    retriever: str = "chroma"


class ExperimentCreate(ExperimentBase):
    dataset_id: Optional[str] = None
    vector_db_config: Optional[Dict[str, Any]] = {}
    llm_config: Optional[Dict[str, Any]] = {}


class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    hallucination_rate: Optional[float] = None
    mrr: Optional[float] = None
    ndcg: Optional[float] = None
    hit_rate: Optional[float] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None


class ExperimentResponse(ExperimentBase):
    id: str
    metrics: MetricsSummary
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    dataset_id: Optional[str] = None

    class Config:
        from_attributes = True

    @classmethod
    def from_orm_with_metrics(cls, exp) -> "ExperimentResponse":
        return cls(
            id=exp.id,
            name=exp.name,
            description=exp.description,
            embedding_model=exp.embedding_model,
            llm=exp.llm,
            retriever=exp.retriever,
            metrics=MetricsSummary(
                context_recall=exp.context_recall,
                context_precision=exp.context_precision,
                faithfulness=exp.faithfulness,
                answer_relevancy=exp.answer_relevancy,
                hallucination_rate=exp.hallucination_rate,
                mrr=exp.mrr,
                ndcg=exp.ndcg,
                hit_rate=exp.hit_rate,
            ),
            latency_ms=exp.latency_ms,
            cost_usd=exp.cost_usd,
            created_at=exp.created_at,
            updated_at=exp.updated_at,
            dataset_id=exp.dataset_id,
        )


class EvaluationRequest(BaseModel):
    dataset_id: str
    question: str
    ground_truth_answer: Optional[str] = None


class RetrievalEvaluationRequest(BaseModel):
    dataset_id: str
    questions: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)


class GenerationEvaluationRequest(BaseModel):
    experiment_id: str
    questions: Optional[List[str]] = None


class HallucinationRequest(BaseModel):
    answer: str
    context: List[str]


class HallucinationResponse(BaseModel):
    claims: List[str]
    verified_claims: List[str]
    hallucinated_claims: List[str]
    hallucination_score: float = Field(ge=0, le=1)
    total_claims: int
    verified_count: int
    hallucinated_count: int


class JudgeRequest(BaseModel):
    question: str
    ground_truth_answer: str
    generated_answer: str
    context: List[str] = []
    judge_model: str = "gpt-4o"


class JudgeResponse(BaseModel):
    score: float = Field(ge=0, le=1)
    reasoning: str
    dimensions: Dict[str, float]


class BenchmarkConfig(BaseModel):
    embedding_models: List[str] = ["text-embedding-3-small", "bge-large"]
    vector_dbs: List[str] = ["chroma", "qdrant", "faiss"]
    llms: List[str] = ["gpt-4o", "claude-3-opus"]
    dataset_id: str
    sample_size: int = Field(default=10, ge=1, le=100)


class BenchmarkRunRequest(BenchmarkConfig):
    name: str = "Benchmark Run"


class BenchmarkResult(BaseModel):
    embedding_model: str
    vector_db: str
    llm: str
    metrics: MetricsSummary
    latency_ms: float
    cost_usd: float
    experiment_id: str


class BenchmarkResponse(BaseModel):
    name: str
    total_runs: int
    results: List[BenchmarkResult]
    best_overall: Optional[str] = None
    best_retrieval: Optional[str] = None
    best_generation: Optional[str] = None


class CompareRequest(BaseModel):
    experiment_ids: List[str] = Field(min_length=2, max_length=5)


class CompareResponse(BaseModel):
    experiments: List[ExperimentResponse]
    winning_metrics: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]
