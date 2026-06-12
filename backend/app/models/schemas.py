from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum
from uuid import UUID


class QuestionDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class QuestionType(str, Enum):
    FACTUAL = "factual"
    INFERENTIAL = "inferential"
    ANALYTICAL = "analytical"
    ADVERSARIAL = "adversarial"


class QuestionSource(str, Enum):
    CONTENT = "content"
    TEMPLATE = "template"


class GenerationPreset(str, Enum):
    BALANCED = "balanced"
    HARD_EVALUATION = "hard_evaluation"
    ADVERSARIAL_HEAVY = "adversarial_heavy"
    RETRIEVAL_STRESS = "retrieval_stress"


PRESET_DISTRIBUTIONS = {
    GenerationPreset.BALANCED: {"easy": 0.4, "medium": 0.3, "hard": 0.2, "adversarial": 0.1},
    GenerationPreset.HARD_EVALUATION: {"easy": 0.2, "medium": 0.3, "hard": 0.3, "adversarial": 0.2},
    GenerationPreset.ADVERSARIAL_HEAVY: {"easy": 0.3, "medium": 0.2, "hard": 0.2, "adversarial": 0.3},
    GenerationPreset.RETRIEVAL_STRESS: {"easy": 0.2, "medium": 0.4, "hard": 0.3, "adversarial": 0.1},
}


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    source: str
    context: List[str] = []
    difficulty: QuestionDifficulty = QuestionDifficulty.MEDIUM
    question_type: QuestionType = QuestionType.FACTUAL
    source_type: QuestionSource = QuestionSource.CONTENT
    traps: List[str] = []
    difficulty_score: Optional[float] = None


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
    judge_model: str = "mock"


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


class GenerateQuestionsRequest(BaseModel):
    document_id: str
    total_questions: int = Field(default=50, ge=10, le=200)
    distribution: Optional[Dict[str, float]] = None
    preset: Optional[GenerationPreset] = None
    verify_difficulty: bool = True


class GenerateQuestionsResponse(BaseModel):
    questions: List[QuestionAnswer]
    distribution: Dict[str, int]
    total_generated: int
    verified_difficulties: bool
    generation_stats: Dict[str, Any] = {}


class MetricChange(BaseModel):
    old: float
    new: float
    change_pct: float
    psi: float


class DriftResult(BaseModel):
    psi: float
    drift_detected: bool
    severity: str
    metric_changes: Dict[str, MetricChange]
    alerts: List[Dict[str, Any]]


class DatasetVersionBase(BaseModel):
    version_number: int
    change_summary: Optional[str] = None


class DatasetVersionResponse(BaseModel):
    id: str
    dataset_id: str
    version_number: int
    version_hash: str
    question_count: int
    drift_score: Optional[float] = None
    drift_details: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DriftAlertBase(BaseModel):
    metric: str
    old_value: float
    new_value: float
    change_pct: float
    psi: float
    severity: str


class DriftAlertResponse(BaseModel):
    id: str
    dataset_id: str
    dataset_version_id: str
    dataset_name: Optional[str] = None
    metric: str
    old_value: float
    new_value: float
    change_pct: float
    psi: float
    severity: str
    status: str
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DriftSummary(BaseModel):
    dataset_id: str
    dataset_name: str
    current_version: int
    total_versions: int
    current_drift_score: Optional[float] = None
    drift_detected: bool
    active_alerts: int
    acknowledged_alerts: int
    resolved_alerts: int
    recent_versions: List[DatasetVersionResponse]
    metrics_comparison: Optional[Dict[str, MetricChange]] = None


class DriftStats(BaseModel):
    total_datasets: int
    datasets_with_drift: int
    avg_drift_score: float
    critical_alerts: int
    warning_alerts: int
    total_active_alerts: int
    recent_alerts: List[DriftAlertResponse]
    drift_history: List[Dict[str, Any]]


class VersionComparison(BaseModel):
    dataset_id: str
    dataset_name: str
    version_a: DatasetVersionResponse
    version_b: DatasetVersionResponse
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    metric_changes: Dict[str, MetricChange]
    drift_result: DriftResult
