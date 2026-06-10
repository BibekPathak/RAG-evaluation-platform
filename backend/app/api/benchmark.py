from fastapi import APIRouter, HTTPException, Depends
from typing import List
import time
import uuid
from app.models.database import get_db
from app.models.models import Dataset, Experiment
from app.models.schemas import (
    BenchmarkRunRequest,
    BenchmarkResponse,
    BenchmarkResult,
    MetricsSummary
)
from app.services.experiment_tracker import ExperimentTracker
from app.core.llm import get_llm_adapter
from app.core.retriever import get_vector_db_adapter
from app.services.retrieval_eval import RetrievalEvalService
from app.services.generation_eval import GenerationEvalService

router = APIRouter()


@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(
    request: BenchmarkRunRequest,
    db = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    questions_data = dataset.questions[:request.sample_size]
    if not questions_data:
        raise HTTPException(status_code=400, detail="Dataset has insufficient questions")
    
    questions = [q.get("question", "") for q in questions_data]
    ground_truths = [q.get("answer", "") for q in questions_data]
    contexts = [q.get("context", []) for q in questions_data]
    
    all_chunks = []
    for ctx_list in contexts:
        all_chunks.extend(ctx_list)
    all_chunks = list(dict.fromkeys(all_chunks))
    
    results = []
    
    for embedding_model in request.embedding_models:
        for vector_db_name in request.vector_dbs:
            for llm_name in request.llms:
                try:
                    start_time = time.time()
                    
                    llm = get_llm_adapter(llm_name)
                    vector_db = get_vector_db_adapter(vector_db_name)
                    
                    embeddings = llm.embed(all_chunks)
                    vector_db.add_embeddings(all_chunks, embeddings)
                    
                    query_embeddings = [llm.embed([q])[0] for q in questions]
                    
                    retrieved_contexts = []
                    for qe in query_embeddings:
                        results_vec = vector_db.search(qe, top_k=5)
                        retrieved_contexts.append([r[0] for r in results_vec])
                    
                    retrieval_service = RetrievalEvalService()
                    retrieval_metrics = retrieval_service.run_retrieval_evaluation(
                        questions=questions,
                        contexts=retrieved_contexts,
                        embeddings=[[qe] for qe in query_embeddings],
                        vector_db=vector_db_name,
                        top_k=5
                    )
                    
                    generated_answers = []
                    for i, q in enumerate(questions):
                        prompt = f"Context: {' '.join(retrieved_contexts[i])}\n\nQuestion: {q}\n\nAnswer:"
                        answer = llm.generate(prompt)
                        generated_answers.append(answer)
                    
                    generation_service = GenerationEvalService()
                    generation_metrics = generation_service.run_full_evaluation(
                        questions=questions,
                        answers=generated_answers,
                        contexts=retrieved_contexts,
                        ground_truths=ground_truths
                    )
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    cost_usd = _estimate_cost(llm_name, len(questions), len(all_chunks))
                    
                    tracker = ExperimentTracker(db)
                    experiment = tracker.create_experiment(
                        ExperimentCreate(
                            name=f"Benchmark-{embedding_model}-{vector_db_name}-{llm_name}",
                            embedding_model=embedding_model,
                            llm=llm_name,
                            retriever=vector_db_name,
                            dataset_id=request.dataset_id
                        )
                    )
                    
                    combined_metrics = MetricsSummary(
                        context_recall=retrieval_metrics.get("context_recall"),
                        context_precision=retrieval_metrics.get("context_precision"),
                        faithfulness=generation_metrics.get("faithfulness"),
                        answer_relevancy=generation_metrics.get("answer_relevancy"),
                        hallucination_rate=generation_metrics.get("hallucination_rate"),
                        mrr=retrieval_metrics.get("mrr"),
                        ndcg=retrieval_metrics.get("ndcg"),
                        hit_rate=retrieval_metrics.get("hit_rate")
                    )
                    
                    tracker.update_metrics(experiment.id, combined_metrics, latency_ms, cost_usd)
                    
                    benchmark_result = BenchmarkResult(
                        embedding_model=embedding_model,
                        vector_db=vector_db_name,
                        llm=llm_name,
                        metrics=combined_metrics,
                        latency_ms=latency_ms,
                        cost_usd=cost_usd,
                        experiment_id=experiment.id
                    )
                    results.append(benchmark_result)
                    
                except Exception as e:
                    results.append(BenchmarkResult(
                        embedding_model=embedding_model,
                        vector_db=vector_db_name,
                        llm=llm_name,
                        metrics=MetricsSummary(),
                        latency_ms=0,
                        cost_usd=0,
                        experiment_id=""
                    ))
    
    best_overall = _find_best_overall(results)
    best_retrieval = _find_best_retrieval(results)
    best_generation = _find_best_generation(results)
    
    return BenchmarkResponse(
        name=request.name,
        total_runs=len(results),
        results=results,
        best_overall=best_overall,
        best_retrieval=best_retrieval,
        best_generation=best_generation
    )


@router.get("/configs")
async def get_benchmark_configs():
    return {
        "embedding_models": [
            {"id": "text-embedding-3-small", "name": "OpenAI text-embedding-3-small", "dims": 1536},
            {"id": "text-embedding-3-large", "name": "OpenAI text-embedding-3-large", "dims": 3072},
            {"id": "bge-large", "name": "BAAI BGE-large", "dims": 1024},
            {"id": "e5-large", "name": "Microsoft E5-large", "dims": 1024},
        ],
        "vector_dbs": [
            {"id": "chroma", "name": "Chroma", "type": "local"},
            {"id": "qdrant", "name": "Qdrant", "type": "cloud"},
            {"id": "faiss", "name": "FAISS", "type": "local"},
        ],
        "llms": [
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI"},
            {"id": "gpt-4-turbo", "name": "GPT-4-turbo", "provider": "OpenAI"},
            {"id": "claude-3-opus", "name": "Claude 3 Opus", "provider": "Anthropic"},
            {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "provider": "Anthropic"},
            {"id": "llama-3", "name": "Llama 3", "provider": "Ollama"},
        ]
    }


def _estimate_cost(llm_name: str, num_questions: int, num_chunks: int) -> float:
    costs = {
        "gpt-4o": 0.00001,
        "gpt-4-turbo": 0.000005,
        "claude-3-opus": 0.000015,
        "claude-3-sonnet": 0.000003,
        "llama-3": 0.0,
    }
    return costs.get(llm_name, 0.00001) * num_questions


def _find_best_overall(results: List[BenchmarkResult]) -> str:
    best = None
    best_score = -1
    
    for r in results:
        if not r.experiment_id:
            continue
        score = (
            (r.metrics.context_recall or 0) * 0.2 +
            (r.metrics.faithfulness or 0) * 0.4 +
            (r.metrics.answer_relevancy or 0) * 0.4
        )
        if score > best_score:
            best_score = score
            best = r.experiment_id
    
    return best


def _find_best_retrieval(results: List[BenchmarkResult]) -> str:
    best = None
    best_score = -1
    
    for r in results:
        if not r.experiment_id:
            continue
        score = (r.metrics.context_recall or 0) * 0.5 + (r.metrics.mrr or 0) * 0.5
        if score > best_score:
            best_score = score
            best = r.experiment_id
    
    return best


def _find_best_generation(results: List[BenchmarkResult]) -> str:
    best = None
    best_score = -1
    
    for r in results:
        if not r.experiment_id:
            continue
        score = (r.metrics.faithfulness or 0) * 0.6 + (r.metrics.answer_relevancy or 0) * 0.4
        if score > best_score:
            best_score = score
            best = r.experiment_id
    
    return best
