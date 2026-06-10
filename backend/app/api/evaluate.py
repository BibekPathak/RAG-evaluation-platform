from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import time
import uuid
from app.models.database import get_db
from app.models.models import Dataset, Experiment, EvaluationResult
from app.models.schemas import (
    RetrievalEvaluationRequest,
    GenerationEvaluationRequest,
    RetrievalMetrics,
    GenerationMetrics,
    MetricsSummary
)
from app.services.retrieval_eval import RetrievalEvalService
from app.services.generation_eval import GenerationEvalService
from app.services.experiment_tracker import ExperimentTracker
from app.core.llm import get_llm_adapter
from app.core.retriever import get_vector_db_adapter

router = APIRouter()


@router.post("/retrieval", response_model=RetrievalMetrics)
async def evaluate_retrieval(
    request: RetrievalEvaluationRequest,
    db = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    questions_data = dataset.questions
    if not questions_data:
        raise HTTPException(status_code=400, detail="Dataset has no questions")
    
    questions = [q.get("question", "") for q in questions_data]
    contexts = [q.get("context", []) for q in questions_data]
    
    llm = get_llm_adapter("openai")
    
    embeddings = []
    for ctx_list in contexts:
        if ctx_list:
            emb = llm.embed([ctx_list[0]])[0]
            embeddings.append([[emb]])
        else:
            embeddings.append([[]])
    
    retrieval_service = RetrievalEvalService()
    metrics = retrieval_service.run_retrieval_evaluation(
        questions=questions,
        contexts=contexts,
        embeddings=embeddings,
        vector_db="chroma",
        top_k=request.top_k
    )
    
    return RetrievalMetrics(**metrics)


@router.post("/generation", response_model=GenerationMetrics)
async def evaluate_generation(
    request: GenerationEvaluationRequest,
    db = Depends(get_db)
):
    experiment = db.query(Experiment).filter(Experiment.id == request.experiment_id).first()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    dataset_id = experiment.dataset_id
    if not dataset_id:
        raise HTTPException(status_code=400, detail="Experiment has no associated dataset")
    
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    questions_data = dataset.questions
    if not questions_data:
        raise HTTPException(status_code=400, detail="Dataset has no questions")
    
    questions = [q.get("question", "") for q in questions_data]
    ground_truths = [q.get("answer", "") for q in questions_data]
    contexts = [q.get("context", []) for q in questions_data]
    
    llm = get_llm_adapter(experiment.llm)
    
    generated_answers = []
    for q in questions:
        prompt = f"Context: {' '.join(contexts[questions.index(q)] if contexts else [])}\n\nQuestion: {q}\n\nAnswer:"
        answer = llm.generate(prompt)
        generated_answers.append(answer)
    
    generation_service = GenerationEvalService()
    metrics = generation_service.run_full_evaluation(
        questions=questions,
        answers=generated_answers,
        contexts=contexts,
        ground_truths=ground_truths
    )
    
    return GenerationMetrics(**metrics)


@router.post("/full")
async def run_full_evaluation(
    dataset_id: str,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o",
    retriever: str = "chroma",
    db = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    questions_data = dataset.questions
    if not questions_data:
        raise HTTPException(status_code=400, detail="Dataset has no questions")
    
    questions = [q.get("question", "") for q in questions_data]
    ground_truths = [q.get("answer", "") for q in questions_data]
    contexts = [q.get("context", []) for q in questions_data]
    
    start_time = time.time()
    
    llm = get_llm_adapter(llm_model)
    vector_db = get_vector_db_adapter(retriever)
    
    all_chunks = []
    for ctx_list in contexts:
        all_chunks.extend(ctx_list)
    all_chunks = list(dict.fromkeys(all_chunks))
    
    if all_chunks:
        embeddings = llm.embed(all_chunks)
        vector_db.add_embeddings(all_chunks, embeddings)
    
    retrieval_service = RetrievalEvalService()
    query_embeddings = [llm.embed([q])[0] for q in questions]
    
    retrieved_contexts = []
    for qe in query_embeddings:
        results = vector_db.search(qe, top_k=5)
        retrieved_contexts.append([r[0] for r in results])
    
    retrieval_metrics = retrieval_service.run_retrieval_evaluation(
        questions=questions,
        contexts=retrieved_contexts,
        embeddings=[[qe] for qe in query_embeddings],
        vector_db=retriever,
        top_k=5
    )
    
    generated_answers = []
    for i, q in enumerate(questions):
        prompt = f"Context: {' '.join(retrieved_contexts[i])}\n\nQuestion: {q}\n\nProvide a clear, accurate answer based on the context."
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
    
    tracker = ExperimentTracker(db)
    experiment = tracker.create_experiment(
        ExperimentCreate(
            name=f"Eval-{embedding_model}-{llm_model}-{retriever}",
            embedding_model=embedding_model,
            llm=llm_model,
            retriever=retriever,
            dataset_id=dataset_id
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
    
    tracker.update_metrics(experiment.id, combined_metrics, latency_ms=latency_ms)
    
    return {
        "experiment_id": experiment.id,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "latency_ms": latency_ms
    }
