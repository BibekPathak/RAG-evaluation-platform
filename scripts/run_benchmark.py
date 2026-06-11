#!/usr/bin/env python3
"""
RAG Benchmark Runner

Usage:
    python scripts/run_benchmark.py --config benchmark_config.json --output evaluations/reports/latest.json

This script runs a full benchmark evaluation using the specified configuration
and outputs results in JSON format.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.llm import get_llm_adapter, MockAdapter
from app.core.retriever import ChromaAdapter
from app.services.retrieval_eval import RetrievalEvalService
from app.services.generation_eval import GenerationEvalService


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load evaluation dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def load_config(config_path: str) -> Dict:
    """Load benchmark configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def estimate_cost(llm_id: str, num_queries: int) -> float:
    """Estimate API cost in USD."""
    costs = {
        "huggingface": 0.0,
        "openai-gpt4": 0.00001,
        "openai-gpt35": 0.000002,
        "mock": 0.0
    }
    return costs.get(llm_id, 0.0) * num_queries


def run_retrieval_eval(
    questions: List[str],
    contexts: List[List[str]],
    embeddings: List[List[List[float]]]
) -> Dict[str, float]:
    """Run retrieval evaluation metrics."""
    service = RetrievalEvalService()
    metrics = service.run_retrieval_evaluation(
        questions=questions,
        contexts=contexts,
        embeddings=embeddings,
        vector_db="chroma",
        top_k=5
    )
    return metrics


def run_generation_eval(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str]
) -> Dict[str, float]:
    """Run generation evaluation metrics."""
    service = GenerationEvalService()
    metrics = service.run_full_evaluation(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )
    return metrics


def run_benchmark(config: Dict, embedding_adapter, llm_adapter) -> Dict[str, Any]:
    """Run the full benchmark evaluation."""
    print("Starting RAG Benchmark Evaluation...")
    print("=" * 50)

    all_questions = []
    all_contexts = []
    all_ground_truths = []
    all_answers = []

    for dataset_key in ["standard", "hard", "adversarial"]:
        if dataset_key not in config["datasets"]:
            continue

        dataset_info = config["datasets"][dataset_key]
        dataset_path = dataset_info["path"]

        if not Path(dataset_path).exists():
            print(f"Warning: Dataset not found: {dataset_path}")
            continue

        print(f"Loading {dataset_key} dataset from {dataset_path}...")
        dataset = load_dataset(dataset_path)

        for item in dataset:
            all_questions.append(item["question"])
            all_contexts.append(item.get("context", []))
            all_ground_truths.append(item["answer"])

    if not all_questions:
        print("Error: No questions loaded from datasets")
        return {}

    print(f"\nTotal questions to evaluate: {len(all_questions)}")

    print("\n[1/4] Generating embeddings...")
    start_time = time.time()

    all_chunks = []
    for ctx_list in all_contexts:
        all_chunks.extend(ctx_list)
    all_chunks = list(dict.fromkeys(all_chunks))

    if all_chunks:
        embeddings = embedding_adapter.embed(all_chunks[:100])
    else:
        embeddings = [[]]

    embeddings_time = time.time() - start_time
    print(f"Embedding generation took {embeddings_time:.2f}s")

    print("\n[2/4] Setting up vector store...")
    vector_db = ChromaAdapter(persist_directory="./chroma_db_eval")

    if all_chunks and embeddings:
        vector_db.add_embeddings(all_chunks[:100], embeddings)

    print("\n[3/4] Running retrieval evaluation...")
    query_embeddings = embedding_adapter.embed(all_questions[:20])

    retrieved_contexts = []
    for qe in query_embeddings[:20]:
        results = vector_db.search(qe, top_k=3)
        retrieved_contexts.append([r[0] for r in results])

    retrieval_metrics = run_retrieval_eval(
        questions=all_questions[:20],
        contexts=retrieved_contexts,
        embeddings=[[qe] for qe in query_embeddings[:20]]
    )

    print("\n[4/4] Running generation evaluation...")
    generated_answers = []
    for i, q in enumerate(all_questions[:10]):
        prompt = f"Context: {' '.join(retrieved_contexts[i] if i < len(retrieved_contexts) else [])}\n\nQuestion: {q}\n\nProvide a clear, accurate answer based on the context."
        answer = llm_adapter.generate(prompt)
        generated_answers.append(answer)

    generation_metrics = run_generation_eval(
        questions=all_questions[:10],
        answers=generated_answers,
        contexts=retrieved_contexts[:10] if len(retrieved_contexts) >= 10 else retrieved_contexts,
        ground_truths=all_ground_truths[:10]
    )

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "embedding_model": "hf-embedding",
            "llm": "mock",
            "retriever": "chroma"
        },
        "datasets": {
            "total_questions": len(all_questions),
            "evaluated_retrieval": 20,
            "evaluated_generation": 10
        },
        "metrics": {
            **retrieval_metrics,
            **generation_metrics
        },
        "performance": {
            "total_latency_ms": total_time * 1000,
            "avg_latency_ms": (total_time * 1000) / max(len(all_questions), 1),
            "estimated_cost_usd": estimate_cost("mock", len(all_questions))
        }
    }

    print("\nRetrieval Metrics:")
    for key, value in retrieval_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nGeneration Metrics:")
    for key, value in generation_metrics.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nTotal Time: {total_time:.2f}s")
    print(f"Estimated Cost: ${results['performance']['estimated_cost_usd']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run RAG Benchmark Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.json",
        help="Path to benchmark configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluations/reports/latest.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="mock",
        help="Embedding provider (mock, huggingface, openai)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="mock",
        help="LLM provider (mock, huggingface, openai)"
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as baseline"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Using embedding adapter: {args.embedding}")
    print(f"Using LLM adapter: {args.llm}")

    embedding_adapter = get_llm_adapter(args.embedding)
    llm_adapter = get_llm_adapter(args.llm)

    results = run_benchmark(config, embedding_adapter, llm_adapter)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    if args.save_baseline:
        baseline_path = Path("evaluations/baselines/baseline_metrics.json")
        baseline_path.parent.mkdir(parents=True, exist_ok=True)

        baseline = {
            "version": "1.0.0",
            "created_at": results["timestamp"],
            "metrics": results["metrics"],
            "performance": results["performance"],
            "config": results["config"]
        }

        with open(baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)

        print(f"Baseline saved to: {baseline_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
