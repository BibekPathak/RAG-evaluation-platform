from typing import List, Dict, Any, Optional
import logging
import time
import numpy as np
from app.core.retriever import get_vector_db_adapter, VectorDBAdapter
from app.core.llm import get_llm_adapter
from app.core.evaluator import Evaluator

logger = logging.getLogger(__name__)


class RetrievalEvalService:
    def __init__(self):
        self.evaluator = Evaluator()
        self.vector_dbs: Dict[str, VectorDBAdapter] = {}

    def run_retrieval_evaluation(
        self,
        questions: List[str],
        contexts: List[List[str]],
        embeddings: List[List[List[float]]],
        vector_db: str = "chroma",
        top_k: int = 5
    ) -> Dict[str, float]:
        metrics = {
            "context_recall": 0.0,
            "context_precision": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "hit_rate": 0.0
        }
        
        if not questions or not contexts or not embeddings:
            return metrics
        
        reciprocal_ranks = []
        ndcg_scores = []
        hits = 0
        
        for i, (question_embs, relevant_contexts) in enumerate(zip(embeddings, contexts)):
            if not relevant_contexts:
                continue
            
            relevant_set = set(relevant_contexts)
            retrieved_set = set()
            
            for j, emb in enumerate(question_embs[:top_k]):
                if j < len(contexts[i]):
                    retrieved_set.add(contexts[i][j])
            
            correct = len(relevant_set & retrieved_set)
            recall = correct / len(relevant_set) if relevant_set else 0
            precision = correct / len(retrieved_set) if retrieved_set else 0
            
            metrics["context_recall"] += recall
            metrics["context_precision"] += precision
            
            if correct > 0:
                for rank, ctx in enumerate(contexts[i][:top_k], 1):
                    if ctx in relevant_set:
                        reciprocal_ranks.append(1.0 / rank)
                        ndcg_scores.append(1.0 / np.log2(rank + 1))
                        hits += 1
                        break
        
        n = len(questions)
        if n > 0:
            metrics["context_recall"] /= n
            metrics["context_precision"] /= n
            metrics["mrr"] = sum(reciprocal_ranks) / n if reciprocal_ranks else 0
            metrics["ndcg"] = sum(ndcg_scores) / n if ndcg_scores else 0
            metrics["hit_rate"] = hits / n
        
        return metrics

    def benchmark_retrieval(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        queries: List[str],
        query_embeddings: List[List[float]],
        relevant_indices: List[List[int]],
        vector_dbs: List[str] = ["chroma", "qdrant", "faiss"],
        top_k: int = 5
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        
        for db_name in vector_dbs:
            try:
                db = get_vector_db_adapter(db_name)
                db.add_embeddings(texts, embeddings)
                
                all_retrieved = []
                for query_emb in query_embeddings:
                    retrieved = db.search(query_emb, top_k=top_k)
                    all_retrieved.append([r[0] for r in retrieved])
                
                retrieval_metrics = self.run_retrieval_evaluation(
                    questions=queries,
                    contexts=all_retrieved,
                    embeddings=[[qe] for qe in query_embeddings],
                    vector_db=db_name,
                    top_k=top_k
                )
                
                results[db_name] = retrieval_metrics
                
            except Exception as e:
                logger.error(f"Benchmark failed for {db_name}: {e}")
                results[db_name] = {"error": str(e)}
        
        return results

    def measure_latency(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        query_embeddings: List[List[float]],
        vector_db: str = "chroma",
        iterations: int = 10
    ) -> Dict[str, float]:
        db = get_vector_db_adapter(vector_db)
        db.add_embeddings(texts, embeddings)
        
        latencies = []
        for _ in range(iterations):
            for query_emb in query_embeddings:
                start = time.time()
                db.search(query_emb, top_k=5)
                latencies.append((time.time() - start) * 1000)
        
        return {
            "mean_ms": np.mean(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99)
        }
