from typing import List, Dict, Any, Optional
import logging
from app.core.llm import get_llm_adapter
from app.core.retriever import get_vector_db_adapter

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self):
        self.llm = None
        self.vector_db = None

    def evaluate_retrieval(
        self,
        questions: List[str],
        contexts: List[List[str]],
        embeddings: List[List[List[float]]],
        top_k: int = 5
    ) -> Dict[str, float]:
        metrics = {
            "context_recall": 0.0,
            "context_precision": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "hit_rate": 0.0
        }
        
        if not questions or not contexts:
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

    def evaluate_generation(
        self,
        questions: List[str],
        generated_answers: List[str],
        ground_truths: List[str],
        contexts: List[List[str]],
        framework: str = "ragas"
    ) -> Dict[str, float]:
        if framework == "ragas":
            return self._evaluate_with_ragas(questions, generated_answers, ground_truths, contexts)
        elif framework == "deepeval":
            return self._evaluate_with_deepeval(questions, generated_answers, ground_truths, contexts)
        else:
            return self._simple_evaluation(questions, generated_answers, ground_truths, contexts)

    def _evaluate_with_ragas(self, questions, generated_answers, ground_truths, contexts) -> Dict[str, float]:
        try:
            from ragas import Evaluator
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_utilization,
            )
            from datasets import Dataset
            
            eval_dataset = Dataset.from_dict({
                "question": questions,
                "answer": generated_answers,
                "ground_truth": ground_truths,
                "contexts": contexts
            })
            
            evaluator = Evaluator(llm=self.llm)
            results = evaluator.evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy])
            
            return {
                "faithfulness": results["faithfulness"],
                "answer_relevancy": results["answer_relevancy"],
                "context_utilization": results.get("context_utilization", 0.0),
                "hallucination_rate": 1.0 - results["faithfulness"]
            }
        except Exception as e:
            logger.warning(f"Ragas evaluation failed: {e}. Falling back to simple evaluation.")
            return self._simple_evaluation(questions, generated_answers, ground_truths, contexts)

    def _evaluate_with_deepeval(self, questions, generated_answers, ground_truths, contexts) -> Dict[str, float]:
        try:
            from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
            from deepeval import evaluate as deepeval_evaluate
            
            hallucination_scores = []
            relevancy_scores = []
            
            for question, answer, context in zip(questions, generated_answers, contexts):
                try:
                    hallucination_metric = HallucinationMetric()
                    hallucination_score = hallucination_metric.measure(
                        response=answer,
                        context=context[0] if context else ""
                    )
                    hallucination_scores.append(hallucination_score)
                except:
                    hallucination_scores.append(0.0)
                
                try:
                    relevancy_metric = AnswerRelevancyMetric()
                    relevancy_score = relevancy_metric.measure(
                        question=question,
                        response=answer
                    )
                    relevancy_scores.append(relevancy_score)
                except:
                    relevancy_scores.append(0.0)
            
            return {
                "faithfulness": 1.0 - (sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0),
                "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0,
                "context_utilization": sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0,
                "hallucination_rate": sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0
            }
        except Exception as e:
            logger.warning(f"DeepEval evaluation failed: {e}. Falling back to simple evaluation.")
            return self._simple_evaluation(questions, generated_answers, ground_truths, contexts)

    def _simple_evaluation(self, questions, generated_answers, ground_truths, contexts) -> Dict[str, float]:
        import re
        
        faithfulness_scores = []
        relevancy_scores = []
        
        for answer, context in zip(generated_answers, contexts):
            if not answer or not context:
                faithfulness_scores.append(0.0)
                continue
            
            answer_lower = answer.lower()
            context_combined = " ".join(context).lower()
            
            answer_words = set(re.findall(r'\w+', answer_lower))
            context_words = set(re.findall(r'\w+', context_combined))
            
            if answer_words:
                overlap = len(answer_words & context_words) / len(answer_words)
                faithfulness_scores.append(overlap)
            else:
                faithfulness_scores.append(0.0)
            
            relevancy_scores.append(faithfulness_scores[-1])
        
        return {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
            "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0,
            "context_utilization": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
            "hallucination_rate": 1.0 - (sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0)
        }

    def evaluate_hallucination(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
        import re
        
        claims = self._extract_claims(answer)
        
        verified_claims = []
        hallucinated_claims = []
        
        context_combined = " ".join(contexts).lower() if contexts else ""
        
        for claim in claims:
            claim_lower = claim.lower()
            claim_words = set(re.findall(r'\w+', claim_lower))
            context_words = set(re.findall(r'\w+', context_combined))
            
            if context_words and claim_words:
                overlap = len(claim_words & context_words) / len(claim_words)
                if overlap > 0.5:
                    verified_claims.append(claim)
                else:
                    hallucinated_claims.append(claim)
            else:
                hallucinated_claims.append(claim)
        
        hallucination_score = len(hallucinated_claims) / len(claims) if claims else 0.0
        
        return {
            "claims": claims,
            "verified_claims": verified_claims,
            "hallucinated_claims": hallucinated_claims,
            "hallucination_score": hallucination_score,
            "total_claims": len(claims),
            "verified_count": len(verified_claims),
            "hallucinated_count": len(hallucinated_claims)
        }

    def _extract_claims(self, text: str) -> List[str]:
        import re
        
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and any(char.isupper() for char in sent[:10]):
                claims.append(sent)
            elif len(sent) > 30:
                claims.append(sent)
        
        return claims


import numpy as np
