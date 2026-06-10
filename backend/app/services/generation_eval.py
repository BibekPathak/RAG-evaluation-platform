from typing import List, Dict, Any, Optional
import logging
import re
from app.core.llm import get_llm_adapter

logger = logging.getLogger(__name__)


class GenerationEvalService:
    def __init__(self):
        pass

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        if not answer or not contexts:
            return 0.0
        
        answer_lower = answer.lower()
        context_combined = " ".join(contexts).lower()
        
        answer_words = set(re.findall(r'\w+', answer_lower))
        context_words = set(re.findall(r'\w+', context_combined))
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words & context_words) / len(answer_words)
        return overlap

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        if not question or not answer:
            return 0.0
        
        question_words = set(re.findall(r'\w+', question.lower()))
        answer_words = set(re.findall(r'\w+', answer.lower()))
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words & answer_words) / len(question_words)
        return min(overlap * 2, 1.0)

    def evaluate_context_utilization(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        if not answer or not contexts:
            return 0.0
        
        answer_words = set(re.findall(r'\w+', answer.lower()))
        
        utilized = 0
        total = 0
        
        for context in contexts:
            context_words = set(re.findall(r'\w+', context.lower()))
            if context_words:
                overlap = len(answer_words & context_words) / len(context_words)
                if overlap > 0.2:
                    utilized += 1
                total += 1
        
        return utilized / total if total > 0 else 0.0

    def run_full_evaluation(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        if not questions or not answers or not contexts:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_utilization": 0.0,
                "hallucination_rate": 0.0
            }
        
        faithfulness_scores = []
        relevancy_scores = []
        utilization_scores = []
        
        for question, answer, context_list in zip(questions, answers, contexts):
            faithfulness_scores.append(self.evaluate_faithfulness(answer, context_list))
            relevancy_scores.append(self.evaluate_answer_relevancy(question, answer))
            utilization_scores.append(self.evaluate_context_utilization(answer, context_list))
        
        n = len(questions)
        hallucination_rate = 1.0 - (sum(faithfulness_scores) / n if n > 0 else 0.0)
        
        return {
            "faithfulness": sum(faithfulness_scores) / n if n > 0 else 0.0,
            "answer_relevancy": sum(relevancy_scores) / n if n > 0 else 0.0,
            "context_utilization": sum(utilization_scores) / n if n > 0 else 0.0,
            "hallucination_rate": hallucination_rate
        }

    def evaluate_with_llm_judge(
        self,
        question: str,
        ground_truth: str,
        generated_answer: str,
        context: List[str],
        judge_model: str = "gpt-4o"
    ) -> Dict[str, Any]:
        prompt = f"""You are an expert evaluator for RAG (Retrieval Augmented Generation) systems.

Evaluate the generated answer against the ground truth answer.

Context:
{chr(10).join(context[:3]) if context else "No context provided"}

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {generated_answer}

Evaluate on a scale of 0-1 for each dimension:
1. Faithfulness (does the answer stick to the context?)
2. Answer Relevancy (does the answer address the question?)
3. Overall Quality

Respond in JSON format:
{{
    "faithfulness": 0.0-1.0,
    "answer_relevancy": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Only respond with valid JSON."""

        try:
            llm = get_llm_adapter(judge_model)
            response = llm.generate(prompt)
            
            import json
            result = json.loads(response)
            return result
        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            return {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "overall_score": 0.5,
                "reasoning": f"Evaluation failed: {str(e)}"
            }
