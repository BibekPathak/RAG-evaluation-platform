from fastapi import APIRouter, HTTPException
from typing import List, Dict
from app.models.schemas import JudgeRequest, JudgeResponse
from app.core.llm import get_llm_adapter

router = APIRouter()


@router.post("/evaluate", response_model=JudgeResponse)
async def judge_answer(
    request: JudgeRequest
):
    prompt = f"""You are an expert evaluator for RAG (Retrieval Augmented Generation) systems.
Your task is to evaluate how well a generated answer matches the ground truth answer.

Evaluate the generated answer on the following dimensions:
1. **Faithfulness** (0-1): Does the answer stick to facts from the provided context? If the answer contains information not supported by the context, score lower.
2. **Answer Relevancy** (0-1): Does the answer directly address the question asked?
3. **Overall Score** (0-1): General quality considering both dimensions above.

Context:
{chr(10).join([f"- {c[:300]}" for c in request.context[:3]]) if request.context else "No context provided"}

Question: {request.question}

Ground Truth Answer: {request.ground_truth_answer}

Generated Answer: {request.generated_answer}

Respond in JSON format with these exact fields:
{{
    "score": 0.0-1.0,
    "reasoning": "2-3 sentence explanation of your evaluation",
    "dimensions": {{
        "faithfulness": 0.0-1.0,
        "answer_relevancy": 0.0-1.0
    }}
}}

Only respond with valid JSON, no markdown or explanations outside the JSON."""

    try:
        llm = get_llm_adapter(request.judge_model)
        response_text = llm.generate(prompt)
        
        import json
        result = json.loads(response_text)
        
        return JudgeResponse(
            score=result.get("score", 0.5),
            reasoning=result.get("reasoning", "No reasoning provided"),
            dimensions=result.get("dimensions", {"faithfulness": 0.5, "answer_relevancy": 0.5})
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse judge response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Judge evaluation failed: {str(e)}")


@router.get("/models")
async def list_judge_models():
    return {
        "models": [
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI", "recommended": True},
            {"id": "gpt-4-turbo", "name": "GPT-4-turbo", "provider": "OpenAI", "recommended": False},
            {"id": "claude-3-opus", "name": "Claude 3 Opus", "provider": "Anthropic", "recommended": True},
            {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "provider": "Anthropic", "recommended": False},
        ]
    }


@router.post("/batch-evaluate")
async def batch_judge_answers(
    requests: List[JudgeRequest],
    judge_model: str = "gpt-4o"
) -> Dict:
    results = []
    total_score = 0.0
    
    for req in requests:
        prompt = f"""Evaluate the following answer:

Question: {req.question}
Ground Truth: {req.ground_truth_answer}
Generated Answer: {req.generated_answer}

Score overall quality from 0-1:"""
        
        try:
            llm = get_llm_adapter(judge_model)
            response = llm.generate(prompt)
            
            import re
            score_match = re.search(r'0?\.\d+|1\.0', response)
            score = float(score_match.group()) if score_match else 0.5
            
            results.append({
                "question": req.question,
                "score": score,
                "generated_answer": req.generated_answer
            })
            total_score += score
        except Exception as e:
            results.append({
                "question": req.question,
                "score": 0.0,
                "error": str(e)
            })
    
    avg_score = total_score / len(results) if results else 0.0
    
    return {
        "individual_results": results,
        "average_score": avg_score,
        "judge_model": judge_model
    }
