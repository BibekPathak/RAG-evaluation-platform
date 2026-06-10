from fastapi import APIRouter, HTTPException
from app.models.schemas import HallucinationRequest, HallucinationResponse
from app.services.hallucination import HallucinationService

router = APIRouter()


@router.post("/detect", response_model=HallucinationResponse)
async def detect_hallucination(
    request: HallucinationRequest
):
    service = HallucinationService()
    
    result = service.detect_hallucination(
        answer=request.answer,
        contexts=request.context
    )
    
    return HallucinationResponse(**result)


@router.post("/detect-llm")
async def detect_hallucination_llm(
    answer: str,
    context: list[str],
    model: str = "gpt-4o"
):
    service = HallucinationService()
    
    result = service.detect_with_llm(
        answer=answer,
        contexts=context,
        model=model
    )
    
    return result


@router.post("/batch-detect")
async def batch_detect_hallucination(
    answers: list[str],
    contexts: list[list[str]],
    use_llm: bool = False,
    model: str = "gpt-4o"
):
    service = HallucinationService()
    
    results = service.batch_detect(
        answers=answers,
        contexts=contexts,
        use_llm=use_llm,
        model=model
    )
    
    aggregate_rate = service.aggregate_hallucination_rate(results)
    
    return {
        "individual_results": results,
        "aggregate_hallucination_rate": aggregate_rate
    }
