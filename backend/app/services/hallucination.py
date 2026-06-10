from typing import List, Dict, Any, Optional
import logging
import re
from app.core.llm import get_llm_adapter

logger = logging.getLogger(__name__)


class HallucinationService:
    def __init__(self):
        pass

    def extract_claims(self, text: str) -> List[str]:
        if not text:
            return []
        
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:
                if any(char.isupper() for char in sent[:10] if char.isalpha()):
                    claims.append(sent)
                elif any(keyword in sent.lower() for keyword in ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'will', 'should']):
                    claims.append(sent)
                elif len(sent) > 30:
                    claims.append(sent)
        
        return claims

    def verify_claim_against_context(self, claim: str, context: str) -> float:
        if not claim or not context:
            return 0.0
        
        claim_words = set(re.findall(r'\w+', claim.lower()))
        context_words = set(re.findall(r'\w+', context.lower()))
        
        if not claim_words:
            return 0.0
        
        overlap = len(claim_words & context_words) / len(claim_words)
        return overlap

    def detect_hallucination(
        self,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        if not answer:
            return {
                "claims": [],
                "verified_claims": [],
                "hallucinated_claims": [],
                "hallucination_score": 0.0,
                "total_claims": 0,
                "verified_count": 0,
                "hallucinated_count": 0
            }
        
        claims = self.extract_claims(answer)
        
        if not claims:
            return {
                "claims": [],
                "verified_claims": [],
                "hallucinated_claims": [],
                "hallucination_score": 0.0,
                "total_claims": 0,
                "verified_count": 0,
                "hallucinated_count": 0
            }
        
        context_combined = " ".join(contexts) if contexts else ""
        
        verified_claims = []
        hallucinated_claims = []
        
        for claim in claims:
            verification_score = self.verify_claim_against_context(claim, context_combined)
            if verification_score > 0.5:
                verified_claims.append(claim)
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

    def detect_with_llm(
        self,
        answer: str,
        contexts: List[str],
        model: str = "gpt-4o"
    ) -> Dict[str, Any]:
        prompt = f"""You are an expert at detecting hallucinations in AI-generated answers.

Given an answer and supporting context, identify any hallucinated (fabricated or incorrect) claims.

Context:
{chr(10).join([f"- {c[:500]}" for c in contexts[:3]]) if contexts else "No context provided"}

Answer to evaluate:
{answer}

For each claim in the answer:
1. Determine if it is supported by the context
2. Identify any hallucinated or fabricated information

Respond in JSON format:
{{
    "claims": ["list of all claims identified"],
    "verified_claims": ["claims supported by context"],
    "hallucinated_claims": ["claims not supported or contradicted by context"],
    "hallucination_score": 0.0-1.0 (proportion of hallucinated claims),
    "explanation": "detailed explanation of the evaluation"
}}

Only respond with valid JSON."""

        try:
            llm = get_llm_adapter(model)
            response = llm.generate(prompt)
            
            import json
            result = json.loads(response)
            return result
        except Exception as e:
            logger.error(f"LLM hallucination detection failed: {e}")
            return self.detect_hallucination(answer, contexts)

    def batch_detect(
        self,
        answers: List[str],
        contexts: List[List[str]],
        use_llm: bool = False,
        model: str = "gpt-4o"
    ) -> List[Dict[str, Any]]:
        results = []
        
        for answer, context_list in zip(answers, contexts):
            if use_llm:
                result = self.detect_with_llm(answer, context_list, model)
            else:
                result = self.detect_hallucination(answer, context_list)
            results.append(result)
        
        return results

    def aggregate_hallucination_rate(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        if not results:
            return 0.0
        
        scores = [r.get("hallucination_score", 0.0) for r in results]
        return sum(scores) / len(scores) if scores else 0.0
