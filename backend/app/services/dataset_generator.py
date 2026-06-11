from typing import List, Dict, Any, Optional
import uuid
import logging
import json
import random
from app.core.extractors import get_extractor
from app.core.llm import get_llm_adapter
from app.models.schemas import (
    QuestionAnswer, QuestionDifficulty, QuestionType, 
    QuestionSource, GenerationPreset, PRESET_DISTRIBUTIONS
)

logger = logging.getLogger(__name__)


ADVERSARIAL_TEMPLATES = [
    {
        "template": "What is NOT mentioned about {entity}?",
        "trap_type": "negation",
        "requires_entity": True
    },
    {
        "template": "According to the document, what is the salary of {entity}?",
        "trap_type": "false_pretense",
        "requires_entity": True
    },
    {
        "template": "Who is the CTO of {entity}?",
        "trap_type": "specific_entity",
        "requires_entity": True
    },
    {
        "template": "What year did {event} happen?",
        "trap_type": "temporal",
        "requires_entity": True
    },
    {
        "template": "According to the document, {claim}. Is this true?",
        "trap_type": "overclaiming",
        "requires_entity": False
    },
    {
        "template": "What is the exact revenue figure mentioned in the document?",
        "trap_type": "specific_number",
        "requires_entity": False
    },
    {
        "template": "Does the document mention {entity}?",
        "trap_type": "entity_test",
        "requires_entity": True
    },
    {
        "template": "What specific metrics are provided about {entity}?",
        "trap_type": "specific_detail",
        "requires_entity": True
    },
]


class DifficultyClassifier:
    def __init__(self):
        self.llm = None

    def _get_llm(self):
        if self.llm is None:
            self.llm = get_llm_adapter("openai")
        return self.llm

    def classify(self, question: str, answer: str, context: List[str]) -> tuple[QuestionDifficulty, float]:
        prompt = f"""Analyze this question-answer pair and score its difficulty.

Question: {question}
Answer: {answer}
Context: {' '.join(context)[:500]}

Score difficulty based on:
1. Reasoning steps required (0-1)
2. Document chunks needed to answer (0-1)
3. Inference complexity (0-1)
4. Ambiguity (0-1)

Respond in JSON format:
{{
    "reasoning_steps": 0.0-1.0,
    "chunks_needed": 0.0-1.0,
    "inference_complexity": 0.0-1.0,
    "ambiguity": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "difficulty": "easy", "medium", or "hard"
}}

Overall score calculation:
- reasoning_steps * 0.3 + chunks_needed * 0.3 + inference_complexity * 0.2 + ambiguity * 0.2

Score interpretation:
- > 0.7: hard
- 0.4-0.7: medium
- < 0.4: easy"""

        try:
            llm = self._get_llm()
            response = llm.generate(prompt)
            result = json.loads(response)
            score = result.get("overall_score", 0.5)
            
            if score > 0.7:
                difficulty = QuestionDifficulty.HARD
            elif score >= 0.4:
                difficulty = QuestionDifficulty.MEDIUM
            else:
                difficulty = QuestionDifficulty.EASY
            
            return difficulty, score
        except Exception as e:
            logger.warning(f"Difficulty classification failed: {e}")
            return QuestionDifficulty.MEDIUM, 0.5


class DatasetGeneratorService:
    def __init__(self):
        self.llm = None
        self.classifier = DifficultyClassifier()
        self._adversarial_templates = ADVERSARIAL_TEMPLATES

    def _get_llm(self):
        if self.llm is None:
            self.llm = get_llm_adapter("openai")
        return self.llm

    def generate_questions(
        self,
        text: str,
        total_questions: int = 50,
        distribution: Optional[Dict[str, float]] = None,
        preset: Optional[GenerationPreset] = None,
        verify_difficulty: bool = True
    ) -> Dict[str, Any]:
        if not text:
            return {"questions": [], "distribution": {}, "total_generated": 0, "verified_difficulties": False}

        if preset and distribution is None:
            distribution = PRESET_DISTRIBUTIONS[preset]
        elif distribution is None:
            distribution = PRESET_DISTRIBUTIONS[GenerationPreset.BALANCED]

        easy_count = int(total_questions * distribution.get("easy", 0.4))
        medium_count = int(total_questions * distribution.get("medium", 0.3))
        hard_count = int(total_questions * distribution.get("hard", 0.2))
        adversarial_count = total_questions - easy_count - medium_count - hard_count

        chunks = self._chunk_text(text, chunk_size=2000)

        easy_qs = self._generate_easy_questions(chunks, easy_count)
        medium_qs = self._generate_medium_questions(chunks, medium_count)
        hard_qs = self._generate_hard_questions(chunks, hard_count)
        adversarial_qs = self._generate_adversarial_questions(chunks, adversarial_count, text)

        all_questions = easy_qs + medium_qs + hard_qs + adversarial_qs
        random.shuffle(all_questions)

        verified_questions = []
        if verify_difficulty:
            verified_questions = self._verify_difficulties(all_questions)
        else:
            for q in all_questions:
                qa = QuestionAnswer(
                    question=q["question"],
                    answer=q["answer"],
                    source=q["source"],
                    context=q["context"],
                    difficulty=q.get("difficulty", QuestionDifficulty.MEDIUM),
                    question_type=q.get("question_type", QuestionType.FACTUAL),
                    source_type=q.get("source_type", QuestionSource.CONTENT),
                    traps=q.get("traps", [])
                )
                verified_questions.append(qa)

        result_distribution = {
            "easy": len(easy_qs),
            "medium": len(medium_qs),
            "hard": len(hard_qs),
            "adversarial": len(adversarial_qs)
        }

        return {
            "questions": verified_questions,
            "distribution": result_distribution,
            "total_generated": len(verified_questions),
            "verified_difficulties": verify_difficulty
        }

    def _generate_easy_questions(self, chunks: List[str], count: int) -> List[Dict[str, Any]]:
        questions = []
        for i in range(count):
            chunk = chunks[i % len(chunks)] if chunks else ""
            q = self._generate_single_qa(chunk, "content", QuestionDifficulty.EASY, QuestionType.FACTUAL)
            if q:
                q["difficulty"] = QuestionDifficulty.EASY
                q["question_type"] = QuestionType.FACTUAL
                questions.append(q)
        return questions

    def _generate_medium_questions(self, chunks: List[str], count: int) -> List[Dict[str, Any]]:
        questions = []
        for i in range(count):
            chunk = chunks[i % len(chunks)] if chunks else ""
            q = self._generate_single_qa(chunk, "content", QuestionDifficulty.MEDIUM, QuestionType.INFERENTIAL)
            if q:
                q["difficulty"] = QuestionDifficulty.MEDIUM
                q["question_type"] = QuestionType.INFERENTIAL
                questions.append(q)
        return questions

    def _generate_hard_questions(self, chunks: List[str], count: int) -> List[Dict[str, Any]]:
        questions = []
        for i in range(count):
            combined_context = ""
            if i < len(chunks) - 1:
                combined_context = chunks[i] + " " + chunks[i + 1]
            elif chunks:
                combined_context = chunks[i % len(chunks)]
            else:
                combined_context = ""
            q = self._generate_hard_qa(combined_context, "content")
            if q:
                questions.append(q)
        return questions

    def _generate_adversarial_questions(
        self, 
        chunks: List[str], 
        count: int,
        full_text: str
    ) -> List[Dict[str, Any]]:
        questions = []
        content_count = int(count * 0.7)
        template_count = count - content_count

        for i in range(content_count):
            chunk = chunks[i % len(chunks)] if chunks else ""
            q = self._generate_adversarial_content_aware(chunk, "content")
            if q:
                questions.append(q)

        for i in range(template_count):
            q = self._generate_adversarial_from_template(full_text, chunks)
            if q:
                questions.append(q)

        return questions

    def _generate_hard_qa(self, context: str, source: str) -> Optional[Dict[str, Any]]:
        prompt = f"""Based on the following context, generate ONE complex analytical question that requires multi-step reasoning.

Context: {context[:2000]}

Generate a question that requires:
- Comparing multiple concepts or tradeoffs
- Multi-step reasoning or analysis
- Deep understanding of the technical content
- Evaluating claims or making reasoned arguments

Respond in JSON format:
{{
    "question": "your complex question here",
    "answer": "the analytical answer based on the context",
    "source": "{source}",
    "difficulty": "hard",
    "question_type": "analytical"
}}

Only respond with valid JSON, no explanations."""

        try:
            llm = self._get_llm()
            response = llm.generate(prompt)
            qa = json.loads(response)
            if "question" in qa and "answer" in qa:
                return {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "source": qa.get("source", source),
                    "context": [context],
                    "difficulty": QuestionDifficulty.HARD,
                    "question_type": QuestionType.ANALYTICAL,
                    "source_type": QuestionSource.CONTENT,
                    "traps": []
                }
        except Exception as e:
            logger.warning(f"Hard QA generation failed: {e}")

        return self._fallback_hard_qa(context, source)

    def _generate_adversarial_content_aware(self, context: str, source: str) -> Optional[Dict[str, Any]]:
        prompt = f"""Based on the following context, generate ONE question that tests hallucination resistance.

Context: {context[:2000]}

Generate a question that:
- Tests whether the model will hallucinate information NOT in the context
- Contains specific entity names or dates that sound plausible but may not be in the context
- Has a false premise that the model might accept
- Tests temporal reasoning about events not mentioned

Respond in JSON format:
{{
    "question": "your adversarial question here",
    "answer": "Information not available in the provided context",
    "source": "{source}",
    "difficulty": "adversarial",
    "traps": ["list of trap types used"]
}}

Only respond with valid JSON, no explanations."""

        try:
            llm = self._get_llm()
            response = llm.generate(prompt)
            qa = json.loads(response)
            if "question" in qa:
                return {
                    "question": qa["question"],
                    "answer": qa.get("answer", "Information not available in the provided context"),
                    "source": qa.get("source", source),
                    "context": [context],
                    "difficulty": QuestionDifficulty.ADVERSARIAL,
                    "question_type": QuestionType.ADVERSARIAL,
                    "source_type": QuestionSource.CONTENT,
                    "traps": qa.get("traps", ["content_hallucination"])
                }
        except Exception as e:
            logger.warning(f"Adversarial content-aware generation failed: {e}")

        return self._fallback_adversarial(context, source)

    def _generate_adversarial_from_template(
        self, 
        full_text: str, 
        chunks: List[str]
    ) -> Optional[Dict[str, Any]]:
        import re
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', full_text[:3000])
        entities = list(set([e for e in entities if len(e) > 2]))[:10]

        template = random.choice(self._adversarial_templates)

        entity = random.choice(entities) if entities and template.get("requires_entity") else "the company"

        question_text = template["template"].format(entity=entity)

        prompt = f"""Check if this question is answerable from the context:

Question: {question_text}
Context: {full_text[:2000]}

Respond in JSON:
{{
    "answerable": true or false,
    "answer": "the answer if answerable, or 'Information not available' if not"
}}

Only respond with valid JSON."""

        try:
            llm = self._get_llm()
            response = llm.generate(prompt)
            result = json.loads(response)

            return {
                "question": question_text,
                "answer": result.get("answer", "Information not available in the provided context"),
                "source": "template",
                "context": chunks[:2] if chunks else [full_text[:500]],
                "difficulty": QuestionDifficulty.ADVERSARIAL,
                "question_type": QuestionType.ADVERSARIAL,
                "source_type": QuestionSource.TEMPLATE,
                "traps": [template["trap_type"]]
            }
        except Exception as e:
            logger.warning(f"Adversarial template generation failed: {e}")
            return {
                "question": question_text,
                "answer": "Information not available in the provided context",
                "source": "template",
                "context": chunks[:1] if chunks else [],
                "difficulty": QuestionDifficulty.ADVERSARIAL,
                "question_type": QuestionType.ADVERSARIAL,
                "source_type": QuestionSource.TEMPLATE,
                "traps": [template["trap_type"]]
            }

    def _generate_single_qa(
        self, 
        context: str, 
        source: str,
        difficulty: QuestionDifficulty,
        question_type: QuestionType
    ) -> Optional[Dict[str, Any]]:
        prompt = f"""Based on the following context, generate ONE question-answer pair.

Context: {context[:1500]}

Difficulty: {difficulty.value}
Question Type: {question_type.value}

"""

        if difficulty == QuestionDifficulty.EASY:
            prompt += """Generate a simple factual question focusing on:
- Who, what, when, where information
- Direct recall from the context
- No inference required
"""
        elif difficulty == QuestionDifficulty.MEDIUM:
            prompt += """Generate a question requiring understanding and basic inference:
- How and why questions
- Some context clues needed
- Basic comprehension
"""
        else:
            prompt += """Generate an analytical question requiring:
- Comparing concepts
- Tradeoff evaluation
- Complex reasoning
"""

        prompt += """
Respond in JSON format:
{
    "question": "your question here",
    "answer": "the answer based on the context",
    "source": "{source}"
}

Only respond with valid JSON, no explanations."""

        try:
            llm = self._get_llm()
            response = llm.generate(prompt)
            qa = json.loads(response)

            if "question" in qa and "answer" in qa:
                return {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "source": qa.get("source", source),
                    "context": [context],
                    "difficulty": difficulty,
                    "question_type": question_type,
                    "source_type": QuestionSource.CONTENT,
                    "traps": []
                }
        except Exception as e:
            logger.warning(f"QA generation failed: {e}")

        return None

    def _verify_difficulties(self, questions: List[Dict[str, Any]]) -> List[QuestionAnswer]:
        verified = []
        for q in questions:
            try:
                if q.get("source_type") == QuestionSource.TEMPLATE:
                    difficulty = q.get("difficulty", QuestionDifficulty.ADVERSARIAL)
                    score = 0.9
                else:
                    difficulty, score = self.classifier.classify(
                        q["question"],
                        q["answer"],
                        q.get("context", [])
                    )

                qa = QuestionAnswer(
                    question=q["question"],
                    answer=q["answer"],
                    source=q["source"],
                    context=q.get("context", []),
                    difficulty=difficulty,
                    question_type=q.get("question_type", QuestionType.FACTUAL),
                    source_type=q.get("source_type", QuestionSource.CONTENT),
                    traps=q.get("traps", []),
                    difficulty_score=score
                )
                verified.append(qa)
            except Exception as e:
                logger.warning(f"Verification failed for question: {e}")
                qa = QuestionAnswer(
                    question=q["question"],
                    answer=q["answer"],
                    source=q["source"],
                    context=q.get("context", []),
                    difficulty=q.get("difficulty", QuestionDifficulty.MEDIUM),
                    question_type=q.get("question_type", QuestionType.FACTUAL),
                    source_type=q.get("source_type", QuestionSource.CONTENT),
                    traps=q.get("traps", [])
                )
                verified.append(qa)

        return verified

    def _fallback_qa(self, context: str, source: str) -> Optional[Dict[str, Any]]:
        import re
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return None

        key_sentence = sentences[0]

        question_prompt = f"""Create a question from this statement: "{key_sentence[:200]}"

Respond with just the question."""

        try:
            llm = self._get_llm()
            question = llm.generate(question_prompt).strip()

            return {
                "question": question,
                "answer": key_sentence.strip(),
                "source": source,
                "context": [context],
                "difficulty": QuestionDifficulty.EASY,
                "question_type": QuestionType.FACTUAL,
                "source_type": QuestionSource.CONTENT,
                "traps": []
            }
        except:
            return {
                "question": f"What is mentioned about {source}?",
                "answer": key_sentence.strip() if key_sentence else "Information not available",
                "source": source,
                "context": [context],
                "difficulty": QuestionDifficulty.EASY,
                "question_type": QuestionType.FACTUAL,
                "source_type": QuestionSource.CONTENT,
                "traps": []
            }

    def _fallback_hard_qa(self, context: str, source: str) -> Optional[Dict[str, Any]]:
        return {
            "question": f"What are the key implications of the information provided about {source}?",
            "answer": context[:500] if context else "No information available",
            "source": source,
            "context": [context] if context else [],
            "difficulty": QuestionDifficulty.HARD,
            "question_type": QuestionType.ANALYTICAL,
            "source_type": QuestionSource.CONTENT,
            "traps": []
        }

    def _fallback_adversarial(self, context: str, source: str) -> Optional[Dict[str, Any]]:
        return {
            "question": f"According to the document, what specific metrics are provided?",
            "answer": "Information not available in the provided context",
            "source": source,
            "context": [context] if context else [],
            "difficulty": QuestionDifficulty.ADVERSARIAL,
            "question_type": QuestionType.ADVERSARIAL,
            "source_type": QuestionSource.CONTENT,
            "traps": ["specific_number"]
        }

    def _chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            start = end

        return chunks

    def extract_and_chunk(self, file_path: str, source_type: str) -> tuple[str, List[str]]:
        extractor = get_extractor(source_type)
        text = extractor.extract(file_path)
        chunks = extractor.chunk(text)
        return text, chunks
