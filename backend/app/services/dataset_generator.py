from typing import List, Dict, Any, Optional
import uuid
import logging
from app.core.extractors import get_extractor
from app.core.llm import get_llm_adapter

logger = logging.getLogger(__name__)


class DatasetGeneratorService:
    def __init__(self):
        self.llm = None

    def generate_qa_pairs(
        self,
        text: str,
        num_pairs: int = 10,
        source: str = "document"
    ) -> List[Dict[str, Any]]:
        if not text:
            return []
        
        chunks = self._chunk_text(text, chunk_size=2000)
        
        qa_pairs = []
        for i, chunk in enumerate(chunks[:num_pairs]):
            qa = self._generate_single_qa(chunk, source, i)
            if qa:
                qa_pairs.append(qa)
        
        if len(qa_pairs) < num_pairs:
            for i in range(len(chunks), min(len(chunks) * 2, num_pairs)):
                qa = self._generate_single_qa(chunks[i % len(chunks)] if chunks else text, source, i)
                if qa:
                    qa_pairs.append(qa)
                    if len(qa_pairs) >= num_pairs:
                        break
        
        return qa_pairs[:num_pairs]

    def _generate_single_qa(self, context: str, source: str, index: int) -> Optional[Dict[str, Any]]:
        prompt = f"""Based on the following context, generate ONE question-answer pair that tests understanding of the content.

Context: {context[:1500]}

Generate a question that:
1. Tests factual recall from the context
2. Has a specific, verifiable answer

Respond in JSON format:
{{
    "question": "your question here",
    "answer": "the answer based on the context",
    "source": "{source}"
}}

Only respond with valid JSON, no explanations."""

        try:
            llm = get_llm_adapter("openai")
            response = llm.generate(prompt)
            
            import json
            qa = json.loads(response)
            
            if "question" in qa and "answer" in qa:
                return {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "source": qa.get("source", source),
                    "context": [context]
                }
        except Exception as e:
            logger.warning(f"QA generation failed for chunk {index}: {e}")
        
        return self._fallback_qa(context, source)

    def _fallback_qa(self, context: str, source: str) -> Optional[Dict[str, Any]]:
        import re
        
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return None
        
        key_sentence = sentences[0] if sentences else ""
        
        question_prompt = f"""Create a question from this statement: "{key_sentence[:200]}"

Respond with just the question."""
        
        try:
            llm = get_llm_adapter("openai")
            question = llm.generate(question_prompt).strip()
            
            return {
                "question": question,
                "answer": key_sentence.strip(),
                "source": source,
                "context": [context]
            }
        except:
            return {
                "question": f"What is mentioned about {source}?",
                "answer": key_sentence.strip() if key_sentence else "Information not available",
                "source": source,
                "context": [context]
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
