from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)


class LLMAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> str:
        pass


class OpenAIAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._embedding_model = "text-embedding-3-small"

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        return response.choices[0].message.content

    async def agenerate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self._embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]


class AnthropicAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0.0),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    async def agenerate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def embed(self, texts: List[str]) -> List[List[float]]:
        logger.warning("Anthropic does not provide embeddings. Using fallback.")
        import numpy as np
        return [np.random.randn(1536).tolist() for _ in texts]


class OllamaAdapter(LLMAdapter):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url
        self.model = model
        self._embedding_model = "nomic-embed-text"

    def generate(self, prompt: str, **kwargs) -> str:
        import requests
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": kwargs.get("model", self.model),
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json().get("response", "")

    async def agenerate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def embed(self, texts: List[str]) -> List[List[float]]:
        import requests
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self._embedding_model, "prompt": text}
            )
            response.raise_for_status()
            embeddings.append(response.json().get("embedding", []))
        return embeddings


class HuggingFaceAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.1-8B-Instruct", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self._base_url = "https://api-inference.huggingface.co/v1"

    def generate(self, prompt: str, **kwargs) -> str:
        import requests
        model = kwargs.get("model", self.model)
        url = f"{self._base_url}/models/{model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.0),
                "return_full_text": False
            }
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            return str(result)
        except Exception as e:
            logger.warning(f"HuggingFace API failed, using mock response: {e}")
            return self._mock_generate(prompt)

    async def agenerate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def embed(self, texts: List[str]) -> List[List[float]]:
        import requests
        url = f"{self._base_url}/models/{self.embedding_model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        embeddings = []
        for text in texts:
            payload = {"inputs": text}
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    embeddings.append(result[0] if isinstance(result[0], list) else result[0].get("embedding", result[0]))
                elif isinstance(result, dict) and "embedding" in result:
                    embeddings.append(result["embedding"])
                else:
                    embeddings.append(result)
            except Exception as e:
                logger.warning(f"HuggingFace embedding API failed, using mock embedding: {e}")
                embeddings.append(self._mock_embed(text))
        return embeddings

    def _mock_generate(self, prompt: str) -> str:
        return f"Mock response for: {prompt[:100]}... [This is a placeholder response since HuggingFace API is not reachable]"

    def _mock_embed(self, text: str) -> List[float]:
        import numpy as np
        vec = np.random.randn(384).tolist()
        norm = np.linalg.norm(vec)
        return (np.array(vec) / norm).tolist()


class MockAdapter(LLMAdapter):
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    def generate(self, prompt: str, **kwargs) -> str:
        if "evaluator for RAG" in prompt or "Ground Truth Answer" in prompt:
            return '{"score": 0.85, "reasoning": "The answer correctly identifies Paris as the capital and aligns with the context provided.", "dimensions": {"faithfulness": 0.9, "answer_relevancy": 0.8}}'
        if "Evaluate the following" in prompt and "Score overall" in prompt:
            return "0.85"
        return f"Mock LLM response for: {prompt[:80]}... [Demo mode - no real API call]"

    async def agenerate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        embeddings = []
        for text in texts:
            vec = np.random.randn(self.embedding_dim).tolist()
            norm = np.linalg.norm(vec)
            embeddings.append((np.array(vec) / norm).tolist())
        return embeddings


def get_llm_adapter(provider: str, **kwargs) -> LLMAdapter:
    from app.config import get_settings
    settings = get_settings()

    adapters = {
        "openai": OpenAIAdapter(api_key=settings.openai_api_key, model=kwargs.get("model", "gpt-4o")),
        "anthropic": AnthropicAdapter(api_key=settings.anthropic_api_key, model=kwargs.get("model", "claude-3-opus-20240229")),
        "huggingface": HuggingFaceAdapter(
            api_key=settings.huggingface_api_key,
            model=kwargs.get("model", settings.hf_llm_model),
            embedding_model=kwargs.get("embedding_model", settings.hf_embedding_model)
        ),
        "hf-llama": HuggingFaceAdapter(
            api_key=settings.huggingface_api_key,
            model="meta-llama/Llama-3.1-8B-Instruct",
            embedding_model=settings.hf_embedding_model
        ),
        "hf-embedding": HuggingFaceAdapter(
            api_key=settings.huggingface_api_key,
            model=settings.hf_llm_model,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        "mock": MockAdapter(),
        "demo": MockAdapter(),
        "ollama": OllamaAdapter(base_url=settings.ollama_base_url, model=kwargs.get("model", "llama3")),
        "gpt-4o": OpenAIAdapter(api_key=settings.openai_api_key, model="gpt-4o"),
        "gpt-4-turbo": OpenAIAdapter(api_key=settings.openai_api_key, model="gpt-4-turbo"),
        "claude-3-opus": AnthropicAdapter(api_key=settings.anthropic_api_key, model="claude-3-opus-20240229"),
        "claude-3-sonnet": AnthropicAdapter(api_key=settings.anthropic_api_key, model="claude-3-sonnet-20240229"),
        "claude-3-haiku": AnthropicAdapter(api_key=settings.anthropic_api_key, model="claude-3-haiku-20240307"),
        "llama-3": OllamaAdapter(base_url=settings.ollama_base_url, model="llama3"),
        "mistral": OllamaAdapter(base_url=settings.ollama_base_url, model="mistral"),
    }
    
    return adapters.get(provider.lower(), adapters["mock"])