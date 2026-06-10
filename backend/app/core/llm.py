from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

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


def get_llm_adapter(provider: str, **kwargs) -> LLMAdapter:
    from app.config import get_settings
    settings = get_settings()

    adapters = {
        "openai": OpenAIAdapter(api_key=settings.openai_api_key, model=kwargs.get("model", "gpt-4o")),
        "anthropic": AnthropicAdapter(api_key=settings.anthropic_api_key, model=kwargs.get("model", "claude-3-opus-20240229")),
        "ollama": OllamaAdapter(base_url=settings.ollama_base_url, model=kwargs.get("model", "llama3")),
        "gpt-4o": OpenAIAdapter(api_key=settings.openai_api_key, model="gpt-4o"),
        "gpt-4-turbo": OpenAIAdapter(api_key=settings.openai_api_key, model="gpt-4-turbo"),
        "claude-3-opus": AnthropicAdapter(api_key=settings.anthropic_api_key, model="claude-3-opus-20240229"),
        "claude-3-sonnet": AnthropicAdapter(api_key=settings.anthropic_api_key, model="claude-3-sonnet-20240229"),
        "claude-3-haiku": AnthropicAdapter(api_key=settings.anthropic_api_key, model="claude-3-haiku-20240307"),
        "llama-3": OllamaAdapter(base_url=settings.ollama_base_url, model="llama3"),
        "mistral": OllamaAdapter(base_url=settings.ollama_base_url, model="mistral"),
    }
    
    return adapters.get(provider.lower(), adapters["openai"])
