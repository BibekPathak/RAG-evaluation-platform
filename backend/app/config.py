from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "RAG Evaluation Platform"
    debug: bool = True
    
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    huggingface_api_key: str = ""
    
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "rag_eval_embeddings"
    
    ollama_base_url: str = "http://localhost:11434"
    
    database_url: str = "sqlite:///./rag_eval.db"
    
    otel_service_name: str = "rag-eval-platform"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    
    default_embedding_model: str = "text-embedding-3-small"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    default_llm: str = "gpt-4o"
    
    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
