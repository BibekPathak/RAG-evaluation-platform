# Enterprise RAG Evaluation Platform

## Overview

A platform that answers: **"How do we know our RAG system is actually good?"**

This is a sophisticated RAG evaluation platform that goes beyond simple chatbots to demonstrate AI evaluation, LLMOps, benchmarking, observability, experiment tracking, hallucination detection, and production AI quality systems.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Dashboard                          │
│  (Experiment Tracking | Metrics | Benchmark Comparison | Runs) │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FastAPI Backend                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐│
│  │ Dataset API  │ │ Evaluate API │ │ Benchmark Runner API    ││
│  └──────────────┘ └──────────────┘ └──────────────────────────┘│
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐│
│  │ Experiment   │ │ Hallucination│ │ LLM-as-a-Judge API      ││
│  │ Tracking     │ │ Detection    │ │                          ││
│  └──────────────┘ └──────────────┘ └──────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
 ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│    Ragas      │    │   DeepEval    │    │  Custom Eval  │
│   (Primary)   │    │  (Secondary)  │    │    Engine     │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Vector DB Adapters (Chroma | Qdrant | FAISS)        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│           LLM Adapters (OpenAI | Claude | Local)                │
└─────────────────────────────────────────────────────────────────┘
```

## Core Features

### 1. Dataset Generator
- Upload PDF, Confluence, Website, Docs
- Auto-generate Q&A pairs: `{question, answer, source}`
- Store and manage evaluation datasets

### 2. Retrieval Evaluation
- **Context Recall** - How much relevant context was retrieved
- **Context Precision** - Quality of retrieved context
- **MRR** - Mean Reciprocal Rank
- **NDCG** - Normalized Discounted Cumulative Gain
- **Hit Rate** - Top-k retrieval accuracy

### 3. Generation Evaluation
- **Faithfulness** - Does answer match retrieved context
- **Answer Relevancy** - Does answer address the question
- **Context Utilization** - How well context is used
- **Hallucination Rate** - Factual errors in answer

### 4. LLM-as-a-Judge
- Use LLM to score answers against ground truth
- Judge agent with structured scoring rubric
- Multi-dimensional evaluation

### 5. Benchmark Runner
Compare across dimensions:
- **Embeddings**: OpenAI, BGE, E5
- **Vector DBs**: Chroma, Qdrant, FAISS
- **LLMs**: GPT-4o, Claude, Llama

### 6. Hallucination Detection
- Answer → Claims Extraction → Source Verification → Score
- Standout feature for portfolio

### 7. Experiment Tracking
- Store experiment configurations and results
- Like MLflow but for RAG
- Versioning and comparison

### 8. Dashboard
- Runs overview
- Top models comparison
- Hallucination % trends
- Faithfulness % trends
- Latency and cost metrics

## Data Models

### Experiment
```python
{
  "id": UUID,
  "name": str,
  "description": str,
  "embedding_model": str,  # "openai", "bge", "e5"
  "llm": str,               # "gpt-4o", "claude-3", "llama-3"
  "retriever": str,         # "qdrant", "chroma", "faiss"
  "vector_db_config": dict,
  "llm_config": dict,
  "metrics": {
    "context_recall": float,
    "context_precision": float,
    "faithfulness": float,
    "answer_relevancy": float,
    "hallucination_rate": float,
    "mrr": float,
    "ndcg": float,
    "hit_rate": float
  },
  "latency_ms": float,
  "cost_usd": float,
  "created_at": datetime,
  "dataset_id": UUID
}
```

### EvaluationDataset
```python
{
  "id": UUID,
  "name": str,
  "source_type": str,  # "pdf", "confluence", "web", "docs"
  "source_path": str,
  "question_count": int,
  "questions": [
    {
      "question": str,
      "answer": str,
      "source": str,
      "context": List[str]
    }
  ],
  "created_at": datetime,
  "updated_at": datetime
}
```

### EvaluationResult
```python
{
  "id": UUID,
  "experiment_id": UUID,
  "dataset_id": UUID,
  "question": str,
  "retrieved_contexts": List[str],
  "generated_answer": str,
  "ground_truth_answer": str,
  "metrics": {
    "faithfulness": float,
    "answer_relevancy": float,
    "context_utilization": float,
    "hallucination_score": float
  },
  "judge_score": float,
  "created_at": datetime
}
```

## API Endpoints

### Datasets
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/datasets/upload` | Upload document, auto-generate Q&A |
| GET | `/api/datasets` | List all datasets |
| GET | `/api/datasets/{id}` | Get dataset with questions |
| DELETE | `/api/datasets/{id}` | Delete dataset |
| POST | `/api/datasets/{id}/generate` | Regenerate Q&A pairs |

### Evaluation
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/evaluate/retrieval` | Run retrieval metrics on dataset |
| POST | `/api/evaluate/generation` | Run generation metrics |
| POST | `/api/evaluate/hallucination` | Detect hallucinations in answer |
| POST | `/api/evaluate/full` | Run full evaluation pipeline |

### Benchmark
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/benchmark/run` | Run full benchmark comparison |
| GET | `/api/benchmark/configs` | Get available benchmark configs |
| POST | `/api/benchmark/compare` | Compare two experiments |

### Experiments
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/experiments` | List experiment history |
| GET | `/api/experiments/{id}` | Get experiment details |
| POST | `/api/experiments` | Create/save experiment |
| DELETE | `/api/experiments/{id}` | Delete experiment |
| GET | `/api/experiments/{id}/history` | Get metric history over time |

### Judge
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/judge/evaluate` | LLM-as-a-Judge scoring |
| GET | `/api/judge/models` | List available judge models |

## Technologies

### Backend
- **Python 3.11+**
- **FastAPI** - Web framework
- **SQLAlchemy** - ORM
- **SQLite** - Database (demo), PostgreSQL (production)
- **Pydantic** - Data validation

### Evaluation
- **Ragas** - Primary evaluation framework (context precision/recall, faithfulness)
- **DeepEval** - Secondary evaluation (hallucination, answer relevancy)
- **OpenAI Evals** - Additional evaluation metrics

### Vector Databases
- **Qdrant** - Cloud/production vector DB
- **Chroma** - Local embedding storage
- **FAISS** - Facebook AI Similarity Search

### LLMs
- **OpenAI** - GPT-4o, GPT-4-turbo
- **Anthropic** - Claude 3 Opus, Haiku
- **Ollama** - Local Llama 3, Mistral

### Observability
- **OpenTelemetry** - Tracing and metrics
- **LangSmith** - LLM observability (optional)

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool
- **Recharts** - Data visualization
- **TypeScript** - Type safety

## Metrics Definitions

### Retrieval Metrics
| Metric | Description | Range |
|--------|-------------|-------|
| Context Recall | % of relevant chunks retrieved | 0-1 |
| Context Precision | Quality of retrieved chunks | 0-1 |
| MRR | Mean Reciprocal Rank | 0-1 |
| NDCG@k | Normalized DCG at k | 0-1 |
| Hit Rate@k | % of queries with hit in top-k | 0-1 |

### Generation Metrics
| Metric | Description | Range |
|--------|-------------|-------|
| Faithfulness | Answer consistency with context | 0-1 |
| Answer Relevancy | Answer addresses question | 0-1 |
| Context Utilization | How well context is used | 0-1 |
| Hallucination Rate | Factual error rate | 0-1 |

## Project Structure

```
rag-eval-platform/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI entry
│   │   ├── config.py              # Settings
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py         # Dataset CRUD + generation
│   │   │   ├── evaluate.py         # Retrieval + generation eval
│   │   │   ├── benchmark.py        # Multi-model comparison
│   │   │   ├── hallucination.py    # Hallucination detection
│   │   │   ├── experiments.py       # Experiment tracking
│   │   │   └── judge.py            # LLM-as-a-Judge
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── evaluator.py        # Ragas + DeepEval wrapper
│   │   │   ├── retriever.py        # Vector DB abstraction
│   │   │   ├── llm.py              # LLM adapter layer
│   │   │   └── extractors.py       # Document parsing
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── schemas.py          # Pydantic models
│   │   │   └── database.py         # SQLite + SQLAlchemy
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── dataset_generator.py
│   │       ├── retrieval_eval.py
│   │       ├── generation_eval.py
│   │       ├── hallucination.py
│   │       └── experiment_tracker.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── api/
│   │   └── types/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── Dockerfile
├── docker-compose.yml
├── SPEC.md
└── README.md
```

## Configuration

### Environment Variables
```
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-...

# Vector DBs
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=...

# Chroma (embedded, no config)

# FAISS (local, no config)

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Database
DATABASE_URL=sqlite:///./rag_eval.db

# OpenTelemetry
OTEL_SERVICE_NAME=rag-eval-platform
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## Deployment

### Development
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

### Docker Compose
```bash
docker-compose up --build
```

## Evaluation Workflow

1. **Upload Document** → Parse and chunk document
2. **Generate Q&A** → Use LLM to create question-answer pairs
3. **Run Retrieval** → Store embeddings, query, measure metrics
4. **Run Generation** → Generate answers, evaluate quality
5. **Run Hallucination Detection** → Extract claims, verify against sources
6. **Save Experiment** → Store all config and metrics
7. **Compare** → Benchmark different configurations

## Benchmark Configurations

### Embedding Models
| Model | Dimensions | Provider |
|-------|------------|----------|
| text-embedding-3-small | 1536 | OpenAI |
| text-embedding-3-large | 3072 | OpenAI |
| BGE-large | 1024 | HuggingFace |
| E5-large | 1024 | HuggingFace |

### Vector DBs
| DB | Type | Best For |
|----|------|----------|
| Qdrant | HNSW | Production, high accuracy |
| Chroma | IVF, HNSW | Local development |
| FAISS | IVF, HNSW | Large-scale, memory-mapped |

### LLMs
| Model | Provider | Use Case |
|-------|----------|----------|
| GPT-4o | OpenAI | General purpose, high quality |
| GPT-4-turbo | OpenAI | Fast, cost-effective |
| Claude 3 Opus | Anthropic | Complex reasoning |
| Claude 3 Haiku | Anthropic | Fast, cheap |
| Llama 3 70B | Ollama | Local, open source |
| Mistral 7B | Ollama | Local, fast |

## Hallucination Detection Pipeline

```
Answer Text
    │
    ▼
┌─────────────────┐
│ Claim Extraction│ (NER, factual extraction)
└─────────────────┘
    │
    ▼
┌─────────────────────┐
│ Source Verification │ (Check against retrieved context)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Hallucination Score │ (1 - verified_claims / total_claims)
└─────────────────────┘
```

## Dashboard Screens

### 1. Overview Dashboard
- Total experiments
- Average metrics across all runs
- Recent activity feed
- Quick benchmark access

### 2. Experiment List
- Filterable table of all experiments
- Sort by any metric
- Compare button to select multiple

### 3. Benchmark Runner
- Select datasets
- Select configurations to compare
- Run button with progress
- Results visualization

### 4. Dataset Manager
- Upload interface
- Dataset preview
- Q&A pair editor
- Regenerate option

### 5. Metrics Deep Dive
- Per-experiment detail view
- Metric explanations
- Improvement suggestions
