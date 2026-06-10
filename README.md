# Enterprise RAG Evaluation Platform

A sophisticated RAG (Retrieval Augmented Generation) evaluation platform that answers the critical question: **"How do we know our RAG system is actually good?"**

## Why This Project Stands Out

Most RAG projects just build:
```
PDF → Vector DB → RAG → LLM
```

This platform evaluates and benchmarks the entire pipeline, demonstrating:
- AI evaluation expertise
- LLMOps knowledge
- Benchmarking methodology
- Observability patterns
- Experiment tracking
- Hallucination detection
- Production AI quality systems

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Dashboard                          │
│  (Experiment Tracking | Metrics | Benchmark Comparison)        │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ FastAPI Backend                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐│
│  │ Dataset API  │ │ Evaluate API│ │ Benchmark Runner API    ││
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
│   (Primary)   │    │  (Secondary) │    │    Engine     │
└───────────────┘    └───────────────┘    └───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         Vector DB Adapters (Chroma | Qdrant | FAISS)            │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### 1. Dataset Generator
Upload PDF, Confluence, Web, or Docs → Auto-generate Q&A pairs for evaluation.

### 2. Retrieval Evaluation
- Context Recall
- Context Precision
- MRR (Mean Reciprocal Rank)
- NDCG
- Hit Rate

### 3. Generation Evaluation
- Faithfulness
- Answer Relevancy
- Context Utilization
- Hallucination Rate

### 4. LLM-as-a-Judge
Use an LLM to score answers against ground truth with structured reasoning.

### 5. Benchmark Runner
Compare across dimensions:
- **Embeddings**: OpenAI text-embedding-3-small, BGE-large, E5
- **Vector DBs**: Chroma, Qdrant, FAISS
- **LLMs**: GPT-4o, Claude 3, Llama 3

### 6. Hallucination Detection
Standout feature: Extract claims → Verify against context → Score hallucination rate.

### 7. Experiment Tracking
Store and compare experiment configurations and results like MLflow for RAG.

### 8. Dashboard
React + Recharts dashboard with:
- Runs overview
- Top models comparison
- Faithfulness trends
- Latency metrics

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- OpenAI API key
- (Optional) Qdrant for vector storage

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Docker Compose

```bash
docker-compose up --build
```

## Environment Variables

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=...
DATABASE_URL=sqlite:///./rag_eval.db
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/datasets/upload` | Upload document, auto-generate Q&A |
| GET | `/api/datasets` | List all datasets |
| POST | `/api/evaluate/retrieval` | Run retrieval metrics |
| POST | `/api/evaluate/generation` | Run generation metrics |
| POST | `/api/evaluate/full` | Run full evaluation pipeline |
| POST | `/api/benchmark/run` | Full benchmark comparison |
| GET | `/api/experiments` | List experiment history |
| POST | `/api/hallucination/detect` | Detect hallucinations |
| POST | `/api/judge/evaluate` | LLM-as-a-Judge scoring |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, FastAPI |
| Evaluation | Ragas, DeepEval |
| Vector DBs | Qdrant, Chroma, FAISS |
| LLMs | OpenAI GPT-4o, Claude 3, Ollama |
| Frontend | React 18, TypeScript, Vite |
| Dashboard | Recharts |
| Observability | OpenTelemetry |

## Hallucination Detection Pipeline

```
Answer Text
    │
    ▼
┌─────────────────┐
│ Claim Extraction│
└─────────────────┘
    │
    ▼
┌─────────────────────┐
│ Source Verification │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Hallucination Score │
└─────────────────────┘
```

## Resume Value

| Typical RAG Chatbot | This Platform |
|--------------------|--------------------|
| 6/10 | 9.5/10 |

Demonstrates advanced skills in:
- AI evaluation systems
- LLMOps and MLOps
- Benchmarking methodology
- Production observability
- Experiment tracking
- Hallucination detection
- Full-stack development

## Project Structure

```
rag-eval-platform/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── core/          # Evaluator, retriever, LLM adapters
│   │   ├── models/       # SQLAlchemy + Pydantic models
│   │   └── services/    # Business logic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── api/         # API client
│   │   └── types/       # TypeScript types
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── SPEC.md
└── README.md
```

## License

MIT
