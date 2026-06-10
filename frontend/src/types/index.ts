export interface QuestionAnswer {
  question: string;
  answer: string;
  source: string;
  context: string[];
}

export interface Dataset {
  id: string;
  name: string;
  source_type: string;
  source_path?: string;
  question_count: number;
  questions: QuestionAnswer[];
  created_at: string;
  updated_at: string;
}

export interface MetricsSummary {
  context_recall?: number;
  context_precision?: number;
  faithfulness?: number;
  answer_relevancy?: number;
  hallucination_rate?: number;
  mrr?: number;
  ndcg?: number;
  hit_rate?: number;
}

export interface Experiment {
  id: string;
  name: string;
  description?: string;
  embedding_model: string;
  llm: string;
  retriever: string;
  metrics: MetricsSummary;
  latency_ms?: number;
  cost_usd?: number;
  created_at: string;
  updated_at: string;
  dataset_id?: string;
}

export interface RetrievalMetrics {
  context_recall: number;
  context_precision: number;
  mrr: number;
  ndcg: number;
  hit_rate: number;
}

export interface GenerationMetrics {
  faithfulness: number;
  answer_relevancy: number;
  context_utilization: number;
  hallucination_rate: number;
}

export interface HallucinationResult {
  claims: string[];
  verified_claims: string[];
  hallucinated_claims: string[];
  hallucination_score: number;
  total_claims: number;
  verified_count: number;
  hallucinated_count: number;
}

export interface JudgeResponse {
  score: number;
  reasoning: string;
  dimensions: {
    faithfulness: number;
    answer_relevancy: number;
  };
}

export interface BenchmarkConfig {
  embedding_models: string[];
  vector_dbs: string[];
  llms: string[];
  dataset_id: string;
  sample_size: number;
}

export interface BenchmarkResult {
  embedding_model: string;
  vector_db: string;
  llm: string;
  metrics: MetricsSummary;
  latency_ms: number;
  cost_usd: number;
  experiment_id: string;
}

export interface BenchmarkResponse {
  name: string;
  total_runs: number;
  results: BenchmarkResult[];
  best_overall?: string;
  best_retrieval?: string;
  best_generation?: string;
}

export interface DashboardStats {
  total_experiments: number;
  avg_faithfulness: number;
  avg_hallucination: number;
  avg_latency_ms: number;
  recent_experiments: Experiment[];
}
