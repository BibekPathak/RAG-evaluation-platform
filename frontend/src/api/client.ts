import axios from 'axios';
import type {
  Dataset,
  Experiment,
  RetrievalMetrics,
  GenerationMetrics,
  HallucinationResult,
  JudgeResponse,
  BenchmarkResponse,
  BenchmarkConfig,
  DashboardStats,
  GenerationPreset,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
});

export const datasetsApi = {
  upload: async (formData: FormData): Promise<Dataset> => {
    const response = await api.post('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data.dataset;
  },

  list: async (skip = 0, limit = 100): Promise<Dataset[]> => {
    const response = await api.get('/datasets', { params: { skip, limit } });
    return response.data;
  },

  get: async (id: string): Promise<Dataset> => {
    const response = await api.get(`/datasets/${id}`);
    return response.data;
  },

  delete: async (id: string): Promise<void> => {
    await api.delete(`/datasets/${id}`);
  },

  generateQuestions: async (params: {
    document_id: string;
    total_questions?: number;
    distribution?: Record<string, number>;
    preset?: GenerationPreset;
    verify_difficulty?: boolean;
  }): Promise<{
    questions: import('../types').QuestionAnswer[];
    distribution: Record<string, number>;
    total_generated: number;
    verified_difficulties: boolean;
  }> => {
    const response = await api.post('/datasets/generate-questions', params);
    return response.data;
  },
};

export const experimentsApi = {
  list: async (params?: {
    skip?: number;
    limit?: number;
    embedding_model?: string;
    llm?: string;
    retriever?: string;
  }): Promise<Experiment[]> => {
    const response = await api.get('/experiments', { params });
    return response.data;
  },

  get: async (id: string): Promise<Experiment> => {
    const response = await api.get(`/experiments/${id}`);
    return response.data;
  },

  create: async (data: {
    name: string;
    description?: string;
    embedding_model: string;
    llm: string;
    retriever: string;
    dataset_id?: string;
  }): Promise<Experiment> => {
    const response = await api.post('/experiments', data);
    return response.data;
  },

  delete: async (id: string): Promise<void> => {
    await api.delete(`/experiments/${id}`);
  },

  compare: async (experimentIds: string[]): Promise<{ experiments: Experiment[]; winning_metrics: Record<string, string> }> => {
    const response = await api.post('/experiments/compare', { experiment_ids: experimentIds });
    return response.data;
  },
};

export const evaluateApi = {
  retrieval: async (datasetId: string, topK = 5): Promise<RetrievalMetrics> => {
    const response = await api.post('/evaluate/retrieval', { dataset_id: datasetId, top_k: topK });
    return response.data;
  },

  generation: async (experimentId: string): Promise<GenerationMetrics> => {
    const response = await api.post('/evaluate/generation', { experiment_id: experimentId });
    return response.data;
  },

  full: async (params: {
    dataset_id: string;
    embedding_model?: string;
    llm_model?: string;
    retriever?: string;
  }): Promise<{
    experiment_id: string;
    retrieval_metrics: RetrievalMetrics;
    generation_metrics: GenerationMetrics;
    latency_ms: number;
  }> => {
    const response = await api.post('/evaluate/full', null, { params });
    return response.data;
  },
};

export const hallucinationApi = {
  detect: async (answer: string, context: string[]): Promise<HallucinationResult> => {
    const response = await api.post('/hallucination/detect', { answer, context });
    return response.data;
  },

  detectLlm: async (answer: string, context: string[], model = 'gpt-4o'): Promise<HallucinationResult> => {
    const response = await api.post('/hallucination/detect-llm', null, {
      params: { answer, context, model },
    });
    return response.data;
  },
};

export const judgeApi = {
  evaluate: async (data: {
    question: string;
    ground_truth_answer: string;
    generated_answer: string;
    context: string[];
    judge_model?: string;
  }): Promise<JudgeResponse> => {
    const response = await api.post('/judge/evaluate', data);
    return response.data;
  },

  models: async (): Promise<{ models: Array<{ id: string; name: string; provider: string; recommended: boolean }> }> => {
    const response = await api.get('/judge/models');
    return response.data;
  },
};

export const benchmarkApi = {
  run: async (config: BenchmarkConfig): Promise<BenchmarkResponse> => {
    const response = await api.post('/benchmark/run', config);
    return response.data;
  },

  configs: async (): Promise<{
    embedding_models: Array<{ id: string; name: string; dims: number }>;
    vector_dbs: Array<{ id: string; name: string; type: string }>;
    llms: Array<{ id: string; name: string; provider: string }>;
  }> => {
    const response = await api.get('/benchmark/configs');
    return response.data;
  },
};

export const dashboardApi = {
  stats: async (): Promise<DashboardStats> => {
    const experiments = await experimentsApi.list({ limit: 100 });
    
    const avgFaithfulness = experiments.length > 0
      ? experiments.reduce((sum, e) => sum + (e.metrics.faithfulness || 0), 0) / experiments.length
      : 0;
    
    const avgHallucination = experiments.length > 0
      ? experiments.reduce((sum, e) => sum + (e.metrics.hallucination_rate || 0), 0) / experiments.length
      : 0;
    
    const avgLatency = experiments.length > 0
      ? experiments.reduce((sum, e) => sum + (e.latency_ms || 0), 0) / experiments.length
      : 0;

    return {
      total_experiments: experiments.length,
      avg_faithfulness: avgFaithfulness,
      avg_hallucination: avgHallucination,
      avg_latency_ms: avgLatency,
      recent_experiments: experiments.slice(0, 10),
    };
  },
};
