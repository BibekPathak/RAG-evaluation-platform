import { useState, useEffect } from 'react';
import { experimentsApi } from '../../api/client';
import type { Experiment } from '../../types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function ExperimentList() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedExp, setSelectedExp] = useState<Experiment | null>(null);

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    try {
      const data = await experimentsApi.list({ limit: 100 });
      setExperiments(data);
    } catch (error) {
      console.error('Failed to load experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  const chartData = experiments.slice(0, 20).reverse().map((exp, i) => ({
    name: exp.name.substring(0, 15),
    faithfulness: (exp.metrics.faithfulness || 0) * 100,
    relevancy: (exp.metrics.answer_relevancy || 0) * 100,
    recall: (exp.metrics.context_recall || 0) * 100,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Experiment Tracking</h1>
        <p className="text-slate-400 mt-1">Track and compare your RAG evaluation experiments</p>
      </div>

      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Metrics Comparison</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={10} />
              <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Bar dataKey="faithfulness" name="Faithfulness %" fill="#3b82f6" />
              <Bar dataKey="relevancy" name="Relevancy %" fill="#10b981" />
              <Bar dataKey="recall" name="Recall %" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">All Experiments</h3>
          <span className="text-slate-400 text-sm">{experiments.length} experiments</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Experiment</th>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Embedding</th>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">LLM</th>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Retriever</th>
                <th className="text-center py-3 px-4 text-slate-400 font-medium">Recall</th>
                <th className="text-center py-3 px-4 text-slate-400 font-medium">Precision</th>
                <th className="text-center py-3 px-4 text-slate-400 font-medium">Faithful</th>
                <th className="text-center py-3 px-4 text-slate-400 font-medium">Relevance</th>
                <th className="text-center py-3 px-4 text-slate-400 font-medium">Halluc.</th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">Latency</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp) => (
                <tr
                  key={exp.id}
                  className="border-b border-slate-700/50 hover:bg-slate-800/50 cursor-pointer"
                  onClick={() => setSelectedExp(exp)}
                >
                  <td className="py-3 px-4 text-white">{exp.name}</td>
                  <td className="py-3 px-4 text-slate-300 text-sm">{exp.embedding_model}</td>
                  <td className="py-3 px-4 text-slate-300 text-sm">{exp.llm}</td>
                  <td className="py-3 px-4 text-slate-300 text-sm">{exp.retriever}</td>
                  <td className="py-3 px-4 text-center">
                    <span className={getMetricClass(exp.metrics.context_recall || 0)}>
                      {((exp.metrics.context_recall || 0) * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={getMetricClass(exp.metrics.context_precision || 0)}>
                      {((exp.metrics.context_precision || 0) * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={getMetricClass(exp.metrics.faithfulness || 0)}>
                      {((exp.metrics.faithfulness || 0) * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={getMetricClass(exp.metrics.answer_relevancy || 0)}>
                      {((exp.metrics.answer_relevancy || 0) * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={getHallucinationClass(exp.metrics.hallucination_rate || 0)}>
                      {((exp.metrics.hallucination_rate || 0) * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">
                    {exp.latency_ms?.toFixed(0)}ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function getMetricClass(value: number): string {
  if (value >= 0.8) return 'metric-badge bg-green-500/20 text-green-400';
  if (value >= 0.6) return 'metric-badge bg-yellow-500/20 text-yellow-400';
  return 'metric-badge bg-red-500/20 text-red-400';
}

function getHallucinationClass(value: number): string {
  if (value <= 0.1) return 'metric-badge bg-green-500/20 text-green-400';
  if (value <= 0.3) return 'metric-badge bg-yellow-500/20 text-yellow-400';
  return 'metric-badge bg-red-500/20 text-red-400';
}
