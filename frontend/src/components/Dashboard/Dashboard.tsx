import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, AlertTriangle, Clock, Activity, ChevronRight } from 'lucide-react';
import { dashboardApi } from '../../api/client';
import type { DashboardStats } from '../../types';

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const data = await dashboardApi.stats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load dashboard stats:', error);
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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">RAG Evaluation Dashboard</h1>
        <p className="text-slate-400 mt-1">Monitor and analyze your RAG system performance</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Experiments"
          value={stats?.total_experiments || 0}
          icon={<Activity className="w-5 h-5" />}
          trend="+12%"
          color="blue"
        />
        <StatCard
          title="Avg Faithfulness"
          value={`${((stats?.avg_faithfulness || 0) * 100).toFixed(1)}%`}
          icon={<TrendingUp className="w-5 h-5" />}
          trend="+5%"
          color="green"
        />
        <StatCard
          title="Avg Hallucination"
          value={`${((stats?.avg_hallucination || 0) * 100).toFixed(1)}%`}
          icon={<AlertTriangle className="w-5 h-5" />}
          trend="-8%"
          color="red"
        />
        <StatCard
          title="Avg Latency"
          value={`${((stats?.avg_latency_ms || 0)).toFixed(0)}ms`}
          icon={<Clock className="w-5 h-5" />}
          trend="-15%"
          color="purple"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Faithfulness Over Time</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={formatTimeSeriesData(stats?.recent_experiments || [], 'faithfulness')}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Top Models by Faithfulness</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={formatBarData(stats?.recent_experiments || [])} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#94a3b8" fontSize={12} domain={[0, 1]} />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={10} width={100} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Bar dataKey="value" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Experiments</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Name</th>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Embedding</th>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">LLM</th>
                <th className="text-left py-3 px-4 text-slate-400 font-medium">Retriever</th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">Faithfulness</th>
                <th className="text-right py-3 px-4 text-slate-400 font-medium">Latency</th>
              </tr>
            </thead>
            <tbody>
              {(stats?.recent_experiments || []).slice(0, 5).map((exp) => (
                <tr key={exp.id} className="border-b border-slate-700/50 hover:bg-slate-800/50">
                  <td className="py-3 px-4 text-white">{exp.name}</td>
                  <td className="py-3 px-4 text-slate-300">{exp.embedding_model}</td>
                  <td className="py-3 px-4 text-slate-300">{exp.llm}</td>
                  <td className="py-3 px-4 text-slate-300">{exp.retriever}</td>
                  <td className="py-3 px-4 text-right">
                    <span className={`metric-badge ${getMetricColor(exp.metrics.faithfulness || 0)}`}>
                      {((exp.metrics.faithfulness || 0) * 100).toFixed(1)}%
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

function StatCard({ title, value, icon, trend, color }: {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend: string;
  color: 'blue' | 'green' | 'red' | 'purple';
}) {
  const colors = {
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    green: 'bg-green-500/10 text-green-400 border-green-500/20',
    red: 'bg-red-500/10 text-red-400 border-red-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <div className={`p-2 rounded-lg ${colors[color].split(' ')[0]}`}>
          <div className={colors[color].split(' ')[1]}>{icon}</div>
        </div>
        <span className="text-xs text-green-400">{trend}</span>
      </div>
      <div className="mt-4">
        <p className="text-2xl font-bold text-white">{value}</p>
        <p className="text-sm text-slate-400 mt-1">{title}</p>
      </div>
    </div>
  );
}

function formatTimeSeriesData(experiments: any[], metric: string) {
  return experiments.slice(0, 20).reverse().map((exp, i) => ({
    name: `Run ${i + 1}`,
    value: exp.metrics[metric] || 0,
  }));
}

function formatBarData(experiments: any[]) {
  const grouped: Record<string, { total: number; count: number }> = {};
  
  experiments.forEach((exp) => {
    const key = exp.llm;
    if (!grouped[key]) {
      grouped[key] = { total: 0, count: 0 };
    }
    grouped[key].total += exp.metrics.faithfulness || 0;
    grouped[key].count += 1;
  });

  return Object.entries(grouped)
    .map(([name, data]) => ({
      name,
      value: data.total / data.count,
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 5);
}

function getMetricColor(value: number): string {
  if (value >= 0.8) return 'bg-green-500/20 text-green-400';
  if (value >= 0.6) return 'bg-yellow-500/20 text-yellow-400';
  return 'bg-red-500/20 text-red-400';
}
