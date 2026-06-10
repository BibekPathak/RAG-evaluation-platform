import { useState, useEffect } from 'react';
import { benchmarkApi, datasetsApi } from '../../api/client';
import { Play, Loader2 } from 'lucide-react';
import type { BenchmarkResponse, Dataset, BenchmarkConfig } from '../../types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function BenchmarkRunner() {
  const [configs, setConfigs] = useState<{
    embedding_models: Array<{ id: string; name: string }>;
    vector_dbs: Array<{ id: string; name: string }>;
    llms: Array<{ id: string; name: string }>;
  } | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<BenchmarkResponse | null>(null);

  const [selectedEmbeddings, setSelectedEmbeddings] = useState<string[]>(['text-embedding-3-small']);
  const [selectedVectorDbs, setSelectedVectorDbs] = useState<string[]>(['chroma']);
  const [selectedLlms, setSelectedLlms] = useState<string[]>(['gpt-4o']);
  const [selectedDataset, setSelectedDataset] = useState<string>('');

  useEffect(() => {
    loadConfigs();
  }, []);

  const loadConfigs = async () => {
    try {
      const [configsData, datasetsData] = await Promise.all([
        benchmarkApi.configs(),
        datasetsApi.list(),
      ]);
      setConfigs(configsData);
      setDatasets(datasetsData);
      if (datasetsData.length > 0) {
        setSelectedDataset(datasetsData[0].id);
      }
    } catch (error) {
      console.error('Failed to load configs:', error);
    } finally {
      setLoading(false);
    }
  };

  const runBenchmark = async () => {
    if (!selectedDataset) {
      alert('Please select a dataset');
      return;
    }

    setRunning(true);
    try {
      const config: BenchmarkConfig = {
        embedding_models: selectedEmbeddings,
        vector_dbs: selectedVectorDbs,
        llms: selectedLlms,
        dataset_id: selectedDataset,
        sample_size: 10,
      };
      const results = await benchmarkApi.run(config);
      setResults(results);
    } catch (error) {
      console.error('Benchmark failed:', error);
    } finally {
      setRunning(false);
    }
  };

  const toggleSelection = (item: string, selected: string[], setSelected: (v: string[]) => void) => {
    if (selected.includes(item)) {
      setSelected(selected.filter((v) => v !== item));
    } else {
      setSelected([...selected, item]);
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
        <h1 className="text-2xl font-bold text-white">Benchmark Runner</h1>
        <p className="text-slate-400 mt-1">Compare different embedding, retrieval, and LLM configurations</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Configuration</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Dataset</label>
              <select
                className="input w-full"
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
              >
                {datasets.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.name} ({ds.question_count} Q&A)
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Embedding Models</label>
              <div className="space-y-2">
                {configs?.embedding_models.map((model) => (
                  <label key={model.id} className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedEmbeddings.includes(model.id)}
                      onChange={() => toggleSelection(model.id, selectedEmbeddings, setSelectedEmbeddings)}
                      className="rounded border-slate-500"
                    />
                    <span className="text-slate-300 text-sm">{model.name}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Vector DBs</label>
              <div className="space-y-2">
                {configs?.vector_dbs.map((db) => (
                  <label key={db.id} className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedVectorDbs.includes(db.id)}
                      onChange={() => toggleSelection(db.id, selectedVectorDbs, setSelectedVectorDbs)}
                      className="rounded border-slate-500"
                    />
                    <span className="text-slate-300 text-sm">{db.name}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">LLMs</label>
              <div className="space-y-2">
                {configs?.llms.map((llm) => (
                  <label key={llm.id} className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedLlms.includes(llm.id)}
                      onChange={() => toggleSelection(llm.id, selectedLlms, setSelectedLlms)}
                      className="rounded border-slate-500"
                    />
                    <span className="text-slate-300 text-sm">{llm.name}</span>
                  </label>
                ))}
              </div>
            </div>

            <button
              onClick={runBenchmark}
              disabled={running || !selectedDataset}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {running ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Benchmark
                </>
              )}
            </button>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          {results && (
            <>
              <div className="card">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Results: {results.name}</h3>
                  <span className="text-slate-400 text-sm">{results.total_runs} runs completed</span>
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-dark-200 rounded-lg p-4">
                    <p className="text-slate-400 text-sm">Best Overall</p>
                    <p className="text-green-400 font-medium">
                      {results.best_overall ? 'Configured' : 'N/A'}
                    </p>
                  </div>
                  <div className="bg-dark-200 rounded-lg p-4">
                    <p className="text-slate-400 text-sm">Best Retrieval</p>
                    <p className="text-blue-400 font-medium">
                      {results.best_retrieval ? 'Configured' : 'N/A'}
                    </p>
                  </div>
                  <div className="bg-dark-200 rounded-lg p-4">
                    <p className="text-slate-400 text-sm">Best Generation</p>
                    <p className="text-purple-400 font-medium">
                      {results.best_generation ? 'Configured' : 'N/A'}
                    </p>
                  </div>
                </div>
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold text-white mb-4">Faithfulness by Configuration</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={formatBenchmarkResults(results.results)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="name" stroke="#94a3b8" fontSize={10} />
                      <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 1]} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                        labelStyle={{ color: '#fff' }}
                      />
                      <Legend />
                      <Bar dataKey="faithfulness" name="Faithfulness" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold text-white mb-4">All Results</h3>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="text-left py-3 px-4 text-slate-400 font-medium">Embedding</th>
                        <th className="text-left py-3 px-4 text-slate-400 font-medium">Vector DB</th>
                        <th className="text-left py-3 px-4 text-slate-400 font-medium">LLM</th>
                        <th className="text-center py-3 px-4 text-slate-400 font-medium">Faithful</th>
                        <th className="text-center py-3 px-4 text-slate-400 font-medium">Recall</th>
                        <th className="text-right py-3 px-4 text-slate-400 font-medium">Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.results.map((r, i) => (
                        <tr key={i} className="border-b border-slate-700/50">
                          <td className="py-3 px-4 text-slate-300 text-sm">{r.embedding_model}</td>
                          <td className="py-3 px-4 text-slate-300 text-sm">{r.vector_db}</td>
                          <td className="py-3 px-4 text-slate-300 text-sm">{r.llm}</td>
                          <td className="py-3 px-4 text-center">
                            <span className={`metric-badge ${r.metrics.faithfulness ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                              {((r.metrics.faithfulness || 0) * 100).toFixed(0)}%
                            </span>
                          </td>
                          <td className="py-3 px-4 text-center">
                            <span className="text-slate-300">
                              {((r.metrics.context_recall || 0) * 100).toFixed(0)}%
                            </span>
                          </td>
                          <td className="py-3 px-4 text-right text-slate-300">
                            {r.latency_ms.toFixed(0)}ms
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}

          {!results && (
            <div className="card flex items-center justify-center h-64">
              <div className="text-center">
                <Play className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400">Configure and run a benchmark to see results</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function formatBenchmarkResults(results: any[]) {
  return results.map((r) => ({
    name: `${r.embedding_model?.substring(0, 8)}/${r.vector_db?.substring(0, 5)}/${r.llm?.substring(0, 6)}`,
    faithfulness: r.metrics?.faithfulness || 0,
    relevancy: r.metrics?.answer_relevancy || 0,
    recall: r.metrics?.context_recall || 0,
  }));
}
