import { useState, useEffect } from 'react';
import { driftApi } from '../../api/client';
import { TrendingDown, AlertTriangle, CheckCircle, Clock, RefreshCw } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface DriftAlert {
  id: string;
  dataset_id: string;
  dataset_name: string;
  metric: string;
  old_value: number;
  new_value: number;
  change_pct: number;
  psi: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'active' | 'acknowledged' | 'resolved';
  created_at: string;
}

interface DriftVersion {
  id: string;
  dataset_id: string;
  version_number: number;
  version_hash: string;
  question_count: number;
  drift_score: number | null;
  created_at: string;
}

interface DatasetDriftSummary {
  dataset_id: string;
  dataset_name: string;
  current_version: number;
  total_versions: number;
  current_drift_score: number | null;
  drift_detected: boolean;
  active_alerts: number;
  acknowledged_alerts: number;
  resolved_alerts: number;
  recent_versions: DriftVersion[];
}

interface MetricChange {
  old: number;
  new: number;
  change_pct: number;
  psi: number;
}

const SEVERITY_COLORS = {
  low: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
  medium: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
  high: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
  critical: 'text-red-400 bg-red-500/10 border-red-500/30',
};

export default function DriftDashboard() {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<{
    total_datasets: number;
    datasets_with_drift: number;
    avg_drift_score: number;
    critical_alerts: number;
    warning_alerts: number;
    total_active_alerts: number;
    recent_alerts: DriftAlert[];
    drift_history: Array<{ dataset_id: string; dataset_name: string; version_number: number; drift_score: number; timestamp: string }>;
  } | null>(null);
  const [datasets, setDatasets] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [driftSummary, setDriftSummary] = useState<DatasetDriftSummary | null>(null);
  const [versionComparison, setVersionComparison] = useState<{
    version_a: DriftVersion;
    version_b: DriftVersion;
    metrics_a: Record<string, number>;
    metrics_b: Record<string, number>;
    metric_changes: Record<string, MetricChange>;
  } | null>(null);
  const [compareVersions, setCompareVersions] = useState<[number, number] | null>(null);

  useEffect(() => {
    loadStats();
    loadDatasets();
  }, []);

  const loadStats = async () => {
    try {
      const data = await driftApi.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load drift stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadDatasets = async () => {
    try {
      const response = await fetch('/api/datasets');
      const data = await response.json();
      setDatasets(data.map((d: { id: string; name: string }) => ({ id: d.id, name: d.name })));
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const loadDriftSummary = async (datasetId: string) => {
    try {
      const data = await driftApi.getDriftSummary(datasetId);
      setDriftSummary(data);
    } catch (error) {
      console.error('Failed to load drift summary:', error);
    }
  };

  const loadVersionComparison = async (datasetId: string, v1: number, v2: number) => {
    try {
      const data = await driftApi.compareVersions(datasetId, v1, v2);
      setVersionComparison(data);
      setCompareVersions([v1, v2]);
    } catch (error) {
      console.error('Failed to load version comparison:', error);
    }
  };

  const handleAcknowledge = async (alertId: string) => {
    try {
      await driftApi.acknowledgeAlert(alertId);
      loadStats();
      if (selectedDataset) {
        loadDriftSummary(selectedDataset);
      }
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const handleAcknowledgeAll = async () => {
    try {
      await driftApi.acknowledgeAllAlerts(selectedDataset || undefined);
      loadStats();
      if (selectedDataset) {
        loadDriftSummary(selectedDataset);
      }
    } catch (error) {
      console.error('Failed to acknowledge all alerts:', error);
    }
  };

  const handleDatasetSelect = (datasetId: string) => {
    setSelectedDataset(datasetId);
    loadDriftSummary(datasetId);
    setVersionComparison(null);
  };

  const formatChange = (change: number) => {
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(1)}%`;
  };

  const formatPsi = (psi: number) => {
    if (psi < 0.1) return { value: psi.toFixed(3), level: 'low', color: 'text-green-400' };
    if (psi < 0.2) return { value: psi.toFixed(3), level: 'moderate', color: 'text-yellow-400' };
    return { value: psi.toFixed(3), level: 'high', color: 'text-red-400' };
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Drift Detection</h1>
          <p className="text-slate-400 mt-1">Monitor dataset and metric drift over time</p>
        </div>
        <button onClick={loadStats} className="btn-secondary flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <CheckCircle className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stats?.total_datasets || 0}</p>
              <p className="text-xs text-slate-400">Total Datasets</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-orange-500/10 rounded-lg">
              <TrendingDown className="w-5 h-5 text-orange-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stats?.datasets_with_drift || 0}</p>
              <p className="text-xs text-slate-400">With Drift</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/10 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stats?.avg_drift_score?.toFixed(3) || '0.000'}</p>
              <p className="text-xs text-slate-400">Avg PSI</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-red-500/10 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-red-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stats?.critical_alerts || 0}</p>
              <p className="text-xs text-slate-400">Critical Alerts</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-yellow-500/10 rounded-lg">
              <Clock className="w-5 h-5 text-yellow-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stats?.total_active_alerts || 0}</p>
              <p className="text-xs text-slate-400">Active Alerts</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="card">
            <h3 className="text-lg font-semibold text-white mb-4">Drift Timeline</h3>
            <div className="h-64">
              {stats?.drift_history && stats.drift_history.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stats.drift_history.slice(0, 20).reverse()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis
                      dataKey="timestamp"
                      tick={{ fill: '#94a3b8', fontSize: 10 }}
                      tickFormatter={(value) => new Date(value).toLocaleDateString()}
                    />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: '8px',
                      }}
                      labelStyle={{ color: '#e2e8f0' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="drift_score"
                      stroke="#f97316"
                      strokeWidth={2}
                      dot={{ fill: '#f97316', r: 4 }}
                      name="Drift Score (PSI)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-full">
                  <p className="text-slate-400">No drift history available yet</p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Active Alerts</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {stats?.recent_alerts && stats.recent_alerts.length > 0 ? (
              stats.recent_alerts.slice(0, 10).map((alert) => (
                <div
                  key={alert.id}
                  className={`p-3 rounded-lg border ${SEVERITY_COLORS[alert.severity]}`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-white">{alert.dataset_name}</p>
                      <p className="text-xs text-slate-400">{alert.metric}</p>
                    </div>
                    <button
                      onClick={() => handleAcknowledge(alert.id)}
                      className="text-xs px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded transition-colors"
                    >
                      Ack
                    </button>
                  </div>
                  <div className="mt-2 flex items-center gap-2 text-xs">
                    <span className={formatPsi(alert.psi).color}>
                      PSI: {formatPsi(alert.psi).value}
                    </span>
                    <span className={alert.change_pct < 0 ? 'text-red-400' : 'text-green-400'}>
                      {formatChange(alert.change_pct)}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8">
                <CheckCircle className="w-8 h-8 text-green-400 mx-auto mb-2" />
                <p className="text-slate-400 text-sm">No active alerts</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <div className="card">
            <h3 className="text-lg font-semibold text-white mb-4">Select Dataset</h3>
            <div className="space-y-2">
              {datasets.map((ds) => (
                <button
                  key={ds.id}
                  onClick={() => handleDatasetSelect(ds.id)}
                  className={`w-full p-3 rounded-lg border text-left transition-colors ${
                    selectedDataset === ds.id
                      ? 'border-primary-500 bg-primary-500/10'
                      : 'border-slate-700 hover:border-slate-600'
                  }`}
                >
                  <p className="text-white text-sm font-medium">{ds.name}</p>
                </button>
              ))}
              {datasets.length === 0 && (
                <p className="text-slate-400 text-sm text-center py-4">No datasets available</p>
              )}
            </div>
          </div>
        </div>

        <div className="lg:col-span-2">
          {driftSummary ? (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-white">{driftSummary.dataset_name}</h3>
                  <p className="text-sm text-slate-400">
                    Version {driftSummary.current_version} of {driftSummary.total_versions}
                    {driftSummary.current_drift_score !== null && (
                      <span className={`ml-2 ${formatPsi(driftSummary.current_drift_score).color}`}>
                        PSI: {formatPsi(driftSummary.current_drift_score).value}
                      </span>
                    )}
                  </p>
                </div>
                {driftSummary.active_alerts > 0 && (
                  <button
                    onClick={handleAcknowledgeAll}
                    className="text-sm px-3 py-1 bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20 rounded transition-colors"
                  >
                    Acknowledge All ({driftSummary.active_alerts})
                  </button>
                )}
              </div>

              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="bg-dark-200 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-red-400">{driftSummary.active_alerts}</p>
                  <p className="text-xs text-slate-400">Active</p>
                </div>
                <div className="bg-dark-200 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-yellow-400">{driftSummary.acknowledged_alerts}</p>
                  <p className="text-xs text-slate-400">Acknowledged</p>
                </div>
                <div className="bg-dark-200 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-green-400">{driftSummary.resolved_alerts}</p>
                  <p className="text-xs text-slate-400">Resolved</p>
                </div>
              </div>

              {driftSummary.recent_versions.length >= 2 && (
                <div className="mb-4">
                  <h4 className="text-sm font-medium text-slate-300 mb-2">Compare Versions</h4>
                  <div className="flex items-center gap-4">
                    <select
                      className="bg-dark-200 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
                      value={compareVersions?.[0] || ''}
                      onChange={(e) => {
                        const v1 = parseInt(e.target.value);
                        if (compareVersions) {
                          loadVersionComparison(driftSummary.dataset_id, v1, compareVersions[1]);
                        } else if (driftSummary.recent_versions.length >= 2) {
                          loadVersionComparison(driftSummary.dataset_id, v1, driftSummary.recent_versions[1].version_number);
                        }
                      }}
                    >
                      {driftSummary.recent_versions.map((v) => (
                        <option key={v.id} value={v.version_number}>
                          v{v.version_number}
                        </option>
                      ))}
                    </select>
                    <span className="text-slate-400">vs</span>
                    <select
                      className="bg-dark-200 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
                      value={compareVersions?.[1] || ''}
                      onChange={(e) => {
                        const v2 = parseInt(e.target.value);
                        if (compareVersions) {
                          loadVersionComparison(driftSummary.dataset_id, compareVersions[0], v2);
                        } else if (driftSummary.recent_versions.length >= 2) {
                          loadVersionComparison(driftSummary.dataset_id, driftSummary.recent_versions[0].version_number, v2);
                        }
                      }}
                    >
                      {driftSummary.recent_versions.map((v) => (
                        <option key={v.id} value={v.version_number}>
                          v{v.version_number}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )}

              {versionComparison && (
                <div className="bg-dark-200 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-slate-300 mb-3">
                    Version Comparison: v{compareVersions?.[0]} vs v{compareVersions?.[1]}
                  </h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-slate-400 mb-2">Metrics</p>
                      <div className="space-y-2">
                        {Object.entries(versionComparison.metric_changes).map(([metric, change]) => (
                          <div key={metric} className="flex items-center justify-between">
                            <span className="text-sm text-slate-300">{metric}</span>
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-slate-500">
                                {change.old.toFixed(3)} → {change.new.toFixed(3)}
                              </span>
                              <span className={`text-xs ${change.change_pct < 0 ? 'text-red-400' : 'text-green-400'}`}>
                                {formatChange(change.change_pct)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400 mb-2">PSI Scores</p>
                      <div className="space-y-2">
                        {Object.entries(versionComparison.metric_changes).map(([metric, change]) => (
                          <div key={metric} className="flex items-center justify-between">
                            <span className="text-sm text-slate-300">{metric}</span>
                            <span className={`text-xs ${formatPsi(change.psi).color}`}>
                              {formatPsi(change.psi).value}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="mt-4">
                <h4 className="text-sm font-medium text-slate-300 mb-2">Version History</h4>
                <div className="space-y-2">
                  {driftSummary.recent_versions.map((version) => (
                    <div
                      key={version.id}
                      className="flex items-center justify-between p-2 bg-dark-200 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-white text-sm font-medium">v{version.version_number}</span>
                        <span className="text-xs text-slate-400">
                          {version.question_count} questions
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        {version.drift_score !== null && (
                          <span className={`text-xs ${formatPsi(version.drift_score).color}`}>
                            PSI: {formatPsi(version.drift_score).value}
                          </span>
                        )}
                        <span className="text-xs text-slate-500">
                          {new Date(version.created_at).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="card flex items-center justify-center h-96">
              <div className="text-center">
                <TrendingDown className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400">Select a dataset to view drift details</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}