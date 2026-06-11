import { useState, useEffect, useRef } from 'react';
import { datasetsApi } from '../../api/client';
import { Upload, Loader2, Trash2, FileText, RefreshCw, Settings, ChevronDown, ChevronUp } from 'lucide-react';
import type { Dataset, GenerationPreset, QuestionAnswer } from '../../types';

const PRESETS: { id: GenerationPreset; name: string; description: string; distribution: Record<string, number> }[] = [
  {
    id: 'balanced',
    name: 'Balanced',
    description: '40% Easy, 30% Medium, 20% Hard, 10% Adversarial',
    distribution: { easy: 0.4, medium: 0.3, hard: 0.2, adversarial: 0.1 },
  },
  {
    id: 'hard_evaluation',
    name: 'Hard Evaluation',
    description: '20% Easy, 30% Medium, 30% Hard, 20% Adversarial',
    distribution: { easy: 0.2, medium: 0.3, hard: 0.3, adversarial: 0.2 },
  },
  {
    id: 'adversarial_heavy',
    name: 'Adversarial Heavy',
    description: '30% Easy, 20% Medium, 20% Hard, 30% Adversarial',
    distribution: { easy: 0.3, medium: 0.2, hard: 0.2, adversarial: 0.3 },
  },
  {
    id: 'retrieval_stress',
    name: 'Retrieval Stress',
    description: '20% Easy, 40% Medium, 30% Hard, 10% Adversarial',
    distribution: { easy: 0.2, medium: 0.4, hard: 0.3, adversarial: 0.1 },
  },
];

const DIFFICULTY_COLORS: Record<string, string> = {
  easy: 'bg-green-500/20 text-green-400 border-green-500/30',
  medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  hard: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  adversarial: 'bg-red-500/20 text-red-400 border-red-500/30',
};

export default function DatasetManager() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<GenerationPreset>('balanced');
  const [totalQuestions, setTotalQuestions] = useState(50);
  const [customDistribution, setCustomDistribution] = useState(false);
  const [distribution, setDistribution] = useState({
    easy: 20,
    medium: 15,
    hard: 10,
    adversarial: 5,
  });
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const data = await datasetsApi.list();
      setDatasets(data);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', file.name.replace(/\.[^/.]+$/, ''));
    formData.append('source_type', getSourceType(file.name));
    formData.append('num_questions', '10');

    setUploading(true);
    try {
      const dataset = await datasetsApi.upload(formData);
      setDatasets([dataset, ...datasets]);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleGenerateQuestions = async () => {
    if (!selectedDataset) return;

    setGenerating(true);
    try {
      const result = await datasetsApi.generateQuestions({
        document_id: selectedDataset.id,
        total_questions: totalQuestions,
        distribution: customDistribution ? distribution : undefined,
        preset: !customDistribution ? selectedPreset : undefined,
        verify_difficulty: true,
      });

      const updatedDataset: Dataset = {
        ...selectedDataset,
        questions: result.questions as QuestionAnswer[],
        question_count: result.total_generated,
      };

      setSelectedDataset(updatedDataset);
      setDatasets(datasets.map(d => d.id === updatedDataset.id ? updatedDataset : d));
      setShowGenerateModal(false);
    } catch (error) {
      console.error('Generate failed:', error);
    } finally {
      setGenerating(false);
    }
  };

  const handlePresetChange = (preset: GenerationPreset) => {
    setSelectedPreset(preset);
    const p = PRESETS.find(p => p.id === preset);
    if (p) {
      setDistribution({
        easy: Math.round(totalQuestions * p.distribution.easy),
        medium: Math.round(totalQuestions * p.distribution.medium),
        hard: Math.round(totalQuestions * p.distribution.hard),
        adversarial: totalQuestions - Math.round(totalQuestions * p.distribution.easy) - Math.round(totalQuestions * p.distribution.medium) - Math.round(totalQuestions * p.distribution.hard),
      });
    }
  };

  const handleTotalChange = (total: number) => {
    setTotalQuestions(total);
    const p = PRESETS.find(p => p.id === selectedPreset);
    if (p) {
      setDistribution({
        easy: Math.round(total * p.distribution.easy),
        medium: Math.round(total * p.distribution.medium),
        hard: Math.round(total * p.distribution.hard),
        adversarial: total - Math.round(total * p.distribution.easy) - Math.round(total * p.distribution.medium) - Math.round(total * p.distribution.hard),
      });
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this dataset?')) return;

    try {
      await datasetsApi.delete(id);
      setDatasets(datasets.filter((d) => d.id !== id));
      if (selectedDataset?.id === id) {
        setSelectedDataset(null);
      }
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const getSourceType = (filename: string): string => {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (ext === 'pdf') return 'pdf';
    if (ext === 'docx') return 'docx';
    if (ext === 'txt') return 'text';
    return 'text';
  };

  const getDifficultyBadge = (difficulty?: string) => {
    if (!difficulty) return null;
    const colorClass = DIFFICULTY_COLORS[difficulty] || DIFFICULTY_COLORS.medium;
    return (
      <span className={`text-xs px-2 py-0.5 rounded border ${colorClass}`}>
        {difficulty}
      </span>
    );
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
          <h1 className="text-2xl font-bold text-white">Dataset Manager</h1>
          <p className="text-slate-400 mt-1">Upload and manage evaluation datasets</p>
        </div>
        <div className="flex items-center gap-3">
          {selectedDataset && (
            <button
              onClick={() => setShowGenerateModal(true)}
              className="btn-secondary flex items-center gap-2"
            >
              <Settings className="w-4 h-4" />
              Generate Questions
            </button>
          )}
          <label className="btn-primary flex items-center gap-2 cursor-pointer">
            <Upload className="w-4 h-4" />
            Upload Dataset
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.txt,.docx"
              onChange={handleUpload}
            />
          </label>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-4">
          {uploading && (
            <div className="card flex items-center gap-3">
              <Loader2 className="w-5 h-5 animate-spin text-primary-500" />
              <span className="text-slate-300">Uploading dataset...</span>
            </div>
          )}

          <div className="card">
            <h3 className="text-lg font-semibold text-white mb-4">Datasets ({datasets.length})</h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {datasets.map((ds) => (
                <div
                  key={ds.id}
                  className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                    selectedDataset?.id === ds.id
                      ? 'border-primary-500 bg-primary-500/10'
                      : 'border-slate-700 hover:border-slate-600'
                  }`}
                  onClick={() => setSelectedDataset(ds)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-slate-400" />
                      <span className="text-white text-sm font-medium">{ds.name}</span>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(ds.id);
                      }}
                      className="p-1 hover:bg-slate-700 rounded"
                    >
                      <Trash2 className="w-4 h-4 text-slate-400 hover:text-red-400" />
                    </button>
                  </div>
                  <div className="mt-2 flex items-center gap-3 text-xs text-slate-400">
                    <span>{ds.source_type.toUpperCase()}</span>
                    <span>{ds.question_count} Q&A pairs</span>
                  </div>
                </div>
              ))}

              {datasets.length === 0 && (
                <div className="text-center py-8">
                  <Upload className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                  <p className="text-slate-400 text-sm">No datasets yet</p>
                  <p className="text-slate-500 text-xs">Upload a PDF, TXT, or DOCX file</p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="lg:col-span-2">
          {selectedDataset ? (
            <div className="card">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-lg font-semibold text-white">{selectedDataset.name}</h3>
                  <p className="text-slate-400 text-sm mt-1">
                    {selectedDataset.question_count} question-answer pairs
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400 bg-dark-200 px-2 py-1 rounded">
                    {selectedDataset.source_type.toUpperCase()}
                  </span>
                  <button
                    onClick={() => setShowGenerateModal(true)}
                    className="p-2 hover:bg-dark-200 rounded-lg transition-colors"
                    title="Regenerate Questions"
                  >
                    <RefreshCw className="w-4 h-4 text-slate-400 hover:text-white" />
                  </button>
                </div>
              </div>

              <div className="space-y-4 max-h-[500px] overflow-y-auto">
                {selectedDataset.questions.map((qa, i) => (
                  <div key={i} className="bg-dark-200 rounded-lg p-4 border border-slate-700/50">
                    <div className="flex items-start gap-3">
                      <span className="flex-shrink-0 w-6 h-6 bg-primary-500/20 text-primary-400 rounded-full flex items-center justify-center text-xs font-medium">
                        {i + 1}
                      </span>
                      <div className="flex-1 space-y-2">
                        <div className="flex items-center justify-between">
                          <p className="text-white text-sm">{qa.question}</p>
                          {getDifficultyBadge(qa.difficulty)}
                        </div>
                        <div>
                          <p className="text-xs text-slate-400 mb-1">Answer</p>
                          <p className="text-slate-300 text-sm">{qa.answer}</p>
                        </div>
                        {qa.context.length > 0 && (
                          <div>
                            <p className="text-xs text-slate-400 mb-1">Context</p>
                            <p className="text-slate-400 text-xs line-clamp-2">
                              {qa.context[0]?.substring(0, 150)}...
                            </p>
                          </div>
                        )}
                        {qa.traps && qa.traps.length > 0 && (
                          <div className="flex items-center gap-2 mt-2">
                            <span className="text-xs text-red-400">Traps:</span>
                            {qa.traps.map((trap, j) => (
                              <span key={j} className="text-xs px-2 py-0.5 bg-red-500/10 text-red-300 rounded">
                                {trap}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="card flex items-center justify-center h-96">
              <div className="text-center">
                <FileText className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400">Select a dataset to view its contents</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {showGenerateModal && selectedDataset && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-dark-100 rounded-xl border border-slate-700 p-6 w-full max-w-2xl">
            <h2 className="text-xl font-bold text-white mb-4">Generate Questions</h2>
            <p className="text-slate-400 mb-6">
              Generate synthetic questions for <span className="text-white">{selectedDataset.name}</span>
            </p>

            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-3">
                  Total Questions: {totalQuestions}
                </label>
                <input
                  type="range"
                  min="10"
                  max="200"
                  value={totalQuestions}
                  onChange={(e) => handleTotalChange(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>10</span>
                  <span>200</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-3">
                  Generation Preset
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {PRESETS.map((preset) => (
                    <button
                      key={preset.id}
                      onClick={() => handlePresetChange(preset.id)}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        selectedPreset === preset.id
                          ? 'border-primary-500 bg-primary-500/10'
                          : 'border-slate-700 hover:border-slate-600'
                      }`}
                    >
                      <div className="text-white text-sm font-medium">{preset.name}</div>
                      <div className="text-slate-400 text-xs mt-1">{preset.description}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <button
                  onClick={() => setAdvancedOpen(!advancedOpen)}
                  className="flex items-center gap-2 text-slate-300 hover:text-white transition-colors"
                >
                  {advancedOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  Advanced Settings
                </button>

                {advancedOpen && (
                  <div className="mt-4 space-y-4 p-4 bg-dark-200 rounded-lg border border-slate-700">
                    <div className="flex items-center gap-3 mb-4">
                      <input
                        type="checkbox"
                        id="customDist"
                        checked={customDistribution}
                        onChange={(e) => setCustomDistribution(e.target.checked)}
                        className="rounded border-slate-600"
                      />
                      <label htmlFor="customDist" className="text-sm text-slate-300">
                        Use custom distribution
                      </label>
                    </div>

                    {customDistribution && (
                      <div className="grid grid-cols-2 gap-4">
                        {(['easy', 'medium', 'hard', 'adversarial'] as const).map((diff) => (
                          <div key={diff}>
                            <label className="block text-xs text-slate-400 mb-1 capitalize">
                              {diff} ({distribution[diff]})
                            </label>
                            <input
                              type="range"
                              min="0"
                              max={totalQuestions}
                              value={distribution[diff]}
                              onChange={(e) => {
                                const val = parseInt(e.target.value);
                                setDistribution({ ...distribution, [diff]: val });
                              }}
                              className="w-full"
                            />
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="bg-dark-200 rounded-lg p-4 border border-slate-700">
                <h4 className="text-sm font-medium text-slate-300 mb-3">Distribution Preview</h4>
                <div className="grid grid-cols-4 gap-3">
                  {Object.entries(distribution).map(([difficulty, count]) => (
                    <div key={difficulty} className="text-center">
                      <div className={`text-2xl font-bold ${DIFFICULTY_COLORS[difficulty].split(' ')[1]}`}>
                        {count}
                      </div>
                      <div className="text-xs text-slate-400 capitalize">{difficulty}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowGenerateModal(false)}
                className="btn-secondary"
                disabled={generating}
              >
                Cancel
              </button>
              <button
                onClick={handleGenerateQuestions}
                className="btn-primary flex items-center gap-2"
                disabled={generating}
              >
                {generating && <Loader2 className="w-4 h-4 animate-spin" />}
                Generate
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
