import { useState, useEffect, useRef } from 'react';
import { datasetsApi } from '../../api/client';
import { Upload, Loader2, Trash2, FileText, RefreshCw } from 'lucide-react';
import type { Dataset } from '../../types';

export default function DatasetManager() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
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
                        <div>
                          <p className="text-xs text-slate-400 mb-1">Question</p>
                          <p className="text-white text-sm">{qa.question}</p>
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
    </div>
  );
}
