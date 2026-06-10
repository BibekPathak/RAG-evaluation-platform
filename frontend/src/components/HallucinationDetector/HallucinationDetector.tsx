import { useState } from 'react';
import { hallucinationApi } from '../../api/client';
import { AlertTriangle, CheckCircle, XCircle, Loader2, Sparkles } from 'lucide-react';
import type { HallucinationResult } from '../../types';

export default function HallucinationDetector() {
  const [answer, setAnswer] = useState('');
  const [context, setContext] = useState('');
  const [results, setResults] = useState<HallucinationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [useLlm, setUseLlm] = useState(false);

  const detectHallucination = async () => {
    if (!answer.trim() || !context.trim()) {
      alert('Please provide both an answer and context');
      return;
    }

    setLoading(true);
    try {
      const contexts = context.split('\n').filter((c) => c.trim());
      const result = useLlm
        ? await hallucinationApi.detectLlm(answer, contexts)
        : await hallucinationApi.detect(answer, contexts);
      setResults(result);
    } catch (error) {
      console.error('Detection failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Hallucination Detection</h1>
        <p className="text-slate-400 mt-1">Detect fabricated or unsupported claims in AI-generated answers</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Input</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Generated Answer
              </label>
              <textarea
                className="input w-full h-32 resize-none"
                placeholder="Enter the AI-generated answer to evaluate..."
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Context (one paragraph per line)
              </label>
              <textarea
                className="input w-full h-40 resize-none"
                placeholder="Enter the source context&#10;Each line will be treated as a separate context block"
                value={context}
                onChange={(e) => setContext(e.target.value)}
              />
            </div>

            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useLlm}
                  onChange={(e) => setUseLlm(e.target.checked)}
                  className="rounded border-slate-500"
                />
                <span className="text-slate-300 text-sm flex items-center gap-1">
                  <Sparkles className="w-4 h-4" />
                  Use LLM Detection
                </span>
              </label>
            </div>

            <button
              onClick={detectHallucination}
              disabled={loading || !answer.trim() || !context.trim()}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <AlertTriangle className="w-4 h-4" />
                  Detect Hallucinations
                </>
              )}
            </button>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Results</h3>
          
          {results ? (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-400 text-sm">Hallucination Score</p>
                  <p className={`text-3xl font-bold ${
                    results.hallucination_score < 0.2
                      ? 'text-green-400'
                      : results.hallucination_score < 0.5
                      ? 'text-yellow-400'
                      : 'text-red-400'
                  }`}>
                    {(results.hallucination_score * 100).toFixed(1)}%
                  </p>
                </div>
                <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                  results.hallucination_score < 0.2
                    ? 'bg-green-500/20'
                    : results.hallucination_score < 0.5
                    ? 'bg-yellow-500/20'
                    : 'bg-red-500/20'
                }`}>
                  {results.hallucination_score < 0.2 ? (
                    <CheckCircle className="w-8 h-8 text-green-400" />
                  ) : results.hallucination_score < 0.5 ? (
                    <AlertTriangle className="w-8 h-8 text-yellow-400" />
                  ) : (
                    <XCircle className="w-8 h-8 text-red-400" />
                  )}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="bg-dark-200 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-white">{results.total_claims}</p>
                  <p className="text-xs text-slate-400">Total Claims</p>
                </div>
                <div className="bg-dark-200 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-green-400">{results.verified_count}</p>
                  <p className="text-xs text-slate-400">Verified</p>
                </div>
                <div className="bg-dark-200 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-red-400">{results.hallucinated_count}</p>
                  <p className="text-xs text-slate-400">Hallucinated</p>
                </div>
              </div>

              {results.hallucinated_claims.length > 0 && (
                <div>
                  <p className="text-sm font-medium text-red-400 mb-2">Hallucinated Claims:</p>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {results.hallucinated_claims.map((claim, i) => (
                      <div key={i} className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                        <p className="text-red-300 text-sm">{claim}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {results.verified_claims.length > 0 && (
                <div>
                  <p className="text-sm font-medium text-green-400 mb-2">Verified Claims:</p>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {results.verified_claims.map((claim, i) => (
                      <div key={i} className="bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                        <p className="text-green-300 text-sm">{claim}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <AlertTriangle className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400">Enter an answer and context to detect hallucinations</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
