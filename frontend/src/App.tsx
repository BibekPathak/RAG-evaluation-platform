import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard/Dashboard';
import ExperimentList from './components/ExperimentList/ExperimentList';
import BenchmarkRunner from './components/BenchmarkRunner/BenchmarkRunner';
import DatasetManager from './components/DatasetManager/DatasetManager';
import HallucinationDetector from './components/HallucinationDetector/HallucinationDetector';
import DriftDashboard from './components/DriftDashboard/DriftDashboard';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-dark-300">
        <nav className="border-b border-slate-700/50 bg-dark-100/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-8">
                <Link to="/" className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-sm">R</span>
                  </div>
                  <span className="text-white font-semibold text-lg">RAG Eval</span>
                </Link>
                <div className="hidden md:flex items-center gap-6">
                  <Link to="/" className="nav-link px-3 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700/50 transition-colors">Dashboard</Link>
                  <Link to="/experiments" className="nav-link px-3 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700/50 transition-colors">Experiments</Link>
                  <Link to="/benchmark" className="nav-link px-3 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700/50 transition-colors">Benchmark</Link>
                  <Link to="/datasets" className="nav-link px-3 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700/50 transition-colors">Datasets</Link>
                  <Link to="/drift" className="nav-link px-3 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700/50 transition-colors">Drift</Link>
                  <Link to="/hallucination" className="nav-link px-3 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700/50 transition-colors">Hallucination</Link>
                </div>
              </div>
            </div>
          </div>
        </nav>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/experiments" element={<ExperimentList />} />
            <Route path="/benchmark" element={<BenchmarkRunner />} />
            <Route path="/datasets" element={<DatasetManager />} />
            <Route path="/drift" element={<DriftDashboard />} />
            <Route path="/hallucination" element={<HallucinationDetector />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
