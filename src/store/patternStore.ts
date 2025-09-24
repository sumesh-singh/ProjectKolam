import { create } from 'zustand';

interface AnalysisResult {
  analysis_id: string;
  design_classification: {
    type: string;
    subtype: string;
    region: string;
    confidence: number;
  };
  mathematical_properties: {
    symmetry_type: string;
    rotational_order: number;
    reflection_axes: number;
    complexity_score: number;
  };
  cultural_context: {
    ceremonial_use: string;
    seasonal_relevance: string;
    symbolic_meaning: string;
  };
  processing_time_ms: number;
}

interface PatternState {
  currentAnalysis: AnalysisResult | null;
  analysisHistory: AnalysisResult[];
  isAnalyzing: boolean;
  error: string | null;

  // Actions
  setAnalyzing: (analyzing: boolean) => void;
  setAnalysisResult: (result: AnalysisResult) => void;
  addToHistory: (result: AnalysisResult) => void;
  clearCurrentAnalysis: () => void;
  setError: (error: string | null) => void;
  clearError: () => void;
}

export const usePatternStore = create<PatternState>((set, get) => ({
  currentAnalysis: null,
  analysisHistory: [],
  isAnalyzing: false,
  error: null,

  setAnalyzing: (analyzing: boolean) => {
    set({ isAnalyzing: analyzing });
  },

  setAnalysisResult: (result: AnalysisResult) => {
    set({
      currentAnalysis: result,
      isAnalyzing: false,
      error: null,
    });
    get().addToHistory(result);
  },

  addToHistory: (result: AnalysisResult) => {
    const history = get().analysisHistory;
    const newHistory = [result, ...history.slice(0, 9)]; // Keep last 10
    set({ analysisHistory: newHistory });
  },

  clearCurrentAnalysis: () => {
    set({ currentAnalysis: null });
  },

  setError: (error: string | null) => {
    set({ error, isAnalyzing: false });
  },

  clearError: () => {
    set({ error: null });
  },
}));