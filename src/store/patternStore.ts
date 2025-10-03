import { create } from 'zustand';

interface SymmetryAnalysis {
  symmetries_detected: Record<string, any>;
  mathematical_properties: Record<string, any>;
  dominant_symmetries: string[];
  pattern_complexity_score: number;
}

interface CNNClassification {
  pattern_type?: string;
  subtype?: string;
  region?: string;
  confidence?: number;
  features?: Record<string, any>;
  error?: string;
}

interface AnalysisResult {
  analysis_id: string;
  processing_time_ms: number;
  analysis_version: string;
  analysis_status: {
    symmetry_analysis: string;
    cnn_prediction: string;
    overall_status: string;
  };
  design_classification: {
    type: string;
    subtype: string;
    region: string;
    confidence: number;
    classification_method: string;
  };
  symmetry_analysis: SymmetryAnalysis | { error: string };
  cnn_classification: CNNClassification;
  mathematical_properties: {
    symmetry_type: string;
    rotational_order: number;
    reflection_axes: number;
    complexity_score: number;
    fractal_dimension: number;
    lacunarity: number;
    correlation_dimension: number;
    connectivity_index: number;
    grid_complexity: number;
  };
  cultural_context: {
    ceremonial_use: string;
    seasonal_relevance: string;
    symbolic_meaning: string;
    traditional_name: string;
  };
  metadata: {
    image_dimensions: [number, number];
    color_mode: string;
    processing_timestamp: number;
  };
  error?: string;
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