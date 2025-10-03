// Upload.tsx
import React, { useState, useRef } from 'react';
import { Upload as UploadIcon, Camera, Image, AlertCircle, CheckCircle, Loader2, Download, Edit3, BarChart3, Layers, Zap, Target, Filter, Eye, EyeOff, Info, X, FileImage, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { patternApi } from '../services/api';
import { usePatternStore } from '../store/patternStore';

// Updated interface to reflect the actual, flatter API response structure.
// The `symmetry_analysis` key is now optional to handle all cases gracefully.
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
  // Made optional to handle cases where analysis fails
  symmetry_analysis?: Record<string, any>;
  cnn_classification: {
    pattern_type?: string;
    subtype?: string;
    region?: string;
    confidence?: number;
    features?: Record<string, any>;
    error?: string;
  };
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

  // Add top-level properties that are likely present in the new API response
  symmetries_detected?: Record<string, any>;
  dominant_symmetries?: string[];
  pattern_complexity_score?: number;
}

// Tooltip definitions for kolam pattern analysis properties
const propertyDefinitions = {
  fractal_dimension: {
    title: "Fractal Dimension",
    description: "Measures pattern complexity and self-similarity. Higher values indicate more intricate, fractal-like structures that repeat at different scales, common in traditional kolam designs."
  },
  lacunarity: {
    title: "Lacunarity",
    description: "Quantifies texture variation and gap distribution. Measures how patterns fill space and the distribution of empty areas, affecting visual rhythm in kolam patterns."
  },
  correlation_dimension: {
    title: "Correlation Dimension",
    description: "Analyzes spatial correlations in pattern elements. Helps understand how pattern components relate to each other spatially in image analysis context."
  },
  connectivity_index: {
    title: "Connectivity Index",
    description: "Component connectedness. Measures how well pattern elements are connected, indicating structural integrity and continuity in kolam designs."
  },
  grid_complexity: {
    title: "Grid Complexity",
    description: "Underlying grid structure. Evaluates the complexity of the foundational grid system used in kolam pattern construction and its geometric intricacy."
  },
  pattern_complexity: {
    title: "Pattern Complexity",
    description: "Overall pattern intricacy. Comprehensive measure of design complexity combining multiple mathematical properties for pattern analysis."
  },
  symmetry_type: {
    title: "Symmetry Type",
    description: "Categorizes rotational/reflection symmetries. Identifies the type of symmetry present (rotational, reflection, translational) in the kolam pattern."
  },
  rotational_order: {
    title: "Rotational Order",
    description: "Number of rotations that map pattern to itself. Indicates how many times the pattern can be rotated while maintaining identical appearance."
  },
  reflection_axes: {
    title: "Reflection Axes",
    description: "Lines of symmetry. Counts the number of mirror lines where the pattern reflects onto itself, common in traditional kolam symmetry."
  },
  complexity_score: {
    title: "Complexity Score",
    description: "Overall pattern intricacy. Quantitative measure of design complexity based on geometric and structural analysis in image processing."
  }
};

const Upload: React.FC = () => {
  // [Keep all the same state and logic as before]
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'symmetry' | 'mathematical' | 'cultural'>('overview');
  const [showVisualizations, setShowVisualizations] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const { currentAnalysis, isAnalyzing, setAnalyzing, setAnalysisResult, setError: setStoreError, clearError } = usePatternStore();
  const safeAnalysis = currentAnalysis as AnalysisResult | null;

  const handleDrag = (e: React.DragEvent): void => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent): void => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>): void => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile: File): void => {
    if (selectedFile.type.startsWith('image/')) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(selectedFile);
      clearError();
      setError(null);
    }
  };

  const analyzePattern = async (): Promise<void> => {
    if (!file) return;

    setAnalyzing(true);
    clearError();

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await patternApi.analyze(formData);

      if (response.data && response.data.result && 'analysis_id' in response.data.result) {
        const result = response.data.result as AnalysisResult;

        if (!result.analysis_status) {
          console.warn('Analysis response missing expected structure, using fallback');
          setStoreError('Analysis completed but with incomplete data. Some features may not be available.');
        }

        setAnalysisResult(result);
      } else {
        throw new Error('Invalid response format from analysis API');
      }
    } catch (error: any) {
      console.error('Analysis error:', error);

      if (error.response) {
        console.error('API Error Response:', {
          status: error.response.status,
          data: error.response.data,
          headers: error.response.headers
        });
      } else if (error.request) {
        console.error('Network Error:', error.request);
      } else {
        console.error('Request Setup Error:', error.message);
      }

      let errorMessage = 'Analysis failed. Please try again.';
      if (error.response?.status === 413) {
        errorMessage = 'Image file is too large. Please choose a smaller image.';
      } else if (error.response?.status === 415) {
        errorMessage = 'Invalid file type. Please upload a valid image file (JPEG, PNG).';
      } else if (error.response?.status === 500) {
        errorMessage = 'Server error. Please try again later.';
      } else if (error.response?.status === 0 || !error.response) {
        errorMessage = 'Network error. Please check your connection and try again.';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }

      setStoreError(errorMessage);
    } finally {
      setAnalyzing(false);
    }
  };

  const regeneratePattern = async (): Promise<void> => {
    if (!safeAnalysis) return;

    try {
      const response = await patternApi.getAnalysis(safeAnalysis.analysis_id);

      if (response.data) {
        console.log('Pattern regenerated successfully:', response.data);
      }
    } catch (error: any) {
      console.error('Pattern regeneration error:', error);
      setStoreError('Failed to regenerate pattern. Please try again.');
    }
  };

  const resetUpload = (): void => {
    setFile(null);
    setPreview(null);
    clearError();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen py-8 bg-gradient-to-br from-cream to-yellow-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header Section - More compact */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-8"
        >
          <h1 className="font-serif text-3xl md:text-4xl lg:text-5xl font-bold text-primary-red mb-3">
            Upload & Recognize Kolam
          </h1>
          <p className="font-sans text-lg text-gray-600 max-w-2xl mx-auto">
            AI-powered analysis and digital recreation of traditional patterns
          </p>
        </motion.div>

        {/* Main Content - Improved Grid Layout */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 lg:gap-8">
          {/* Upload Section - Optimized */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="h-fit"
          >
            <div className="bg-white rounded-xl shadow-lg border-2 border-primary-gold/50 hover:border-primary-gold transition-all duration-300 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-serif text-xl font-bold text-primary-red">Upload Photo</h2>
                {preview && (
                  <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded-full">
                    {(file?.size || 0 / 1024 / 1024).toFixed(2)} MB
                  </span>
                )}
              </div>
              
              {!preview ? (
                <div
                  className={`relative border-2 border-dashed rounded-lg transition-all duration-300 ${
                    dragActive 
                      ? 'border-primary-red bg-red-50/50 scale-[1.02]' 
                      : 'border-gray-300 hover:border-primary-gold hover:bg-yellow-50/30'
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <div className="py-12 px-6 text-center">
                    <div className="relative inline-block mb-4">
                      <div className="absolute inset-0 animate-ping">
                        <UploadIcon className="h-12 w-12 text-primary-gold/30" />
                      </div>
                      <UploadIcon className="h-12 w-12 text-gray-400 relative" />
                    </div>
                    
                    <h3 className="font-sans text-base font-semibold text-gray-700 mb-1">
                      Drop your kolam photo here
                    </h3>
                    <p className="font-sans text-sm text-gray-500 mb-4">or choose a file</p>
                    
                    <div className="flex flex-col sm:flex-row gap-3 justify-center items-center">
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="inline-flex items-center bg-primary-red hover:bg-red-700 text-white font-sans font-medium py-2.5 px-5 rounded-lg transition-all duration-200 shadow-sm hover:shadow-md"
                      >
                        <FileImage className="h-4 w-4 mr-2" />
                        Browse Files
                      </button>
                      
                      <button className="inline-flex items-center text-accent-indigo hover:text-indigo-700 font-sans font-medium transition-colors">
                        <Camera className="h-4 w-4 mr-2" />
                        Take Photo
                      </button>
                    </div>
                    
                    <p className="font-sans text-xs text-gray-400 mt-3">
                      JPEG, PNG • Max 10MB
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Compact Image Preview */}
                  <div className="relative group">
                    <div className="aspect-[4/3] overflow-hidden rounded-lg bg-gray-100">
                      <img
                        src={preview}
                        alt="Uploaded kolam"
                        className="w-full h-full object-contain"
                      />
                    </div>
                    
                    {/* Overlay Controls */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 rounded-lg">
                      <button
                        onClick={resetUpload}
                        className="absolute top-2 right-2 bg-white/90 hover:bg-white text-red-500 p-2 rounded-lg transition-all duration-200 shadow-lg"
                      >
                        <X className="h-4 w-4" />
                      </button>
                      <div className="absolute bottom-2 left-2 right-2">
                        <p className="text-white text-sm font-medium truncate">
                          {file?.name}
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <button
                    onClick={analyzePattern}
                    disabled={isAnalyzing}
                    className="w-full bg-gradient-to-r from-primary-gold to-yellow-400 hover:from-yellow-500 hover:to-primary-gold disabled:from-gray-300 disabled:to-gray-400 text-primary-red disabled:text-gray-600 font-sans font-bold py-3 px-6 rounded-lg transition-all duration-300 shadow-md hover:shadow-lg disabled:shadow-none flex items-center justify-center"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                        Analyzing Pattern...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-5 w-5 mr-2" />
                        Analyze Pattern
                      </>
                    )}
                  </button>
                </div>
              )}
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                className="hidden"
              />
            </div>
          </motion.div>

          {/* Results Section - Enhanced and Compact */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="h-fit"
          >
            <div className="bg-white rounded-xl shadow-lg border-2 border-accent-indigo/50 hover:border-accent-indigo transition-all duration-300">
              {/* Header */}
              <div className="px-6 py-4 border-b border-gray-100">
                <div className="flex items-center justify-between">
                  <h2 className="font-serif text-xl font-bold text-primary-red">Analysis Results</h2>
                  {safeAnalysis && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setShowVisualizations(!showVisualizations)}
                        className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
                      >
                        {showVisualizations ? <EyeOff className="h-4 w-4 text-gray-500" /> : <Eye className="h-4 w-4 text-gray-500" />}
                      </button>
                      <span className="text-xs font-medium bg-accent-green/10 text-accent-green px-2 py-1 rounded-full">
                        {safeAnalysis?.processing_time_ms || 0}ms
                      </span>
                    </div>
                  )}
                </div>
              </div>

              <div className="p-6">
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4"
                  >
                    <div className="flex items-start">
                      <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 mr-2 flex-shrink-0" />
                      <span className="font-sans text-red-700 text-sm">{error}</span>
                    </div>
                  </motion.div>
                )}

                {!currentAnalysis && !isAnalyzing && (
                  <div className="text-center py-16">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-100 rounded-full mb-4">
                      <AlertCircle className="h-8 w-8 text-gray-400" />
                    </div>
                    <p className="font-sans text-gray-500">Upload an image to start analysis</p>
                    <p className="font-sans text-sm text-gray-400 mt-1">AI-powered pattern recognition</p>
                  </div>
                )}

                {isAnalyzing && (
                  <div className="text-center py-16">
                    <div className="relative inline-block mb-4">
                      <div className="absolute inset-0 animate-ping">
                        <div className="w-16 h-16 bg-accent-indigo/20 rounded-full"></div>
                      </div>
                      <Loader2 className="h-16 w-16 text-accent-indigo animate-spin relative" />
                    </div>
                    <p className="font-sans text-gray-600 font-medium">Analyzing kolam pattern...</p>
                    <p className="font-sans text-sm text-gray-500 mt-1">Processing mathematical properties</p>
                  </div>
                )}

                {currentAnalysis && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-4"
                  >
                    {/* Compact Success Status */}
                    <div className="bg-gradient-to-r from-accent-green/10 to-accent-indigo/10 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <div className="w-2 h-2 bg-accent-green rounded-full animate-pulse mr-2"></div>
                          <span className="font-sans font-semibold text-gray-800">Pattern Recognized</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            safeAnalysis?.analysis_status?.symmetry_analysis === 'success'
                              ? 'bg-accent-green/20 text-accent-green'
                              : 'bg-red-100 text-red-600'
                          }`}>
                            Symmetry
                          </span>
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            safeAnalysis?.analysis_status?.cnn_prediction === 'success'
                              ? 'bg-accent-indigo/20 text-accent-indigo'
                              : 'bg-red-100 text-red-600'
                          }`}>
                            CNN
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Compact Tab Navigation */}
                    <div className="flex bg-gray-50 rounded-lg p-1">
                      {[
                        { id: 'overview', icon: BarChart3 },
                        { id: 'symmetry', icon: Layers },
                        { id: 'mathematical', icon: Target },
                        { id: 'cultural', icon: Edit3 }
                      ].map((tab) => (
                        <button
                          key={tab.id}
                          onClick={() => setActiveTab(tab.id as any)}
                          className={`flex-1 flex items-center justify-center py-2 px-2 rounded-md transition-all duration-200 ${
                            activeTab === tab.id
                              ? 'bg-white text-primary-red shadow-sm'
                              : 'text-gray-500 hover:text-gray-700'
                          }`}
                        >
                          <tab.icon className="h-4 w-4" />
                          <span className="ml-1.5 text-xs font-medium hidden sm:inline">{tab.id.charAt(0).toUpperCase() + tab.id.slice(1)}</span>
                        </button>
                      ))}
                    </div>

                    {/* Tab Content - Optimized for space */}
                    <div className="min-h-[200px] max-h-[400px] overflow-y-auto custom-scrollbar">
                      <AnimatePresence mode="wait">
                        {activeTab === 'overview' && (
                          <motion.div
                            key="overview"
                            initial={{ opacity: 0, x: 10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -10 }}
                            transition={{ duration: 0.2 }}
                            className="space-y-4"
                          >
                            {/* Pattern Classification Card */}
                            <div className="bg-gray-50 rounded-lg p-4">
                              <h3 className="font-serif text-lg font-semibold text-primary-red mb-3 truncate">
                                {safeAnalysis?.design_classification?.subtype || 'Unknown Pattern'}
                              </h3>
                              <div className="grid grid-cols-2 gap-3">
                                <div className="bg-white rounded-lg p-3 text-center">
                                  <div className="text-xl font-bold text-accent-green">
                                    {Math.round(safeAnalysis?.design_classification?.confidence || 0)}%
                                  </div>
                                  <div className="text-xs text-gray-600">Confidence</div>
                                </div>
                                <div className="bg-white rounded-lg p-3 text-center">
                                  <div className="text-xl font-bold text-primary-red">
                                    {safeAnalysis?.mathematical_properties?.complexity_score?.toFixed(2) || '0.00'}
                                  </div>
                                  <div className="text-xs text-gray-600">Complexity</div>
                                </div>
                                <div className="bg-white rounded-lg p-3 text-center">
                                  <div className="text-xl font-bold text-accent-indigo">
                                    {safeAnalysis?.mathematical_properties?.rotational_order || 0}
                                  </div>
                                  <div className="text-xs text-gray-600">Rotation</div>
                                </div>
                                <div className="bg-white rounded-lg p-3 text-center">
                                  <div className="text-xl font-bold text-primary-gold">
                                    {safeAnalysis?.mathematical_properties?.reflection_axes || 0}
                                  </div>
                                  <div className="text-xs text-gray-600">Reflection</div>
                                </div>
                              </div>
                            </div>

                           
                          </motion.div>
                        )}

                        {activeTab === 'symmetry' && (
                          <motion.div
                            key="symmetry"
                            initial={{ opacity: 0, x: 10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -10 }}
                            transition={{ duration: 0.2 }}
                            className="space-y-4"
                          >
                            <div className="bg-gray-50 rounded-lg p-4">
                              <h4 className="font-sans font-semibold text-gray-700 mb-3 text-sm">Detected Symmetries</h4>
                              <div className="space-y-2">
                                {(() => {
                                  const symmetries = safeAnalysis?.symmetries_detected ?? safeAnalysis?.symmetry_analysis?.symmetries_detected;
                                  if (!symmetries || Object.keys(symmetries).length === 0) {
                                    return <div className="text-gray-500 text-center py-3 text-sm">No symmetry data available</div>;
                                  }
                                  return Object.entries(symmetries).slice(0, 3).map(([symmetryType, data]: [string, any]) => (
                                    <div key={symmetryType} className="flex items-center justify-between p-2 bg-white rounded-lg">
                                      <div className="flex-1 min-w-0">
                                        <div className="font-sans text-sm font-medium text-primary-red capitalize truncate">
                                          {symmetryType.replace(/_/g, ' ')}
                                        </div>
                                        <div className="text-xs text-gray-600">
                                          Order: {data.order || 'N/A'}
                                        </div>
                                      </div>
                                      <div className="w-16 bg-gray-200 rounded-full h-1.5 ml-3">
                                        <div
                                          className="bg-accent-green h-1.5 rounded-full"
                                          style={{ width: `${(data.confidence || 0) * 100}%` }}
                                        ></div>
                                      </div>
                                    </div>
                                  ));
                                })()}
                              </div>
                            </div>

                            <div className="bg-gray-50 rounded-lg p-4">
                              <h4 className="font-sans font-semibold text-gray-700 mb-3 text-sm">Dominant Symmetries</h4>
                              <div className="flex flex-wrap gap-2">
                                {(() => {
                                  const dominantSymmetries = safeAnalysis?.dominant_symmetries ?? safeAnalysis?.symmetry_analysis?.dominant_symmetries;
                                  if (!dominantSymmetries || dominantSymmetries.length === 0) {
                                    return <span className="text-gray-500 text-sm">None detected</span>;
                                  }
                                  return dominantSymmetries.map((symmetry: string) => (
                                    <span key={symmetry} className="px-2 py-1 bg-primary-gold/20 text-primary-red rounded-full text-xs font-medium truncate max-w-[150px]">
                                      {symmetry.replace(/_/g, ' ')}
                                    </span>
                                  ));
                                })()}
                              </div>
                            </div>
                          </motion.div>
                        )}

                        {activeTab === 'mathematical' && (
                          <motion.div
                            key="mathematical"
                            initial={{ opacity: 0, x: 10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -10 }}
                            transition={{ duration: 0.2 }}
                            className="space-y-4"
                          >
                            <div className="bg-gray-50 rounded-lg p-4">
                              <h4 className="font-sans font-semibold text-gray-700 mb-3 text-sm">Mathematical Properties</h4>
                              <div className="space-y-2">
                                {[
                                  { label: 'Fractal Dimension', value: safeAnalysis?.mathematical_properties?.fractal_dimension, color: 'text-primary-red' },
                                  { label: 'Lacunarity', value: safeAnalysis?.mathematical_properties?.lacunarity, color: 'text-accent-indigo' },
                                  { label: 'Correlation Dim.', value: safeAnalysis?.mathematical_properties?.correlation_dimension, color: 'text-accent-green' },
                                  { label: 'Connectivity', value: safeAnalysis?.mathematical_properties?.connectivity_index, color: 'text-primary-gold' },
                                  { label: 'Grid Complexity', value: safeAnalysis?.mathematical_properties?.grid_complexity, color: 'text-accent-indigo' }
                                ].map((prop) => (
                                  <div key={prop.label} className="flex items-center justify-between py-2 border-b border-gray-200 last:border-0">
                                    <span className="text-xs font-sans text-gray-600">{prop.label}</span>
                                    <span className={`text-sm font-sans font-semibold ${prop.color}`}>
                                      {prop.value?.toFixed(3) || '0.000'}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </motion.div>
                        )}

                        {activeTab === 'cultural' && (
                          <motion.div
                            key="cultural"
                            initial={{ opacity: 0, x: 10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -10 }}
                            transition={{ duration: 0.2 }}
                            className="space-y-4"
                          >
                            <div className="bg-gray-50 rounded-lg p-4">
                              <h4 className="font-sans font-semibold text-gray-700 mb-3 text-sm">Cultural Context</h4>
                              <div className="space-y-3">
                                {[
                                  { label: 'Ceremonial Use', value: safeAnalysis?.cultural_context?.ceremonial_use },
                                  { label: 'Seasonal Relevance', value: safeAnalysis?.cultural_context?.seasonal_relevance },
                                  { label: 'Symbolic Meaning', value: safeAnalysis?.cultural_context?.symbolic_meaning },
                                  { label: 'Traditional Name', value: safeAnalysis?.cultural_context?.traditional_name }
                                ].map((item) => (
                                  <div key={item.label} className="flex justify-between items-center">
                                    <span className="text-xs text-gray-600">{item.label}</span>
                                    <span className="text-xs font-medium text-gray-800 text-right truncate max-w-[150px]">
                                      {item.value || 'Unknown'}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    {/* Action Buttons - Compact */}
                    <div className="grid grid-cols-3 gap-2 pt-3 border-t border-gray-100">
                      <button className="bg-primary-red hover:bg-red-700 text-white font-sans text-sm font-medium py-2.5 px-3 rounded-lg transition-colors flex items-center justify-center">
                        <Download className="h-3.5 w-3.5 mr-1.5" />
                        <span className="hidden sm:inline">Export</span>
                      </button>
                      <button
                        onClick={regeneratePattern}
                        className="bg-accent-indigo hover:bg-indigo-700 text-white font-sans text-sm font-medium py-2.5 px-3 rounded-lg transition-colors flex items-center justify-center"
                      >
                        <Edit3 className="h-3.5 w-3.5 mr-1.5" />
                        <span className="hidden sm:inline">Regenerate</span>
                      </button>
                      <button
                        onClick={resetUpload}
                        className="bg-gray-600 hover:bg-gray-700 text-white font-sans text-sm font-medium py-2.5 px-3 rounded-lg transition-colors flex items-center justify-center"
                      >
                        <UploadIcon className="h-3.5 w-3.5 mr-1.5" />
                        <span className="hidden sm:inline">New</span>
                      </button>
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>
        </div>

        {/* Info Section - More Compact */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-8 bg-white/80 backdrop-blur-sm rounded-xl shadow-md border border-gray-200/50 p-6"
        >
          <h3 className="font-serif text-lg font-bold text-primary-red mb-3">How It Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { step: '1', title: 'Upload', desc: 'Take or upload a photo of your kolam' },
              { step: '2', title: 'Analyze', desc: 'AI processes mathematical properties' },
              { step: '3', title: 'Generate', desc: 'Get digital recreation and insights' }
            ].map((item) => (
              <div key={item.step} className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 bg-primary-gold/20 text-primary-red rounded-full flex items-center justify-center font-bold text-sm">
                  {item.step}
                </div>
                <div>
                  <h4 className="font-sans font-semibold text-gray-800 text-sm">{item.title}</h4>
                  <p className="font-sans text-xs text-gray-600 mt-0.5">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Custom Scrollbar Styles */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #cbd5e0;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #a0aec0;
        }
      `}</style>

      <Tooltip />
    </div>
  );
};

export default Upload;