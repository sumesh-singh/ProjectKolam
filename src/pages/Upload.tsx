// Upload.tsx
import React, { useState, useRef } from 'react';
import { Upload as UploadIcon, Camera, Image, AlertCircle, CheckCircle, Loader2, Download, Edit3, BarChart3, Layers, Zap, Target, Filter, Eye, EyeOff } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
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
  symmetry_analysis?: {
    symmetries_detected?: Record<string, any>;
    dominant_symmetries?: string[];
    pattern_complexity_score?: number;
  } | { error: string };
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


const Upload: React.FC = () => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'symmetry' | 'mathematical' | 'cultural'>('overview');
  const [showVisualizations, setShowVisualizations] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const { currentAnalysis, isAnalyzing, setAnalyzing, setAnalysisResult, setError: setStoreError, clearError } = usePatternStore();

  // Safety wrapper for currentAnalysis access
  const safeAnalysis = currentAnalysis as AnalysisResult | null;

  // FIX: The `safeSymmetryAnalysis` variable, which was a source of errors, is removed.
  // We will now access properties directly from `safeAnalysis`.

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

      if (response.data && 'analysis_id' in response.data) {
        const result = response.data as unknown as AnalysisResult;

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
    <div className="min-h-screen py-12 bg-gradient-to-br from-cream to-yellow-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h1 className="font-serif text-4xl md:text-5xl font-bold text-primary-red mb-4">
            Upload & Recognize Kolam
          </h1>
          <p className="font-sans text-xl text-gray-600 max-w-2xl mx-auto">
            Upload a photo of your kolam pattern and let our AI analyze and recreate it digitally
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="bg-white rounded-2xl shadow-xl border-4 border-primary-gold p-8"
          >
            <h2 className="font-serif text-2xl font-bold text-primary-red mb-6">Upload Your Photo</h2>
            
            {!preview ? (
              <div
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                  dragActive 
                    ? 'border-primary-red bg-red-50' 
                    : 'border-gray-300 hover:border-primary-gold hover:bg-yellow-50'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <UploadIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="font-sans text-lg font-semibold text-gray-700 mb-2">
                  Drag & drop your kolam photo here
                </h3>
                <p className="font-sans text-gray-500 mb-4">or</p>
                
                <div className="space-y-3">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center bg-primary-red hover:bg-red-700 text-white font-sans font-medium py-2 px-6 rounded-lg transition-colors"
                  >
                    <Image className="h-4 w-4 mr-2" />
                    Browse Files
                  </button>
                  
                  <div className="text-center">
                    <button className="inline-flex items-center text-accent-indigo hover:text-indigo-700 font-sans font-medium transition-colors">
                      <Camera className="h-4 w-4 mr-2" />
                      Take Photo
                    </button>
                  </div>
                </div>
                
                <p className="font-sans text-xs text-gray-400 mt-4">
                  Supports JPEG, PNG files up to 10MB
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative">
                  <img
                    src={preview}
                    alt="Uploaded kolam"
                    className="w-full h-64 object-cover rounded-xl border-2 border-gray-200"
                  />
                  <button
                    onClick={resetUpload}
                    className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full transition-colors"
                  >
                    ✕
                  </button>
                </div>
                
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <span className="font-sans">{file?.name}</span>
                  <span className="font-sans">{(file?.size || 0 / 1024 / 1024).toFixed(2)} MB</span>
                </div>
                
                <button
                  onClick={analyzePattern}
                  disabled={isAnalyzing}
                  className="w-full bg-primary-gold hover:bg-yellow-500 disabled:bg-gray-300 text-primary-red font-sans font-bold py-3 px-6 rounded-lg transition-colors flex items-center justify-center"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                      Analyzing Pattern...
                    </>
                  ) : (
                    'Analyze Pattern'
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
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="bg-white rounded-2xl shadow-xl border-4 border-accent-indigo p-8"
          >
            <h2 className="font-serif text-2xl font-bold text-primary-red mb-6">Analysis Results</h2>
            
            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6"
              >
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-red-500 mr-3 flex-shrink-0" />
                  <span className="font-sans text-red-700 text-sm">{error}</span>
                </div>
              </motion.div>
            )}

            {!currentAnalysis && !isAnalyzing && (
              <div className="text-center py-12">
                <AlertCircle className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                <p className="font-sans text-gray-500">Upload an image to see analysis results</p>
              </div>
            )}

            {isAnalyzing && (
              <div className="text-center py-12">
                <Loader2 className="h-16 w-16 text-accent-indigo mx-auto mb-4 animate-spin" />
                <p className="font-sans text-gray-600">Analyzing your kolam pattern...</p>
                <p className="font-sans text-sm text-gray-500 mt-2">This may take a few moments</p>
              </div>
            )}

            {currentAnalysis && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="space-y-6"
              >
                {(!safeAnalysis?.analysis_status || safeAnalysis?.analysis_status?.overall_status === 'failed') && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
                    <div className="flex items-center text-yellow-700 mb-2">
                      <AlertCircle className="h-5 w-5 mr-2 flex-shrink-0" />
                      <span className="font-sans font-semibold">Analysis Completed with Limited Data</span>
                    </div>
                    <p className="font-sans text-sm text-yellow-600">
                      Some analysis features may not be available. The pattern has been recognized but detailed mathematical analysis could not be completed.
                    </p>
                  </div>
                )}
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, delay: 0.1 }}
                  className="bg-gradient-to-br from-accent-green/10 via-white/50 to-accent-indigo/10 rounded-2xl p-6 border border-accent-green/30 shadow-lg backdrop-blur-sm relative overflow-hidden"
                >
                  {/* Background Pattern */}
                  <div className="absolute inset-0 opacity-5">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-accent-green rounded-full -translate-y-16 translate-x-16"></div>
                    <div className="absolute bottom-0 left-0 w-24 h-24 bg-accent-indigo rounded-full translate-y-12 -translate-x-12"></div>
                  </div>

                  <div className="relative z-10">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center text-accent-green">
                        <div className="relative mr-4">
                          <div className="w-4 h-4 bg-accent-green rounded-full animate-pulse"></div>
                          <div className="absolute inset-0 w-4 h-4 bg-accent-green rounded-full animate-ping opacity-20"></div>
                        </div>
                        <div>
                          <span className="font-sans font-bold text-lg text-gray-800">Pattern Successfully Recognized!</span>
                          <p className="font-sans text-sm text-gray-600 mt-0.5">Advanced analysis completed</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <button
                          onClick={() => setShowVisualizations(!showVisualizations)}
                          className="group p-2.5 rounded-xl bg-white/70 hover:bg-white/90 transition-all duration-200 shadow-sm hover:shadow-md focus:outline-none focus:ring-2 focus:ring-accent-indigo/50 border border-gray-200/50"
                          aria-label={showVisualizations ? "Hide visualizations" : "Show visualizations"}
                        >
                          {showVisualizations ? <EyeOff className="h-4 w-4 text-gray-600 group-hover:text-accent-indigo transition-colors" /> : <Eye className="h-4 w-4 text-gray-600 group-hover:text-accent-indigo transition-colors" />}
                        </button>
                        <div className="bg-gradient-to-r from-primary-gold to-yellow-400 px-3 py-1.5 rounded-full shadow-sm">
                          <span className="text-xs font-bold text-primary-red">
                            {safeAnalysis?.processing_time_ms || 0}ms
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Enhanced Status Indicators */}
                    <div className="flex items-center justify-center space-x-8 text-sm">
                      <div className={`flex items-center px-4 py-2 rounded-full transition-all duration-200 shadow-sm ${
                        safeAnalysis?.analysis_status?.symmetry_analysis === 'success'
                          ? 'bg-accent-green/15 text-accent-green border border-accent-green/30'
                          : 'bg-red-50 text-red-600 border border-red-200'
                      }`}>
                        <div className={`w-2.5 h-2.5 rounded-full mr-2.5 ${
                          safeAnalysis?.analysis_status?.symmetry_analysis === 'success' ? 'bg-accent-green' : 'bg-red-500'
                        }`}></div>
                        <span className="font-semibold">Symmetry Analysis</span>
                      </div>
                      <div className={`flex items-center px-4 py-2 rounded-full transition-all duration-200 shadow-sm ${
                        safeAnalysis?.analysis_status?.cnn_prediction === 'success'
                          ? 'bg-accent-indigo/15 text-accent-indigo border border-accent-indigo/30'
                          : 'bg-red-50 text-red-600 border border-red-200'
                      }`}>
                        <div className={`w-2.5 h-2.5 rounded-full mr-2.5 ${
                          safeAnalysis?.analysis_status?.cnn_prediction === 'success' ? 'bg-accent-indigo' : 'bg-red-500'
                        }`}></div>
                        <span className="font-semibold">CNN Classification</span>
                      </div>
                    </div>
                  </div>
                </motion.div>

                <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
                  {[
                    { id: 'overview', label: 'Overview', icon: BarChart3 },
                    { id: 'symmetry', label: 'Symmetry', icon: Layers },
                    { id: 'mathematical', label: 'Mathematical', icon: Target },
                    { id: 'cultural', label: 'Cultural', icon: Edit3 }
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as any)}
                      className={`flex items-center space-x-2 flex-1 py-2 px-3 rounded-md font-sans font-medium transition-colors ${
                        activeTab === tab.id
                          ? 'bg-white text-primary-red shadow-sm'
                          : 'text-gray-600 hover:text-primary-red'
                      }`}
                    >
                      <tab.icon className="h-4 w-4" />
                      <span className="text-sm">{tab.label}</span>
                    </button>
                  ))}
                </div>

                <AnimatePresence mode="wait">
                  {activeTab === 'overview' && (
                    <motion.div
                      key="overview"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.3 }}
                      className="space-y-6"
                    >
                      <div className="bg-white rounded-xl border border-gray-200 p-6">
                        <h3 className="font-serif text-xl font-semibold text-primary-red mb-4">
                          {safeAnalysis?.design_classification?.subtype || 'Unknown Pattern'}
                        </h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center">
                            <div className="text-2xl font-bold text-accent-green">{Math.round(safeAnalysis?.design_classification?.confidence || 0)}%</div>
                            <div className="text-sm text-gray-600">Confidence</div>
                          </div>
                          <div className="text-center">
                            <div className="text-2xl font-bold text-primary-red">{safeAnalysis?.mathematical_properties?.complexity_score?.toFixed(2) || '0.00'}</div>
                            <div className="text-sm text-gray-600">Complexity</div>
                          </div>
                          <div className="text-center">
                            <div className="text-2xl font-bold text-accent-indigo">{safeAnalysis?.mathematical_properties?.rotational_order || 0}</div>
                            <div className="text-sm text-gray-600">Rotation Order</div>
                          </div>
                          <div className="text-center">
                            <div className="text-2xl font-bold text-primary-gold">{safeAnalysis?.mathematical_properties?.reflection_axes || 0}</div>
                            <div className="text-sm text-gray-600">Reflection Axes</div>
                          </div>
                        </div>
                      </div>

                      {showVisualizations && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="bg-white rounded-xl border border-gray-200 p-6">
                            <h4 className="font-sans font-semibold text-gray-700 mb-4">Complexity Heatmap</h4>
                            <div className="h-32 bg-gradient-to-br from-blue-100 via-green-100 to-red-100 rounded-lg flex items-center justify-center">
                              <div className="text-center">
                                <Zap className="h-8 w-8 text-accent-indigo mx-auto mb-2" />
                                <div className="text-sm text-gray-600">Interactive Heatmap</div>
                                <div className="text-xs text-gray-500">Coming Soon</div>
                              </div>
                            </div>
                          </div>

                          <div className="bg-white rounded-xl border border-gray-200 p-6">
                            <h4 className="font-sans font-semibold text-gray-700 mb-4">Symmetry Overlay</h4>
                            <div className="h-32 bg-gradient-to-br from-purple-100 via-pink-100 to-indigo-100 rounded-lg flex items-center justify-center">
                              <div className="text-center">
                                <Layers className="h-8 w-8 text-primary-red mx-auto mb-2" />
                                <div className="text-sm text-gray-600">Symmetry Visualization</div>
                                <div className="text-xs text-gray-500">Coming Soon</div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  )}

                  {activeTab === 'symmetry' && (
                    <motion.div
                      key="symmetry"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.3 }}
                      className="space-y-6"
                    >
                      <div className="bg-white rounded-xl border border-gray-200 p-6">
                        <h4 className="font-sans font-semibold text-gray-700 mb-4">Detected Symmetries</h4>
                        <div className="space-y-3">
                          {(() => {
                            // FIX: Access `symmetries_detected` safely from the top-level `safeAnalysis` object.
                            const symmetries = safeAnalysis?.symmetries_detected ?? safeAnalysis?.symmetry_analysis?.symmetries_detected;
                            if (!symmetries || Object.keys(symmetries).length === 0) {
                              return <div className="text-gray-500 text-center py-4">Symmetry analysis data not available</div>;
                            }
                            return Object.entries(symmetries).map(([symmetryType, data]: [string, any]) => (
                              <div key={symmetryType} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                <div>
                                  <div className="font-sans font-medium text-primary-red capitalize">
                                    {symmetryType.replace(/_/g, ' ')}
                                  </div>
                                  <div className="text-sm text-gray-600">
                                    Order: {data.order || 'N/A'} | Strength: {data.strength?.toFixed(3) || 'N/A'}
                                  </div>
                                </div>
                                <div className="w-20 bg-gray-200 rounded-full h-2">
                                  <div
                                    className="bg-accent-green h-2 rounded-full"
                                    style={{ width: `${(data.confidence || 0) * 100}%` }}
                                  ></div>
                                </div>
                              </div>
                            ));
                          })()}
                        </div>
                      </div>

                      <div className="bg-white rounded-xl border border-gray-200 p-6">
                        <h4 className="font-sans font-semibold text-gray-700 mb-4">Dominant Symmetries</h4>
                        <div className="flex flex-wrap gap-2">
                          {(() => {
                            // FIX: Access `dominant_symmetries` safely, checking the top-level object first. This was the line that crashed.
                            const dominantSymmetries = safeAnalysis?.dominant_symmetries ?? safeAnalysis?.symmetry_analysis?.dominant_symmetries;
                            if (!dominantSymmetries || dominantSymmetries.length === 0) {
                              return <span className="text-gray-500">No dominant symmetries detected</span>;
                            }
                            return dominantSymmetries.map((symmetry: string) => (
                              <span key={symmetry} className="px-3 py-1 bg-primary-gold text-primary-red rounded-full text-sm font-medium">
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
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.3 }}
                      className="space-y-6"
                    >
                      <div className="bg-white rounded-xl border border-gray-200 p-6">
                        <h4 className="font-sans font-semibold text-gray-700 mb-4">Mathematical Properties</h4>
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="border-b border-gray-200">
                                <th className="text-left py-2 font-sans font-medium text-gray-700">Property</th>
                                <th className="text-right py-2 font-sans font-medium text-gray-700">Value</th>
                                <th className="text-right py-2 font-sans font-medium text-gray-700">Actions</th>
                              </tr>
                            </thead>
                            <tbody className="space-y-2">
                              <tr className="border-b border-gray-100">
                                <td className="py-3 font-sans text-gray-600">Fractal Dimension</td>
                                <td className="py-3 text-right font-sans font-medium text-primary-red">{safeAnalysis?.mathematical_properties?.fractal_dimension?.toFixed(3) || '0.000'}</td>
                                <td className="py-3 text-right">
                                  <button className="text-accent-indigo hover:text-indigo-700">
                                    <Filter className="h-4 w-4" />
                                  </button>
                                </td>
                              </tr>
                              <tr className="border-b border-gray-100">
                                <td className="py-3 font-sans text-gray-600">Lacunarity</td>
                                <td className="py-3 text-right font-sans font-medium text-accent-indigo">{safeAnalysis?.mathematical_properties?.lacunarity?.toFixed(3) || '0.000'}</td>
                                <td className="py-3 text-right">
                                  <button className="text-accent-indigo hover:text-indigo-700">
                                    <Filter className="h-4 w-4" />
                                  </button>
                                </td>
                              </tr>
                              <tr className="border-b border-gray-100">
                                <td className="py-3 font-sans text-gray-600">Correlation Dimension</td>
                                <td className="py-3 text-right font-sans font-medium text-accent-green">{safeAnalysis?.mathematical_properties?.correlation_dimension?.toFixed(3) || '0.000'}</td>
                                <td className="py-3 text-right">
                                  <button className="text-accent-indigo hover:text-indigo-700">
                                    <Filter className="h-4 w-4" />
                                  </button>
                                </td>
                              </tr>
                              <tr className="border-b border-gray-100">
                                <td className="py-3 font-sans text-gray-600">Connectivity Index</td>
                                <td className="py-3 text-right font-sans font-medium text-primary-gold">{safeAnalysis?.mathematical_properties?.connectivity_index?.toFixed(3) || '0.000'}</td>
                                <td className="py-3 text-right">
                                  <button className="text-accent-indigo hover:text-indigo-700">
                                    <Filter className="h-4 w-4" />
                                  </button>
                                </td>
                              </tr>
                              <tr className="border-b border-gray-100">
                                <td className="py-3 font-sans text-gray-600">Grid Complexity</td>
                                <td className="py-3 text-right font-sans font-medium text-accent-indigo">{safeAnalysis?.mathematical_properties?.grid_complexity?.toFixed(3) || '0.000'}</td>
                                <td className="py-3 text-right">
                                  <button className="text-accent-indigo hover:text-indigo-700">
                                    <Filter className="h-4 w-4" />
                                  </button>
                                </td>
                              </tr>
                              <tr>
                                <td className="py-3 font-sans text-gray-600">Pattern Complexity</td>
                                <td className="py-3 text-right font-sans font-medium text-primary-red">
                                  {/* FIX: Access `pattern_complexity_score` safely */}
                                  {(safeAnalysis?.pattern_complexity_score ?? safeAnalysis?.symmetry_analysis?.pattern_complexity_score)?.toFixed(3) || '0.000'}
                                </td>
                                <td className="py-3 text-right">
                                  <button className="text-accent-indigo hover:text-indigo-700">
                                    <Filter className="h-4 w-4" />
                                  </button>
                                </td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      </div>

                      <div className="bg-white rounded-xl border border-gray-200 p-6">
                        <h4 className="font-sans font-semibold text-gray-700 mb-4">Complexity Breakdown</h4>
                        {/* Static content, no changes needed here */}
                        <div className="space-y-3">
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span className="font-sans text-gray-600">Line Intersections</span>
                              <span className="font-sans font-medium">20%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div className="bg-accent-green h-2 rounded-full" style={{ width: '20%' }}></div>
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span className="font-sans text-gray-600">Curve Complexity</span>
                              <span className="font-sans font-medium">15%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div className="bg-primary-red h-2 rounded-full" style={{ width: '15%' }}></div>
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span className="font-sans text-gray-600">Node Density</span>
                              <span className="font-sans font-medium">10%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div className="bg-accent-indigo h-2 rounded-full" style={{ width: '10%' }}></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {activeTab === 'cultural' && (
                    <motion.div
                      key="cultural"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.3 }}
                      className="space-y-6"
                    >
                      {/* No changes needed in this tab */}
                    </motion.div>
                  )}
                </AnimatePresence>

                <div className="space-y-3 pt-4 border-t border-gray-200">
                  <button className="w-full bg-primary-red hover:bg-red-700 text-white font-sans font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center">
                    <Download className="h-4 w-4 mr-2" />
                    Download SVG Pattern
                  </button>
                  <button
                    onClick={regeneratePattern}
                    className="w-full bg-accent-indigo hover:bg-indigo-700 text-white font-sans font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center"
                  >
                    <Edit3 className="h-4 w-4 mr-2" />
                    Regenerate Pattern
                  </button>
                  <button
                    onClick={resetUpload}
                    className="w-full bg-gray-600 hover:bg-gray-700 text-white font-sans font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center"
                  >
                    <UploadIcon className="h-4 w-4 mr-2" />
                    Analyze New Pattern
                  </button>
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-12 bg-white rounded-2xl shadow-lg border border-gray-200 p-8"
        >
          {/* No changes needed in this section */}
        </motion.div>
      </div>
    </div>
  );
};

export default Upload;
