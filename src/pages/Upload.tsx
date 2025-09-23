import React, { useState, useRef } from 'react';
import { Upload as UploadIcon, Camera, Image, AlertCircle, CheckCircle, Loader2, Download, Edit3 } from 'lucide-react';
import { motion } from 'framer-motion';

interface AnalysisResult {
  patternName: string;
  patternType: string;
  confidence: number;
  description: string;
  culturalNotes: string;
}

const Upload: React.FC = () => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
      setAnalysisResult(null);
    }
  };

  const analyzePattern = async (): Promise<void> => {
    if (!file) return;
    
    setIsAnalyzing(true);
    
    // Simulate AI analysis
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    setAnalysisResult({
      patternName: "Lotus Pulli Kolam",
      patternType: "Pulli (Dot-based)",
      confidence: 94.5,
      description: "A traditional lotus-inspired kolam pattern featuring concentric circular designs with intricate dot connections. This pattern symbolizes purity, prosperity, and spiritual awakening.",
      culturalNotes: "Lotus patterns in kolam art represent divine beauty and the emergence of consciousness. Traditionally drawn during festivals and special occasions to invite positive energy into the home."
    });
    
    setIsAnalyzing(false);
  };

  const resetUpload = (): void => {
    setFile(null);
    setPreview(null);
    setAnalysisResult(null);
    setIsAnalyzing(false);
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
            
            {!analysisResult && !isAnalyzing && (
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
            
            {analysisResult && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="space-y-6"
              >
                <div className="flex items-center text-accent-green mb-4">
                  <CheckCircle className="h-6 w-6 mr-2" />
                  <span className="font-sans font-semibold">Pattern Successfully Recognized!</span>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h3 className="font-serif text-xl font-semibold text-primary-red mb-2">
                      {analysisResult.patternName}
                    </h3>
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-sans text-gray-600">Type: {analysisResult.patternType}</span>
                      <span className="font-sans font-semibold text-accent-green">
                        {analysisResult.confidence}% confidence
                      </span>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-sans font-semibold text-gray-700 mb-2">Description</h4>
                    <p className="font-sans text-gray-600 text-sm leading-relaxed">
                      {analysisResult.description}
                    </p>
                  </div>
                  
                  <div>
                    <h4 className="font-sans font-semibold text-gray-700 mb-2">Cultural Significance</h4>
                    <p className="font-sans text-gray-600 text-sm leading-relaxed">
                      {analysisResult.culturalNotes}
                    </p>
                  </div>
                  
                  <div className="flex space-x-3 pt-4">
                    <button className="flex-1 bg-primary-red hover:bg-red-700 text-white font-sans font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center">
                      <Download className="h-4 w-4 mr-2" />
                      Download SVG
                    </button>
                    <button className="flex-1 bg-accent-indigo hover:bg-indigo-700 text-white font-sans font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center">
                      <Edit3 className="h-4 w-4 mr-2" />
                      Edit Pattern
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>

        {/* Tips Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-12 bg-white rounded-2xl shadow-lg border border-gray-200 p-8"
        >
          <h3 className="font-serif text-2xl font-bold text-primary-red mb-6">Tips for Best Results</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="w-12 h-12 bg-primary-gold rounded-full flex items-center justify-center mx-auto mb-3">
                <Camera className="h-6 w-6 text-primary-red" />
              </div>
              <h4 className="font-sans font-semibold text-gray-700 mb-2">Good Lighting</h4>
              <p className="font-sans text-sm text-gray-600">Take photos in bright, even lighting for better recognition</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-accent-indigo rounded-full flex items-center justify-center mx-auto mb-3">
                <Image className="h-6 w-6 text-white" />
              </div>
              <h4 className="font-sans font-semibold text-gray-700 mb-2">Clear Focus</h4>
              <p className="font-sans text-sm text-gray-600">Ensure the pattern is sharp and in focus</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-accent-green rounded-full flex items-center justify-center mx-auto mb-3">
                <UploadIcon className="h-6 w-6 text-white" />
              </div>
              <h4 className="font-sans font-semibold text-gray-700 mb-2">Full Pattern</h4>
              <p className="font-sans text-sm text-gray-600">Capture the complete kolam pattern in the frame</p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Upload;
