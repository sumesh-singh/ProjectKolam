import React from 'react';
import { Link } from 'react-router-dom';
import { Upload, Eye, Download, Palette, ArrowRight, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

const Home: React.FC = () => {
  const fadeInUp = {
    initial: { opacity: 0, y: 60 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  };

  const staggerContainer = {
    animate: {
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const steps = [
    {
      icon: Upload,
      title: 'Upload Photo',
      description: 'Upload a clear photo of your kolam pattern or take one using your camera',
      color: 'text-accent-indigo'
    },
    {
      icon: Eye,
      title: 'AI Analysis',
      description: 'Our advanced AI analyzes and recognizes the pattern structure and style',
      color: 'text-primary-red'
    },
    {
      icon: Download,
      title: 'Digital Recreation',
      description: 'Get a perfect digital recreation that you can download, edit, or share',
      color: 'text-accent-green'
    }
  ];

  const features = [
    'Pattern Recognition Technology',
    'Cultural Pattern Database',
    'Interactive Pattern Editor',
    'High-Quality Downloads'
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-primary-red via-red-600 to-red-800 text-white overflow-hidden">
        <div className="absolute inset-0 bg-black bg-opacity-20"></div>
        <div className="absolute inset-0 bg-kolam-pattern opacity-10"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
          <motion.div 
            className="text-center"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.h1 
              className="font-serif text-4xl md:text-6xl lg:text-7xl font-bold mb-6"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              Discover, Recognize, and
              <span className="text-primary-gold block">Recreate Beautiful</span>
              <span className="text-white block">Kolam Patterns</span>
            </motion.h1>
            
            <motion.p 
              className="font-sans text-xl md:text-2xl text-red-100 mb-8 max-w-3xl mx-auto"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              Preserve and celebrate the ancient art of Kolam with our AI-powered pattern recognition system
            </motion.p>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
            >
              <Link
                to="/upload"
                className="inline-flex items-center bg-primary-gold hover:bg-yellow-500 text-primary-red font-sans font-bold py-4 px-8 rounded-full text-lg transition-all duration-300 transform hover:scale-105 shadow-2xl"
              >
                Get Started
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div {...fadeInUp} className="text-center mb-16">
            <h2 className="font-serif text-4xl md:text-5xl font-bold text-primary-red mb-4">
              How It Works
            </h2>
            <p className="font-sans text-xl text-gray-600 max-w-2xl mx-auto">
              Transform your kolam photos into perfect digital recreations in three simple steps
            </p>
          </motion.div>

          <motion.div 
            className="grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-12"
            variants={staggerContainer}
            initial="initial"
            whileInView="animate"
            viewport={{ once: true }}
          >
            {steps.map((step, index) => (
              <motion.div
                key={index}
                variants={fadeInUp}
                className="text-center group"
              >
                <div className="relative mb-6">
                  <div className="mx-auto w-20 h-20 bg-cream rounded-full flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                    <step.icon className={`h-10 w-10 ${step.color}`} />
                  </div>
                  <div className="absolute -top-2 -right-2 w-8 h-8 bg-primary-gold rounded-full flex items-center justify-center text-primary-red font-bold">
                    {index + 1}
                  </div>
                </div>
                <h3 className="font-serif text-2xl font-semibold text-primary-red mb-4">
                  {step.title}
                </h3>
                <p className="font-sans text-gray-600 leading-relaxed">
                  {step.description}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gradient-to-r from-cream to-yellow-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 className="font-serif text-4xl md:text-5xl font-bold text-primary-red mb-6">
                Preserve Cultural Heritage
              </h2>
              <p className="font-sans text-lg text-gray-700 mb-8 leading-relaxed">
                Our advanced AI technology helps preserve the beautiful art of Kolam for future generations. 
                Whether you&apos;re a beginner learning traditional patterns or an expert creating new designs, 
                our system provides the tools you need.
              </p>
              
              <div className="space-y-4 mb-8">
                {features.map((feature, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -30 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    className="flex items-center"
                  >
                    <Sparkles className="h-5 w-5 text-primary-gold mr-3 flex-shrink-0" />
                    <span className="font-sans text-gray-700">{feature}</span>
                  </motion.div>
                ))}
              </div>
              
              <Link
                to="/about"
                className="inline-flex items-center text-primary-red hover:text-red-700 font-sans font-semibold transition-colors"
              >
                Learn More About Kolam
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="relative"
            >
              <div className="bg-white rounded-2xl shadow-2xl p-8 border-4 border-primary-gold">
                <div className="flex items-center justify-center h-64 bg-cream rounded-xl border-2 border-dashed border-gray-300">
                  <div className="text-center">
                    <Palette className="h-16 w-16 text-primary-red mx-auto mb-4" />
                    <p className="font-sans text-gray-600">Beautiful Kolam Pattern Preview</p>
                  </div>
                </div>
                <div className="mt-6 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="font-sans text-sm text-gray-600">Pattern Type:</span>
                    <span className="font-sans font-semibold text-primary-red">Pulli Kolam</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-sans text-sm text-gray-600">Complexity:</span>
                    <span className="font-sans font-semibold text-accent-green">Intermediate</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-sans text-sm text-gray-600">Recognition:</span>
                    <span className="font-sans font-semibold text-accent-indigo">98% Match</span>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-accent-indigo to-indigo-700 text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Ready to Start Creating?
            </h2>
            <p className="font-sans text-xl text-indigo-100 mb-8">
              Join thousands of artists and enthusiasts preserving the beautiful art of Kolam
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/upload"
                className="inline-flex items-center bg-primary-gold hover:bg-yellow-500 text-primary-red font-sans font-bold py-3 px-8 rounded-full transition-all duration-300 transform hover:scale-105"
              >
                Upload Your Kolam
                <Upload className="ml-2 h-5 w-5" />
              </Link>
              <Link
                to="/gallery"
                className="inline-flex items-center border-2 border-white hover:bg-white hover:text-accent-indigo text-white font-sans font-bold py-3 px-8 rounded-full transition-all duration-300"
              >
                Explore Gallery
                <Eye className="ml-2 h-5 w-5" />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;
