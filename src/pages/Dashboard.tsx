import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Upload, Eye, Palette, TrendingUp, Star } from 'lucide-react';
import { motion } from 'framer-motion';
import { usePatternStore } from '../store/patternStore';

const Dashboard: React.FC = () => {
  const { analysisHistory } = usePatternStore();
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    patternsCreated: 0,
    culturalInsights: 0,
  });

  useEffect(() => {
    // Calculate stats from analysis history
    setStats({
      totalAnalyses: analysisHistory.length,
      patternsCreated: Math.floor(analysisHistory.length * 0.7), // Mock data
      culturalInsights: analysisHistory.length * 3, // Mock data
    });
  }, [analysisHistory]);

  const quickActions = [
    {
      icon: Upload,
      title: 'Upload Pattern',
      description: 'Analyze a new kolam design',
      link: '/upload',
      color: 'bg-primary-red hover:bg-red-700',
    },
    {
      icon: Eye,
      title: 'Browse Gallery',
      description: 'Explore existing patterns',
      link: '/gallery',
      color: 'bg-accent-indigo hover:bg-indigo-700',
    },
    {
      icon: Palette,
      title: 'Create Pattern',
      description: 'Design your own kolam',
      link: '/create',
      color: 'bg-accent-green hover:bg-green-700',
    },
  ];

  const recentActivity = analysisHistory.slice(0, 3).map((analysis, index) => ({
    id: analysis.analysis_id,
    title: analysis.design_classification.subtype || 'Unknown Pattern',
    type: analysis.design_classification.type,
    confidence: analysis.design_classification.confidence,
  }));

  return (
    <div className="min-h-screen py-12 bg-gradient-to-br from-cream to-yellow-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <h1 className="font-serif text-3xl md:text-4xl font-bold text-primary-red mb-2">
            Kolam Pattern Dashboard
          </h1>
          <p className="font-sans text-gray-600">
            Discover and analyze traditional Kolam patterns
          </p>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8"
        >
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
            <div className="flex items-center">
              <div className="bg-primary-red bg-opacity-10 p-3 rounded-lg">
                <Eye className="h-6 w-6 text-primary-red" />
              </div>
              <div className="ml-4">
                <p className="font-sans text-sm text-gray-600">Patterns Analyzed</p>
                <p className="font-serif text-2xl font-bold text-primary-red">{stats.totalAnalyses}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
            <div className="flex items-center">
              <div className="bg-accent-green bg-opacity-10 p-3 rounded-lg">
                <Palette className="h-6 w-6 text-accent-green" />
              </div>
              <div className="ml-4">
                <p className="font-sans text-sm text-gray-600">Patterns Created</p>
                <p className="font-serif text-2xl font-bold text-accent-green">{stats.patternsCreated}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
            <div className="flex items-center">
              <div className="bg-accent-indigo bg-opacity-10 p-3 rounded-lg">
                <Star className="h-6 w-6 text-accent-indigo" />
              </div>
              <div className="ml-4">
                <p className="font-sans text-sm text-gray-600">Cultural Insights</p>
                <p className="font-serif text-2xl font-bold text-accent-indigo">{stats.culturalInsights}</p>
              </div>
            </div>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="lg:col-span-2"
          >
            <h2 className="font-serif text-2xl font-bold text-primary-red mb-6">Quick Actions</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {quickActions.map((action, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.1 * index }}
                >
                  <Link
                    to={action.link}
                    className={`${action.color} text-white rounded-2xl p-6 block transition-all duration-300 transform hover:scale-105 shadow-lg`}
                  >
                    <div className="flex items-center mb-4">
                      <action.icon className="h-8 w-8 mr-3" />
                      <h3 className="font-serif text-xl font-bold">{action.title}</h3>
                    </div>
                    <p className="font-sans text-white text-opacity-90">{action.description}</p>
                  </Link>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Recent Activity */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <h2 className="font-serif text-2xl font-bold text-primary-red mb-6">Recent Activity</h2>
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
              {recentActivity.length > 0 ? (
                <div className="space-y-4">
                  {recentActivity.map((activity, index) => (
                    <motion.div
                      key={activity.id}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.4, delay: index * 0.1 }}
                      className="flex items-start space-x-3 pb-4 border-b border-gray-100 last:border-b-0 last:pb-0"
                    >
                      <div className="bg-primary-gold bg-opacity-10 p-2 rounded-lg flex-shrink-0">
                        <Eye className="h-4 w-4 text-primary-gold" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-sans font-medium text-gray-900 truncate">
                          {activity.title}
                        </p>
                        <p className="font-sans text-sm text-gray-600">
                          {activity.type} • {Math.round(activity.confidence * 100)}% confidence
                        </p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Eye className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                  <p className="font-sans text-gray-500">No recent activity</p>
                  <p className="font-sans text-sm text-gray-400 mt-1">
                    Start by uploading your first pattern!
                  </p>
                </div>
              )}

              {recentActivity.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <Link
                    to="/history"
                    className="font-sans text-primary-red hover:text-red-700 font-medium text-sm transition-colors"
                  >
                    View all activity →
                  </Link>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Getting Started Guide */}
        {stats.totalAnalyses === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            className="mt-12 bg-gradient-to-r from-primary-red to-red-600 text-white rounded-2xl p-8 text-center"
          >
            <TrendingUp className="h-16 w-16 mx-auto mb-4" />
            <h3 className="font-serif text-2xl font-bold mb-4">Ready to Get Started?</h3>
            <p className="font-sans text-red-100 mb-6 max-w-2xl mx-auto">
              Upload your first kolam pattern and discover the mathematical beauty and cultural significance
              hidden within these traditional designs.
            </p>
            <Link
              to="/upload"
              className="inline-flex items-center bg-primary-gold hover:bg-yellow-500 text-primary-red font-sans font-bold py-3 px-8 rounded-full transition-all duration-300 transform hover:scale-105"
            >
              <Upload className="h-5 w-5 mr-2" />
              Upload Your First Pattern
            </Link>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;