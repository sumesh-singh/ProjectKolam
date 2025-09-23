import React, { useState } from 'react';
import { Search, Filter, Download, Eye, Heart } from 'lucide-react';
import { motion } from 'framer-motion';

interface KolamPattern {
  id: number;
  name: string;
  type: string;
  complexity: 'Beginner' | 'Intermediate' | 'Advanced';
  imageUrl: string;
  description: string;
  likes: number;
  downloads: number;
}

const Gallery: React.FC = () => {
  const [selectedFilter, setSelectedFilter] = useState('All');
  const [searchTerm, setSearchTerm] = useState('');

  const patterns: KolamPattern[] = [
    {
      id: 1,
      name: "Lotus Pulli Kolam",
      type: "Pulli",
      complexity: "Intermediate",
      imageUrl: "https://img-wrapper.vercel.app/image?url=https://placehold.co/300x300/B22222/FFFFFF?text=Lotus+Kolam",
      description: "Traditional lotus pattern with intricate dot connections",
      likes: 245,
      downloads: 89
    },
    {
      id: 2,
      name: "Peacock Sikku",
      type: "Sikku",
      complexity: "Advanced",
      imageUrl: "https://img-wrapper.vercel.app/image?url=https://placehold.co/300x300/3F51B5/FFFFFF?text=Peacock+Kolam",
      description: "Elegant peacock design with flowing curves",
      likes: 192,
      downloads: 67
    },
    {
      id: 3,
      name: "Simple Flower",
      type: "Freehand",
      complexity: "Beginner",
      imageUrl: "https://img-wrapper.vercel.app/image?url=https://placehold.co/300x300/388E3C/FFFFFF?text=Flower+Kolam",
      description: "Basic flower pattern perfect for beginners",
      likes: 156,
      downloads: 124
    },
    {
      id: 4,
      name: "Geometric Mandala",
      type: "Pulli",
      complexity: "Advanced",
      imageUrl: "https://img-wrapper.vercel.app/image?url=https://placehold.co/300x300/FFD700/000000?text=Mandala+Kolam",
      description: "Complex geometric mandala with symmetrical patterns",
      likes: 312,
      downloads: 95
    },
    {
      id: 5,
      name: "Traditional Rangoli",
      type: "Freehand",
      complexity: "Intermediate",
      imageUrl: "https://img-wrapper.vercel.app/image?url=https://placehold.co/300x300/B22222/FFFFFF?text=Rangoli+Kolam",
      description: "Classic rangoli design with traditional motifs",
      likes: 203,
      downloads: 78
    },
    {
      id: 6,
      name: "Star Pattern",
      type: "Sikku",
      complexity: "Beginner",
      imageUrl: "https://img-wrapper.vercel.app/image?url=https://placehold.co/300x300/3F51B5/FFFFFF?text=Star+Kolam",
      description: "Simple star pattern with connecting lines",
      likes: 187,
      downloads: 156
    }
  ];

  const filters = ['All', 'Pulli', 'Sikku', 'Freehand'];
  const complexityColors = {
    Beginner: 'bg-accent-green text-white',
    Intermediate: 'bg-primary-gold text-primary-red',
    Advanced: 'bg-primary-red text-white'
  };

  const filteredPatterns = patterns.filter(pattern => {
    const matchesFilter = selectedFilter === 'All' || pattern.type === selectedFilter;
    const matchesSearch = pattern.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         pattern.description.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  return (
    <div className="min-h-screen py-12 bg-gradient-to-br from-cream to-yellow-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h1 className="font-serif text-4xl md:text-5xl font-bold text-primary-red mb-4">
            Kolam Gallery
          </h1>
          <p className="font-sans text-xl text-gray-600 max-w-2xl mx-auto">
            Explore our collection of traditional and contemporary kolam patterns
          </p>
        </motion.div>

        {/* Search and Filter */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6 mb-8"
        >
          <div className="flex flex-col md:flex-row gap-4 items-center">
            {/* Search */}
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search patterns..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-sans"
              />
            </div>

            {/* Filters */}
            <div className="flex items-center space-x-2">
              <Filter className="h-5 w-5 text-gray-600" />
              <div className="flex space-x-2">
                {filters.map((filter) => (
                  <button
                    key={filter}
                    onClick={() => setSelectedFilter(filter)}
                    className={`px-4 py-2 rounded-lg font-sans font-medium transition-colors ${
                      selectedFilter === filter
                        ? 'bg-primary-red text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {filter}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Gallery Grid */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
        >
          {filteredPatterns.map((pattern, index) => (
            <motion.div
              key={pattern.id}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden group hover:shadow-2xl transition-all duration-300"
            >
              <div className="relative">
                <img
                  src={pattern.imageUrl}
                  alt={pattern.name}
                  className="w-full h-64 object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute top-4 right-4">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${complexityColors[pattern.complexity]}`}>
                    {pattern.complexity}
                  </span>
                </div>
                <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-300 flex items-center justify-center">
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex space-x-3">
                    <button className="bg-white text-primary-red p-3 rounded-full hover:bg-gray-100 transition-colors">
                      <Eye className="h-5 w-5" />
                    </button>
                    <button className="bg-white text-primary-red p-3 rounded-full hover:bg-gray-100 transition-colors">
                      <Download className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              </div>

              <div className="p-6">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-serif text-xl font-semibold text-primary-red">
                    {pattern.name}
                  </h3>
                  <span className="text-sm font-medium text-accent-indigo bg-indigo-50 px-2 py-1 rounded">
                    {pattern.type}
                  </span>
                </div>
                
                <p className="font-sans text-gray-600 text-sm mb-4 leading-relaxed">
                  {pattern.description}
                </p>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <div className="flex items-center">
                      <Heart className="h-4 w-4 mr-1" />
                      {pattern.likes}
                    </div>
                    <div className="flex items-center">
                      <Download className="h-4 w-4 mr-1" />
                      {pattern.downloads}
                    </div>
                  </div>
                  
                  <button className="bg-primary-gold hover:bg-yellow-500 text-primary-red font-sans font-medium py-2 px-4 rounded-lg transition-colors text-sm">
                    View Details
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {filteredPatterns.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12"
          >
            <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Search className="h-12 w-12 text-gray-400" />
            </div>
            <h3 className="font-serif text-xl text-gray-600 mb-2">No patterns found</h3>
            <p className="font-sans text-gray-500">Try adjusting your search or filter criteria</p>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default Gallery;
