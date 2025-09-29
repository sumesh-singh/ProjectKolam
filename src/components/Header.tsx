import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Flower2 } from 'lucide-react';
import { motion } from 'framer-motion';

const Header: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/dashboard', label: 'Dashboard' },
    { path: '/upload', label: 'Upload' },
    { path: '/gallery', label: 'Gallery' },
    { path: '/about', label: 'About' },
    { path: '/contact', label: 'Contact' },
  ];

  const isActive = (path: string): boolean => location.pathname === path;

  return (
    <header className="bg-white shadow-lg border-b-4 border-primary-gold">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          {/* Logo and Title */}
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="relative">
              <Flower2 className="h-10 w-10 text-primary-red transform group-hover:scale-110 transition-transform duration-300" />
              <div className="absolute inset-0 bg-primary-gold rounded-full opacity-20 scale-125"></div>
            </div>
            <div>
              <h1 className="font-serif text-xl md:text-2xl font-bold text-primary-red">
                Kolam Pattern Recognition
              </h1>
              <p className="text-sm text-gray-600 hidden sm:block">& Recreation System</p>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            {navItems.map(({ path, label }) => (
              <Link
                key={path}
                to={path}
                className={`font-sans font-medium transition-colors duration-300 relative ${
                  isActive(path)
                    ? 'text-primary-red'
                    : 'text-gray-700 hover:text-primary-red'
                }`}
              >
                {label}
                {isActive(path) && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute -bottom-1 left-0 right-0 h-0.5 bg-primary-gold"
                    initial={false}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  />
                )}
              </Link>
            ))}

            {/* Upload Button */}
            <div className="ml-8 pl-8 border-l border-gray-300">
              <Link
                to="/upload"
                className="bg-primary-red hover:bg-red-700 text-white font-sans font-medium py-2 px-6 rounded-lg transition-colors"
              >
                Upload Pattern
              </Link>
            </div>
          </nav>

          {/* Mobile menu button */}
          <button
            className="md:hidden p-2 rounded-md text-gray-700 hover:text-primary-red hover:bg-gray-100"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <motion.nav
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="md:hidden pb-4"
          >
            <div className="space-y-2">
              {navItems.map(({ path, label }) => (
                <Link
                  key={path}
                  to={path}
                  className={`block py-2 px-4 rounded-md font-sans font-medium transition-colors ${
                    isActive(path)
                      ? 'bg-primary-red text-white'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                  onClick={() => setIsMenuOpen(false)}
                >
                  {label}
                </Link>
              ))}

              {/* Mobile Upload Button */}
              <div className="border-t border-gray-200 pt-4 mt-4">
                <Link
                  to="/upload"
                  className="block py-2 px-4 rounded-md font-sans font-medium bg-primary-red hover:bg-red-700 text-white text-center transition-colors"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Upload Pattern
                </Link>
              </div>
            </div>
          </motion.nav>
        )}
      </div>
    </header>
  );
};

export default Header;
