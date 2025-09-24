import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Flower2, User, LogOut } from 'lucide-react';
import { motion } from 'framer-motion';
import { useAuthStore } from '../store/authStore';

const Header: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const { isAuthenticated, user, logout } = useAuthStore();

  const publicNavItems = [
    { path: '/', label: 'Home' },
    { path: '/gallery', label: 'Gallery' },
    { path: '/about', label: 'About' },
    { path: '/contact', label: 'Contact' },
  ];

  const authenticatedNavItems = [
    { path: '/', label: 'Home' },
    { path: '/dashboard', label: 'Dashboard' },
    { path: '/upload', label: 'Upload' },
    { path: '/gallery', label: 'Gallery' },
    { path: '/about', label: 'About' },
    { path: '/contact', label: 'Contact' },
  ];

  const navItems = isAuthenticated ? authenticatedNavItems : publicNavItems;

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

            {/* Authentication Buttons */}
            <div className="flex items-center space-x-4 ml-8 pl-8 border-l border-gray-300">
              {isAuthenticated ? (
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2 text-gray-700">
                    <User className="h-4 w-4" />
                    <span className="font-sans font-medium">{user?.username}</span>
                  </div>
                  <button
                    onClick={logout}
                    className="flex items-center space-x-2 bg-primary-red hover:bg-red-700 text-white font-sans font-medium py-2 px-4 rounded-lg transition-colors"
                  >
                    <LogOut className="h-4 w-4" />
                    <span>Sign Out</span>
                  </button>
                </div>
              ) : (
                <div className="flex items-center space-x-4">
                  <Link
                    to="/login"
                    className="font-sans font-medium text-gray-700 hover:text-primary-red transition-colors"
                  >
                    Sign In
                  </Link>
                  <Link
                    to="/register"
                    className="bg-primary-red hover:bg-red-700 text-white font-sans font-medium py-2 px-4 rounded-lg transition-colors"
                  >
                    Sign Up
                  </Link>
                </div>
              )}
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

              {/* Mobile Authentication */}
              <div className="border-t border-gray-200 pt-4 mt-4">
                {isAuthenticated ? (
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2 px-4 py-2 text-gray-700">
                      <User className="h-4 w-4" />
                      <span className="font-sans font-medium">{user?.username}</span>
                    </div>
                    <button
                      onClick={() => {
                        logout();
                        setIsMenuOpen(false);
                      }}
                      className="w-full flex items-center space-x-2 bg-primary-red hover:bg-red-700 text-white font-sans font-medium py-2 px-4 rounded-lg transition-colors"
                    >
                      <LogOut className="h-4 w-4" />
                      <span>Sign Out</span>
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Link
                      to="/login"
                      className="block py-2 px-4 rounded-md font-sans font-medium text-gray-700 hover:bg-gray-100 transition-colors"
                      onClick={() => setIsMenuOpen(false)}
                    >
                      Sign In
                    </Link>
                    <Link
                      to="/register"
                      className="block py-2 px-4 rounded-md font-sans font-medium bg-primary-red hover:bg-red-700 text-white transition-colors"
                      onClick={() => setIsMenuOpen(false)}
                    >
                      Sign Up
                    </Link>
                  </div>
                )}
              </div>
            </div>
          </motion.nav>
        )}
      </div>
    </header>
  );
};

export default Header;
