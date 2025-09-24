import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Eye, EyeOff, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { useAuthStore } from '../store/authStore';
import { userApi } from '../services/api';

const Register: React.FC = () => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState({
    length: false,
    hasUpper: false,
    hasLower: false,
    hasNumber: false,
  });

  const { login, error, setError, clearError } = useAuthStore();
  const navigate = useNavigate();

  useEffect(() => {
    clearError();
  }, [clearError]);

  useEffect(() => {
    // Check password strength
    const password = formData.password;
    setPasswordStrength({
      length: password.length >= 8,
      hasUpper: /[A-Z]/.test(password),
      hasLower: /[a-z]/.test(password),
      hasNumber: /\d/.test(password),
    });
  }, [formData.password]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    if (error) clearError();
  };

  const validateForm = () => {
    if (!formData.username || !formData.email || !formData.password || !formData.confirmPassword) {
      setError('Please fill in all fields');
      return false;
    }

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return false;
    }

    if (!passwordStrength.length || !passwordStrength.hasUpper || !passwordStrength.hasLower || !passwordStrength.hasNumber) {
      setError('Password does not meet requirements');
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      setError('Please enter a valid email address');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await userApi.register({
        username: formData.username,
        email: formData.email,
        password: formData.password,
      });

      if (response.data?.data) {
        // Auto-login after successful registration
        const loginResponse = await userApi.login({
          username: formData.username,
          password: formData.password,
        });

        if (loginResponse.data?.data) {
          const { access_token } = loginResponse.data.data;
          login(response.data.data, access_token);
          navigate('/dashboard');
        }
      }
    } catch (error: any) {
      console.error('Registration error:', error);
      const errorMessage = error.response?.data?.detail || 'Registration failed. Please try again.';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const passwordStrengthScore = Object.values(passwordStrength).filter(Boolean).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-cream to-yellow-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="max-w-md w-full space-y-8"
      >
        <div className="text-center">
          <h1 className="font-serif text-3xl md:text-4xl font-bold text-primary-red mb-2">
            Join Kolam Design
          </h1>
          <p className="font-sans text-gray-600">
            Create your account to start preserving cultural heritage
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="bg-white rounded-2xl shadow-xl border-4 border-primary-gold p-8"
        >
          <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center"
              >
                <AlertCircle className="h-5 w-5 text-red-500 mr-3 flex-shrink-0" />
                <span className="font-sans text-red-700 text-sm">{error}</span>
              </motion.div>
            )}

            <div>
              <label htmlFor="username" className="block font-sans font-medium text-gray-700 mb-2">
                Username
              </label>
              <input
                id="username"
                name="username"
                type="text"
                required
                value={formData.username}
                onChange={handleInputChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-sans transition-colors"
                placeholder="Choose a username"
                disabled={isLoading}
              />
            </div>

            <div>
              <label htmlFor="email" className="block font-sans font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <input
                id="email"
                name="email"
                type="email"
                required
                value={formData.email}
                onChange={handleInputChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-sans transition-colors"
                placeholder="Enter your email"
                disabled={isLoading}
              />
            </div>

            <div>
              <label htmlFor="password" className="block font-sans font-medium text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  required
                  value={formData.password}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-sans transition-colors"
                  placeholder="Create a password"
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
                  disabled={isLoading}
                >
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>

              {/* Password Strength Indicator */}
              {formData.password && (
                <div className="mt-2 space-y-1">
                  <div className="flex items-center text-xs">
                    {passwordStrength.length ? (
                      <CheckCircle className="h-3 w-3 text-green-500 mr-2" />
                    ) : (
                      <AlertCircle className="h-3 w-3 text-red-500 mr-2" />
                    )}
                    <span className={passwordStrength.length ? 'text-green-700' : 'text-red-700'}>
                      At least 8 characters
                    </span>
                  </div>
                  <div className="flex items-center text-xs">
                    {passwordStrength.hasUpper && passwordStrength.hasLower ? (
                      <CheckCircle className="h-3 w-3 text-green-500 mr-2" />
                    ) : (
                      <AlertCircle className="h-3 w-3 text-red-500 mr-2" />
                    )}
                    <span className={passwordStrength.hasUpper && passwordStrength.hasLower ? 'text-green-700' : 'text-red-700'}>
                      Uppercase and lowercase letters
                    </span>
                  </div>
                  <div className="flex items-center text-xs">
                    {passwordStrength.hasNumber ? (
                      <CheckCircle className="h-3 w-3 text-green-500 mr-2" />
                    ) : (
                      <AlertCircle className="h-3 w-3 text-red-500 mr-2" />
                    )}
                    <span className={passwordStrength.hasNumber ? 'text-green-700' : 'text-red-700'}>
                      At least one number
                    </span>
                  </div>
                </div>
              )}
            </div>

            <div>
              <label htmlFor="confirmPassword" className="block font-sans font-medium text-gray-700 mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  required
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-sans transition-colors"
                  placeholder="Confirm your password"
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
                  disabled={isLoading}
                >
                  {showConfirmPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-primary-red hover:bg-red-700 disabled:bg-gray-400 text-white font-sans font-bold py-3 px-6 rounded-lg transition-colors flex items-center justify-center"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Creating Account...
                </>
              ) : (
                'Create Account'
              )}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="font-sans text-gray-600">
              Already have an account?{' '}
              <Link
                to="/login"
                className="text-primary-red hover:text-red-700 font-semibold transition-colors"
              >
                Sign in here
              </Link>
            </p>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Register;