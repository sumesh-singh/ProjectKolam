import React, { useEffect } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { userApi } from '../services/api';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requireAuth?: boolean;
  redirectTo?: string;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requireAuth = true,
  redirectTo = '/login'
}) => {
  const { isAuthenticated, user, login, logout, setLoading } = useAuthStore();
  const location = useLocation();

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('access_token');

      if (token && !isAuthenticated) {
        try {
          setLoading(true);
          const response = await userApi.getCurrentUser();
          if (response.data?.data) {
            login(response.data.data, token);
          }
        } catch (error) {
          console.error('Auth check failed:', error);
          logout();
        } finally {
          setLoading(false);
        }
      }
    };

    checkAuth();
  }, [isAuthenticated, login, logout, setLoading]);

  if (requireAuth && !isAuthenticated) {
    // Redirect to login with return url
    return <Navigate to={redirectTo} state={{ from: location }} replace />;
  }

  if (!requireAuth && isAuthenticated) {
    // Redirect authenticated users away from auth pages
    const from = location.state?.from?.pathname || '/dashboard';
    return <Navigate to={from} replace />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;