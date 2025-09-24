import axios, { AxiosInstance, AxiosResponse } from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle token refresh
api.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem('refresh_token');
        if (refreshToken) {
          // TODO: Implement token refresh endpoint
          // const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
          //   refresh_token: refreshToken
          // });
          // const { access_token } = response.data;
          // localStorage.setItem('access_token', access_token);
          // api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          // return api(originalRequest);
        }
      } catch (refreshError) {
        // Refresh failed, redirect to login
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

// API Response types
export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  status: 'success' | 'error';
}

// User API
export const userApi = {
  register: (userData: { username: string; email: string; password: string }) =>
    api.post<ApiResponse>('/users/register', userData),

  login: (credentials: { username: string; password: string }) =>
    api.post<ApiResponse<{ access_token: string; token_type: string }>>('/users/login', credentials),

  getCurrentUser: () =>
    api.get<ApiResponse>('/users/me'),

  updateProfile: (userData: Partial<{ username: string; email: string; profile_data: any }>) =>
    api.put<ApiResponse>('/users/me', userData),
};

// Pattern API
export const patternApi = {
  analyze: (formData: FormData) =>
    api.post<ApiResponse>('/patterns/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),

  getAnalysis: (analysisId: string) =>
    api.get<ApiResponse>(`/patterns/analysis/${analysisId}`),
};

// Design API (to be implemented)
export const designApi = {
  getAll: (params?: { page?: number; limit?: number; type?: string; region?: string }) =>
    api.get<ApiResponse>('/designs', { params }),

  getById: (id: string) =>
    api.get<ApiResponse>(`/designs/${id}`),

  create: (designData: any) =>
    api.post<ApiResponse>('/designs', designData),

  update: (id: string, designData: any) =>
    api.put<ApiResponse>(`/designs/${id}`, designData),

  delete: (id: string) =>
    api.delete<ApiResponse>(`/designs/${id}`),
};

export default api;