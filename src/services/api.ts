import axios, { AxiosInstance, AxiosResponse } from 'axios';

// API Configuration
const API_BASE_URL = 'http://localhost:8000/api/v1';

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for logging
api.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log('API Response:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('Response error:', error.response?.status, error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API Response types
export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  status: 'success' | 'error';
}

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