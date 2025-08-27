import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for model operations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Platform Health API
export const healthAPI = {
  checkHealth: () => apiClient.get('/health'),
  getStatus: () => apiClient.get('/device/status'),
};

// Video Stream API
export const videoAPI = {
  getVideoInfo: () => apiClient.get('/video/info'),
  captureScreenshot: () => apiClient.get('/video/screenshot'),
  getStreamUrl: (deviceId, outlet, resolution) => 
    `${API_BASE_URL}/video/stream?device=${deviceId}&outlet=${outlet}&resolution=${resolution}`,
  getImageUrl: (filename) => `${API_BASE_URL}/image/${filename}`,
};

// Device Control API
export const deviceAPI = {
  lockDevice: () => apiClient.post('/device/lock'),
  unlockDevice: () => apiClient.post('/device/unlock'),
  powerOn: () => apiClient.post('/device/power/on'),
  powerOff: () => apiClient.post('/device/power/off'),
  reboot: () => apiClient.post('/device/reboot'),
  pressKey: (keyName, holdTime = 0) => 
    apiClient.post(`/device/key/${keyName}`, { holdTime }),
  pressKeySequence: (keys, delayMs = 500) => 
    apiClient.post('/device/keys', { keys, delay_ms: delayMs }),
  getStatus: () => apiClient.get('/device/status'),
};

// Screen Analysis API
export const analysisAPI = {
  analyzeCurrentScreen: () => apiClient.get('/screen/analyze'),
  validateScreen: (expectedState) => 
    apiClient.post('/screen/validate', { expected_state: expectedState }),
  getAnalysisHistory: () => apiClient.get('/screen/history'),
  analyzeCustomImage: (imageBase64, prompt, model = 'llava:7b') => 
    apiClient.post('/screen/analyze-custom', { 
      image: imageBase64, 
      prompt, 
      model 
    }),
};

// Dataset Management API
export const datasetAPI = {
  createDataset: (name, description, metadata = {}) => 
    apiClient.post('/dataset/create', { 
      name, 
      description, 
      ...metadata 
    }),
  listDatasets: () => apiClient.get('/dataset/list'),
  getDataset: (datasetId) => apiClient.get(`/dataset/${datasetId}`),
  deleteDataset: (datasetId) => apiClient.delete(`/dataset/${datasetId}`),
  
  // Image management
  addImage: (datasetId, imageData, labels) => 
    apiClient.post(`/dataset/${datasetId}/images`, { 
      image: imageData, 
      labels 
    }),
  updateImage: (datasetId, imageId, labels) => 
    apiClient.put(`/dataset/${datasetId}/images/${imageId}`, { labels }),
  deleteImage: (datasetId, imageId) => 
    apiClient.delete(`/dataset/${datasetId}/images/${imageId}`),
  getImage: (datasetId, imageId) => 
    apiClient.get(`/dataset/${datasetId}/images/${imageId}`),
  
  // Dataset operations
  captureToDataset: (datasetName) => 
    apiClient.post(`/dataset/${datasetName}/capture`),
  labelImage: (datasetName, imageName, screenType, notes, labelData = null) =>
    apiClient.post(`/dataset/${datasetName}/label`, { 
      image_name: imageName,
      screen_type: screenType,
      notes,
      label_data: labelData // Extended data for different model types
    }),
  generateTrainingDataset: (datasetName, format = 'yolo', trainSplit = 0.8, augment = true, augmentFactor = 3) =>
    apiClient.post(`/dataset/${datasetName}/generate-training`, { 
      format,
      train_split: trainSplit,
      augment,
      augment_factor: augmentFactor
    }),
  exportDataset: (datasetId, format = 'llava') => 
    apiClient.post(`/dataset/${datasetId}/export`, { format }),
  importDataset: (file, format = 'llava') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('format', format);
    return apiClient.post('/dataset/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  validateDataset: (datasetId) => 
    apiClient.post(`/dataset/${datasetId}/validate`),
};

// Model Training API
export const trainingAPI = {
  listModels: () => apiClient.get('/models/list'),
  getModelInfo: (modelName) => apiClient.get(`/models/${modelName}/info`),
  
  // Training operations
  startTraining: (config) => apiClient.post('/training/start', config),
  stopTraining: (jobId) => apiClient.post(`/training/${jobId}/stop`),
  getTrainingStatus: (jobId) => apiClient.get(`/training/${jobId}/status`),
  getTrainingLogs: (jobId, lines = 100) => 
    apiClient.get(`/training/${jobId}/logs?lines=${lines}`),
  listTrainingJobs: () => apiClient.get('/training/jobs'),
  
  // Model management
  saveModel: (jobId, modelName) => 
    apiClient.post(`/training/${jobId}/save`, { model_name: modelName }),
  loadModel: (modelName) => apiClient.post(`/models/${modelName}/load`),
  deleteModel: (modelName) => apiClient.delete(`/models/${modelName}`),
  
  // Model evaluation
  evaluateModel: (modelName, datasetId) => 
    apiClient.post('/training/evaluate', { 
      model_name: modelName, 
      dataset_id: datasetId 
    }),
};

// Testing API
export const testingAPI = {
  testModel: (modelName, imageBase64, prompt) => 
    apiClient.post('/test/model', { 
      model_name: modelName, 
      image: imageBase64, 
      prompt 
    }),
  benchmarkModel: (modelName, testImages) => 
    apiClient.post('/test/benchmark', { 
      model_name: modelName, 
      test_images: testImages 
    }),
  compareModels: (models, testImages) => 
    apiClient.post('/test/compare', { 
      models, 
      test_images: testImages 
    }),
  
  // Live testing
  startLiveTest: (modelName, streamConfig) => 
    apiClient.post('/test/live/start', { 
      model_name: modelName, 
      stream_config: streamConfig 
    }),
  stopLiveTest: (sessionId) => 
    apiClient.post(`/test/live/${sessionId}/stop`),
  getLiveTestResults: (sessionId) => 
    apiClient.get(`/test/live/${sessionId}/results`),
};

// WebSocket for real-time updates
export class WebSocketManager {
  constructor() {
    this.ws = null;
    this.listeners = new Map();
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.reconnectAttempts = 0;
  }
  
  connect(url = `${API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://')}/ws`) {
    try {
      this.ws = new WebSocket(url);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this.emit('connected');
      };
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit(data.type, data.payload);
        } catch (error) {
          console.error('WebSocket message parse error:', error);
        }
      };
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.emit('disconnected');
        this.scheduleReconnect();
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.scheduleReconnect();
    }
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
  
  send(type, payload) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, payload }));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }
  
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
  }
  
  off(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }
  
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => callback(data));
    }
  }
  
  scheduleReconnect() {
    setTimeout(() => {
      console.log(`Attempting WebSocket reconnection (${this.reconnectAttempts + 1})`);
      this.reconnectAttempts++;
      this.connect();
      
      // Exponential backoff
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 1.5,
        this.maxReconnectDelay
      );
    }, this.reconnectDelay);
  }
}

// Singleton WebSocket instance
export const wsManager = new WebSocketManager();

// Utility functions
export const utils = {
  downloadFile: (blob, filename) => {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  },
  
  formatFileSize: (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },
  
  formatDuration: (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  },
};

export default {
  healthAPI,
  videoAPI,
  deviceAPI,
  analysisAPI,
  datasetAPI,
  trainingAPI,
  testingAPI,
  wsManager,
  utils,
};