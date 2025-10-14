import { create } from 'zustand';
import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const useAppStore = create((set, get) => ({
  // State
  currentStep: 'upload', // upload, training, converting, complete
  trainingJobId: null,
  conversionJobId: null,
  uploadedTrainingAudio: null,
  uploadedTargetAudio: null,
  trainedModel: null,
  convertedAudioUrl: null,
  jobStatus: null,
  progress: 0,
  error: null,
  userModels: [],
  eventSource: null,

  // Actions
  uploadTrainingAudio: async (file) => {
    try {
      set({ error: null });

      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_URL}/upload/training-audio`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      set({
        uploadedTrainingAudio: response.data,
        trainingJobId: response.data.job_id
      });

      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      set({ error: errorMessage });
      throw error;
    }
  },

  uploadTargetAudio: async (file) => {
    try {
      set({ error: null });

      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_URL}/upload/target-audio`, formData);

      set({
        uploadedTargetAudio: response.data,
        conversionJobId: response.data.job_id
      });

      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      set({ error: errorMessage });
      throw error;
    }
  },

  startTraining: async (modelName) => {
    try {
      const { trainingJobId } = get();

      if (!trainingJobId) {
        throw new Error('No training job ID found');
      }

      const response = await axios.post(`${API_URL}/jobs/train`, {
        job_id: trainingJobId,
        model_name: modelName
      });

      set({ currentStep: 'training' });

      // Start SSE connection for progress
      get().connectToProgressStream(trainingJobId);

      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      set({ error: errorMessage });
      throw error;
    }
  },

  startConversion: async (modelId) => {
    try {
      const { conversionJobId } = get();

      if (!conversionJobId) {
        throw new Error('No conversion job ID found');
      }

      const response = await axios.post(`${API_URL}/jobs/convert`, {
        job_id: conversionJobId,
        model_id: modelId
      });

      set({ currentStep: 'converting' });

      // Start SSE connection
      get().connectToProgressStream(conversionJobId);

      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      set({ error: errorMessage });
      throw error;
    }
  },

  connectToProgressStream: (jobId) => {
    // Close existing connection if any
    const existingEventSource = get().eventSource;
    if (existingEventSource) {
      existingEventSource.close();
    }

    const eventSource = new EventSource(`${API_URL}/stream/progress/${jobId}`);

    set({ eventSource });

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      set({
        jobStatus: data.status,
        progress: data.progress || 0
      });

      // If job complete, fetch output
      if (data.status === 'completed') {
        eventSource.close();
        set({ eventSource: null });

        if (get().currentStep === 'training') {
          get().fetchTrainedModel(jobId);
        } else if (get().currentStep === 'converting') {
          get().fetchConvertedAudio(jobId);
        }
      }

      if (data.status === 'failed') {
        eventSource.close();
        set({
          eventSource: null,
          error: data.error || 'Job failed'
        });
      }
    };

    eventSource.onerror = (err) => {
      console.error('SSE error:', err);
      eventSource.close();
      set({
        eventSource: null,
        error: 'Connection lost. Please refresh the page.'
      });
    };
  },

  fetchTrainedModel: async (jobId) => {
    try {
      const response = await axios.get(`${API_URL}/jobs/status/${jobId}`);
      set({
        trainedModel: response.data,
        currentStep: 'upload-target'
      });
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      set({ error: errorMessage });
    }
  },

  fetchConvertedAudio: async (jobId) => {
    try {
      const downloadUrl = `${API_URL}/download/${jobId}`;
      set({
        convertedAudioUrl: downloadUrl,
        currentStep: 'complete'
      });
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      set({ error: errorMessage });
    }
  },

  loadUserModels: async (userId = 'default_user') => {
    try {
      const response = await axios.get(`${API_URL}/models/${userId}`);
      set({ userModels: response.data.models });
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      set({ error: errorMessage });
    }
  },

  reset: () => {
    // Close SSE connection
    const eventSource = get().eventSource;
    if (eventSource) {
      eventSource.close();
    }

    set({
      currentStep: 'upload',
      trainingJobId: null,
      conversionJobId: null,
      uploadedTrainingAudio: null,
      uploadedTargetAudio: null,
      trainedModel: null,
      convertedAudioUrl: null,
      jobStatus: null,
      progress: 0,
      error: null,
      eventSource: null
    });
  }
}));
