import React, { createContext, useState, useContext, useEffect } from 'react';

// Create the context
const DatasetCreationContext = createContext();

// Session storage key
const SESSION_STORAGE_KEY = 'datasetCreationSession';

// Provider component
export const DatasetCreationProvider = ({ children }) => {
  // Initialize state from session storage or defaults
  const loadSessionState = () => {
    const savedState = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (savedState) {
      try {
        const parsed = JSON.parse(savedState);
        // Restore thumbnail data properly
        if (parsed.capturedImages) {
          parsed.capturedImages = parsed.capturedImages.map(img => ({
            ...img,
            // Ensure thumbnails are properly formatted
            thumbnail: img.thumbnail || (img.base64 ? `data:image/jpeg;base64,${img.base64}` : null)
          }));
        }
        return parsed;
      } catch (e) {
        console.warn('Failed to parse saved session state');
      }
    }
    return null;
  };

  const savedState = loadSessionState();

  // Configuration state
  const [config, setConfig] = useState(savedState?.config || {
    deviceId: '',
    outlet: '5',
    resolution: '1920x1080',
    macAddress: '',
    keySet: 'SKYQ',
    datasetType: null, // 'vision_llm', 'object_detection', 'image_classification'
    augmentationOptions: {}
  });

  // Platform state
  const [isInitialized, setIsInitialized] = useState(savedState?.isInitialized || false);
  const [deviceLocked, setDeviceLocked] = useState(savedState?.deviceLocked || false);
  const [streamActive, setStreamActive] = useState(savedState?.streamActive || false);
  const [currentStep, setCurrentStep] = useState(savedState?.currentStep || 0);

  // Video stream
  const [streamUrl, setStreamUrl] = useState(savedState?.streamUrl || '');
  const [capturedImages, setCapturedImages] = useState(savedState?.capturedImages || []);
  const [videoInfo, setVideoInfo] = useState(savedState?.videoInfo || null);

  // Dataset management
  const [currentDataset, setCurrentDataset] = useState(savedState?.currentDataset || null);
  const [datasets, setDatasets] = useState(savedState?.datasets || []);

  // Save state to session storage whenever it changes
  useEffect(() => {
    try {
      const stateToSave = {
        config,
        isInitialized,
        deviceLocked,
        streamActive,
        currentStep,
        streamUrl,
        // Only save metadata for images, not the actual base64 data
        capturedImages: capturedImages.map(img => ({
          id: img.id,
          timestamp: img.timestamp,
          path: img.path,
          labels: img.labels,
          datasetType: img.datasetType,
          // Don't save thumbnail - it will be regenerated from backend
        })),
        videoInfo,
        currentDataset,
        datasets,
        savedAt: new Date().toISOString()
      };

      const serialized = JSON.stringify(stateToSave);
      
      // Check if the data is too large (sessionStorage limit is ~5MB)
      if (serialized.length > 4 * 1024 * 1024) { // 4MB limit to be safe
        // If too large, only save essential state without captured images
        const essentialState = {
          config,
          isInitialized,
          deviceLocked,
          streamActive,
          currentStep,
          streamUrl,
          capturedImages: [], // Don't save images if quota exceeded
          videoInfo,
          currentDataset,
          datasets,
          savedAt: new Date().toISOString(),
          warningMessage: 'Captured images not persisted due to storage quota'
        };
        sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(essentialState));
        console.warn('SessionStorage quota nearly exceeded. Saved essential state only.');
      } else {
        sessionStorage.setItem(SESSION_STORAGE_KEY, serialized);
      }
    } catch (error) {
      if (error.name === 'QuotaExceededError') {
        console.warn('SessionStorage quota exceeded. Clearing old data and saving essential state only.');
        try {
          // Clear old data and save only essential state
          const essentialState = {
            config,
            isInitialized,
            deviceLocked,
            streamActive,
            currentStep,
            streamUrl,
            capturedImages: [], // Don't save images
            videoInfo,
            currentDataset,
            datasets,
            savedAt: new Date().toISOString(),
            warningMessage: 'Captured images not persisted due to storage quota'
          };
          sessionStorage.clear();
          sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(essentialState));
        } catch (e) {
          console.error('Failed to save even essential state:', e);
        }
      } else {
        console.error('Failed to save session state:', error);
      }
    }
  }, [
    config,
    isInitialized,
    deviceLocked,
    streamActive,
    currentStep,
    streamUrl,
    capturedImages,
    videoInfo,
    currentDataset,
    datasets
  ]);

  // Function to clear session
  const clearSession = () => {
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
    // Reset all state to defaults
    setConfig({
      deviceId: '',
      outlet: '5',
      resolution: '1920x1080',
      macAddress: '',
      keySet: 'SKYQ',
    });
    setIsInitialized(false);
    setDeviceLocked(false);
    setStreamActive(false);
    setCurrentStep(0);
    setStreamUrl('');
    setCapturedImages([]);
    setVideoInfo(null);
    setCurrentDataset(null);
    // Don't reset datasets as they are global
  };

  const value = {
    // Configuration
    config,
    setConfig,
    
    // Platform state
    isInitialized,
    setIsInitialized,
    deviceLocked,
    setDeviceLocked,
    streamActive,
    setStreamActive,
    currentStep,
    setCurrentStep,
    
    // Video
    streamUrl,
    setStreamUrl,
    capturedImages,
    setCapturedImages,
    videoInfo,
    setVideoInfo,
    
    // Dataset
    currentDataset,
    setCurrentDataset,
    datasets,
    setDatasets,
    
    // Actions
    clearSession,
  };

  return (
    <DatasetCreationContext.Provider value={value}>
      {children}
    </DatasetCreationContext.Provider>
  );
};

// Custom hook to use the context
export const useDatasetCreation = () => {
  const context = useContext(DatasetCreationContext);
  if (!context) {
    throw new Error('useDatasetCreation must be used within a DatasetCreationProvider');
  }
  return context;
};