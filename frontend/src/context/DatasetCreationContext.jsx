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
    const stateToSave = {
      config,
      isInitialized,
      deviceLocked,
      streamActive,
      currentStep,
      streamUrl,
      capturedImages: capturedImages.map(img => ({
        ...img,
        // Don't save the full thumbnail URL, just the base64 part
        base64: img.thumbnail?.includes('base64,') ? img.thumbnail.split('base64,')[1] : img.base64
      })),
      videoInfo,
      currentDataset,
      datasets,
      savedAt: new Date().toISOString()
    };

    sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(stateToSave));
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