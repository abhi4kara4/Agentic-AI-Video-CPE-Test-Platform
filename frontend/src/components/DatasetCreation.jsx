import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Chip,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Checkbox,
  FormControlLabel,
  FormGroup,
  Alert,
  LinearProgress,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  PhotoCamera as CameraIcon,
  PowerSettingsNew as PowerIcon,
  Lock as LockIcon,
  LockOpen as UnlockIcon,
  Save as SaveIcon,
  Label as LabelIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Visibility as ViewIcon,
  KeyboardArrowUp,
  KeyboardArrowDown,
  KeyboardArrowLeft,
  KeyboardArrowRight,
  RadioButtonChecked as OKIcon,
  Home as HomeIcon,
  ArrowBack as BackIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  DragHandle as DragHandleIcon,
  RestartAlt as ResetIcon,
  ContentCopy as CopyIcon,
  ContentPaste as PasteIcon,
  SelectAll as SelectAllIcon,
  CheckBoxOutlineBlank as UncheckedIcon,
  CheckBox as CheckedIcon,
  Deselect as DeselectIcon,
} from '@mui/icons-material';
import { videoAPI, deviceAPI, datasetAPI } from '../services/api.jsx';
import { useDatasetCreation } from '../context/DatasetCreationContext.jsx';
import DatasetTypeSelector from './DatasetTypeSelector.jsx';
import ObjectDetectionLabeler from './ObjectDetectionLabeler.jsx';
import ImageClassificationLabeler from './ImageClassificationLabeler.jsx';
import { DATASET_TYPE_INFO, DATASET_TYPES } from '../constants/datasetTypes.js';

// Screen state definitions
const SCREEN_STATES = {
  'home': 'Home Screen',
  'app_rail': 'App Selection Rail',
  'app_loading': 'App Loading',
  'app_content': 'App Content',
  'login': 'Login Screen',
  'error': 'Error Screen',
  'video_playing': 'Video Playing',
  'menu': 'Menu Screen',
  'settings': 'Settings Screen',
  'search': 'Search Screen',
  'other': 'Other'
};

const COMMON_APPS = [
  'Netflix', 'YouTube', 'Prime Video', 'Disney+', 'Hulu',
  'HBO Max', 'Apple TV+', 'Peacock', 'Paramount+', 'ESPN',
  'Sling TV', 'Spotify', 'Pandora', 'Settings'
];

const UI_ELEMENTS = [
  'menu', 'button', 'video_player', 'rail', 'list',
  'keyboard', 'dialog', 'spinner', 'carousel', 'grid'
];

const KEY_SETS = ['SKYQ', 'PR1_T2', 'LC103'];

// Load saved panel sizes from localStorage
const loadPanelSizes = () => {
  const saved = localStorage.getItem('datasetCreation_panelSizes');
  if (saved) {
    try {
      return JSON.parse(saved);
    } catch (e) {
      console.warn('Failed to parse saved panel sizes');
    }
  }
  return {
    configHeight: 600,
    videoHeight: 700,
    datasetHeight: 400,
    imagesHeight: 400,
  };
};

// Save panel sizes to localStorage
const savePanelSizes = (sizes) => {
  localStorage.setItem('datasetCreation_panelSizes', JSON.stringify(sizes));
};

const DatasetCreation = ({ onNotification }) => {
  // Get state from context
  const {
    config,
    setConfig,
    isInitialized,
    setIsInitialized,
    deviceLocked,
    setDeviceLocked,
    streamActive,
    setStreamActive,
    currentStep,
    setCurrentStep,
    streamUrl,
    setStreamUrl,
    capturedImages,
    setCapturedImages,
    videoInfo,
    setVideoInfo,
    currentDataset,
    setCurrentDataset,
    datasets,
    setDatasets,
    clearSession,
  } = useDatasetCreation();

  // Generate thumbnail URL for display and full image URL for labeling
  const getImageUrls = (image) => {
    // Always prefer backend URL if filename exists (more reliable after restarts)
    if (image.filename) {
      const fullImageUrl = videoAPI.getImageUrl(image.filename);
      return {
        thumbnailUrl: fullImageUrl,
        fullImageUrl: fullImageUrl,
      };
    } else if (image.thumbnail) {
      // Fallback to base64 thumbnail if no filename
      return {
        thumbnailUrl: image.thumbnail,
        fullImageUrl: image.thumbnail,
      };
    }
    return { thumbnailUrl: null, fullImageUrl: null };
  };

  // Panel sizing state (still local as it's UI-specific)
  const [panelSizes, setPanelSizes] = useState(loadPanelSizes());
  const [resizing, setResizing] = useState(null);
  const resizeStartPos = useRef(null);
  const resizeStartSize = useRef(null);

  // Video ref
  const videoRef = useRef(null);

  // Local state for UI-only elements
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);
  const [showDatasetTypeSelector, setShowDatasetTypeSelector] = useState(false);
  const [isLockingDevice, setIsLockingDevice] = useState(false);
  const [isGeneratingDataset, setIsGeneratingDataset] = useState(false);

  // Labeling state
  const [selectedImage, setSelectedImage] = useState(null);
  const [labelingDialogOpen, setLabelingDialogOpen] = useState(false);
  const [viewImageDialog, setViewImageDialog] = useState(false);
  const [currentLabels, setCurrentLabels] = useState({
    screen_type: '',
    app_name: '',
    ui_elements: [],
    visible_text: '',
    anomalies: [],
    navigation: {
      focused_element: '',
      can_navigate: { up: false, down: false, left: false, right: false }
    },
    notes: ''
  });

  // Copy/Paste and Multi-selection state
  const [copiedAnnotations, setCopiedAnnotations] = useState(null);
  const [selectedImages, setSelectedImages] = useState(new Set());
  const [isMultiSelectMode, setIsMultiSelectMode] = useState(false);
  const [showBulkPasteDialog, setShowBulkPasteDialog] = useState(false);

  // Load datasets on component mount
  useEffect(() => {
    loadDatasets();
    
    // If we have an active stream from previous session, make sure it's displayed
    if (streamActive && streamUrl) {
      console.log('Restoring active stream from previous session');
      // Re-fetch video info to ensure stream is still active
      setTimeout(() => {
        fetchVideoInfo();
      }, 500);
    }
    
    // Check if there was a storage quota warning
    const savedState = JSON.parse(sessionStorage.getItem('datasetCreationSession') || '{}');
    if (savedState.warningMessage) {
      onNotification({
        type: 'warning',
        title: 'Storage Quota Exceeded',
        message: 'Your captured images may not persist between page refreshes due to browser storage limits. Consider exporting your dataset regularly.'
      });
    }
    
    // Validate and refresh captured images from previous session
    if (capturedImages.length > 0) {
      validateAndRefreshImages();
    }
  }, []);

  // Validate and refresh images after Docker restart
  const validateAndRefreshImages = async () => {
    console.log('Validating and refreshing captured images from previous session');
    
    try {
      // Test if backend is accessible by checking first image
      if (capturedImages.length > 0 && capturedImages[0].filename) {
        const testUrl = videoAPI.getImageUrl(capturedImages[0].filename);
        
        // Try to fetch the first image to see if backend is working
        try {
          const response = await fetch(testUrl, { method: 'HEAD' });
          if (!response.ok) {
            throw new Error('Backend images not accessible');
          }
        } catch (error) {
          console.warn('Backend images not accessible after restart, clearing thumbnails');
          
          // Clear thumbnails but keep image references
          const updatedImages = capturedImages.map(img => ({
            ...img,
            thumbnail: null, // Clear invalid base64 thumbnails
          }));
          
          setCapturedImages(updatedImages);
          
          onNotification({
            type: 'info',
            title: 'Session Restored',
            message: 'Previous session restored. Images are available in backend but thumbnails need to be reloaded.'
          });
        }
      }
    } catch (error) {
      console.error('Error validating images:', error);
    }
  };

  // Reload images from current dataset
  const reloadImagesFromDataset = async () => {
    if (!currentDataset) {
      onNotification({
        type: 'warning',
        title: 'No Dataset Selected',
        message: 'Please select a dataset to reload images from'
      });
      return;
    }

    try {
      const response = await datasetAPI.getDataset(currentDataset.id);
      const datasetInfo = response.data;
      
      if (datasetInfo.images && datasetInfo.images.length > 0) {
        // Map backend images to our frontend format
        const reloadedImages = datasetInfo.images.map((img, index) => ({
          id: img.id || Date.now() + index,
          path: img.path || img.filename,
          filename: img.filename,
          timestamp: img.timestamp || img.created_at,
          labels: img.labels || null,
          thumbnail: null, // Force reload from backend
        }));
        
        setCapturedImages(reloadedImages);
        
        onNotification({
          type: 'success',
          title: 'Images Reloaded',
          message: `Reloaded ${reloadedImages.length} images from dataset "${currentDataset.name}"`
        });
      } else {
        onNotification({
          type: 'info',
          title: 'No Images Found',
          message: 'No images found in the current dataset'
        });
      }
    } catch (error) {
      console.error('Failed to reload images:', error);
      onNotification({
        type: 'error',
        title: 'Reload Failed',
        message: 'Failed to reload images from dataset'
      });
    }
  };

  const loadDatasets = async () => {
    try {
      const response = await datasetAPI.listDatasets();
      const loadedDatasets = response.data?.datasets || [];
      setDatasets(loadedDatasets);
      
      // Clear currentDataset if it's not in the loaded datasets
      if (currentDataset && !loadedDatasets.find(d => d.id === currentDataset.id)) {
        setCurrentDataset(null);
      }
    } catch (error) {
      console.error('Failed to load datasets:', error);
      setDatasets([]);
      setCurrentDataset(null);
    }
  };

  const fetchVideoInfo = async () => {
    try {
      const response = await videoAPI.getVideoInfo();
      setVideoInfo(response.data);
      console.log('Video info:', response.data);
    } catch (error) {
      console.error('Failed to fetch video info:', error);
    }
  };

  // Resize handlers
  const handleResizeStart = (panelType, event) => {
    setResizing(panelType);
    resizeStartPos.current = event.clientY; // Always use Y for height resizing
    resizeStartSize.current = panelSizes[panelType];
    document.addEventListener('mousemove', handleResizeMove);
    document.addEventListener('mouseup', handleResizeEnd);
    event.preventDefault();
  };

  const handleResizeMove = (event) => {
    if (!resizing || !resizeStartPos.current) return;
    
    const delta = event.clientY - resizeStartPos.current;
    const newSize = Math.max(200, resizeStartSize.current + delta); // Minimum 200px
    
    setPanelSizes(prev => ({
      ...prev,
      [resizing]: newSize
    }));
  };

  const handleResizeEnd = () => {
    if (resizing) {
      // Save to localStorage
      const newSizes = { ...panelSizes };
      savePanelSizes(newSizes);
    }
    
    setResizing(null);
    resizeStartPos.current = null;
    resizeStartSize.current = null;
    document.removeEventListener('mousemove', handleResizeMove);
    document.removeEventListener('mouseup', handleResizeEnd);
  };

  // Reset panel sizes
  const resetPanelSizes = () => {
    const defaultSizes = {
      configHeight: 600,
      videoHeight: 700,
      datasetHeight: 400,
      imagesHeight: 400,
    };
    setPanelSizes(defaultSizes);
    savePanelSizes(defaultSizes);
  };

  // Drag handle component
  const DragHandle = ({ panelType, orientation = 'horizontal' }) => (
    <Box
      onMouseDown={(e) => handleResizeStart(panelType, e)}
      sx={{
        cursor: 'ns-resize', // Always vertical resize for height changes
        backgroundColor: 'grey.300',
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        '&:hover': {
          backgroundColor: 'primary.main',
          opacity: 0.8,
        },
        '&:active': {
          backgroundColor: 'primary.dark',
        },
        height: '12px',
        width: '100%',
        userSelect: 'none',
        zIndex: 10,
        borderRadius: '4px',
        transition: 'background-color 0.2s ease',
      }}
    >
      <DragHandleIcon 
        sx={{ 
          fontSize: 18, 
          color: 'text.secondary',
          transform: 'rotate(90deg)'
        }} 
      />
    </Box>
  );

  const handleInitializePlatform = async () => {
    if (!config.deviceId || !config.macAddress) {
      onNotification({
        type: 'error',
        title: 'Configuration Error',
        message: 'Please provide Device ID and MAC Address'
      });
      return;
    }

    // Check if dataset type is selected
    if (!config.datasetType) {
      setShowDatasetTypeSelector(true);
      return;
    }

    const initializeWithDatasetType = async () => {
    try {
      // Start video stream with cache-busting parameter
      const streamUrl = videoAPI.getStreamUrl(config.deviceId, config.outlet, config.resolution) + `&t=${Date.now()}`;
      setStreamUrl(streamUrl);
      setStreamActive(true);
      
      setIsInitialized(true);
      setCurrentStep(1);
      
      // Fetch video info to confirm resolution
      setTimeout(() => {
        fetchVideoInfo();
      }, 1000);
      
      onNotification({
        type: 'success',
        title: 'Platform Initialized',
        message: 'Video stream started successfully'
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Initialization Failed',
        message: error.message
      });
    }
    };

    // Proceed with initialization after dataset type is confirmed
    initializeWithDatasetType();
  };

  const handleDatasetTypeSelection = (typeConfig) => {
    setConfig(prev => ({
      ...prev,
      datasetType: typeConfig.datasetType,
      augmentationOptions: typeConfig.augmentationOptions
    }));
    setShowDatasetTypeSelector(false);
    
    onNotification({
      type: 'success',
      title: 'Dataset Type Selected',
      message: `Configured for ${DATASET_TYPE_INFO[typeConfig.datasetType].name} training`
    });
    
    // Now proceed with initialization directly (don't call handleInitializePlatform again)
    const initializeWithDatasetType = async () => {
      try {
        // Start video stream with cache-busting parameter
        const streamUrl = videoAPI.getStreamUrl(config.deviceId, config.outlet, config.resolution) + `&t=${Date.now()}`;
        setStreamUrl(streamUrl);
        setStreamActive(true);
        
        setIsInitialized(true);
        setCurrentStep(1);
        
        // Fetch video info to confirm resolution
        setTimeout(() => {
          fetchVideoInfo();
        }, 1000);
        
        onNotification({
          type: 'success',
          title: 'Platform Initialized',
          message: 'Video stream started successfully'
        });
      } catch (error) {
        onNotification({
          type: 'error',
          title: 'Initialization Failed',
          message: error.message
        });
      }
    };
    
    setTimeout(() => {
      initializeWithDatasetType();
    }, 500);
  };

  const handleRestreamVideo = () => {
    // Stop current stream
    setStreamActive(false);
    setStreamUrl('');
    
    // Restart with new settings after a brief delay
    setTimeout(() => {
      const newStreamUrl = videoAPI.getStreamUrl(config.deviceId, config.outlet, config.resolution) + `&t=${Date.now()}`;
      setStreamUrl(newStreamUrl);
      setStreamActive(true);
      
      // Fetch video info to confirm new resolution
      setTimeout(() => {
        fetchVideoInfo();
      }, 1000);
      
      onNotification({
        type: 'info',
        title: 'Stream Updated',
        message: `Video stream updated to ${config.resolution}`
      });
    }, 100);
  };

  const handleDeviceLock = async () => {
    setIsLockingDevice(true);
    try {
      if (deviceLocked) {
        await deviceAPI.unlockDevice();
        setDeviceLocked(false);
        onNotification({
          type: 'info',
          title: 'Device Unlocked',
          message: 'Device is now available for others'
        });
      } else {
        // Show loading state
        onNotification({
          type: 'info',
          title: 'Locking Device',
          message: 'Attempting to lock device...'
        });
        
        await deviceAPI.lockDevice();
        setDeviceLocked(true);
        setCurrentStep(2);
        onNotification({
          type: 'success',
          title: 'Device Locked',
          message: 'Device is now under your control'
        });
      }
    } catch (error) {
      let errorMessage = 'Unknown error occurred';
      
      if (error.response?.status === 409) {
        errorMessage = error.response?.data?.detail || 'Device is currently in use by another session. Please try again in a few moments.';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      onNotification({
        type: 'error',
        title: 'Device Lock Failed',
        message: errorMessage
      });
    } finally {
      setIsLockingDevice(false);
    }
  };

  const handleDeviceControl = async (action, key = null) => {
    try {
      let response;
      switch (action) {
        case 'power_on':
          response = await deviceAPI.powerOn();
          break;
        case 'power_off':
          response = await deviceAPI.powerOff();
          break;
        case 'key':
          // Extract just the key name, remove the key set prefix
          const cleanKeyName = key.includes(':') ? key.split(':')[1] : key;
          response = await deviceAPI.pressKey(cleanKeyName);
          break;
        default:
          return;
      }
      
      onNotification({
        type: 'success',
        title: 'Command Sent',
        message: `${action.replace('_', ' ').toUpperCase()} command executed`
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Command Failed',
        message: error.response?.data?.detail || error.message
      });
    }
  };

  const captureScreenshot = async () => {
    // Check if dataset type is selected
    if (!config.datasetType) {
      onNotification({
        type: 'warning',
        title: 'Dataset Type Required',
        message: 'Please select a dataset type before capturing screenshots. Initialize the platform first.'
      });
      return;
    }

    // Check if dataset is selected
    if (!currentDataset) {
      onNotification({
        type: 'warning',
        title: 'Dataset Required',
        message: 'Please select or create a dataset before capturing screenshots.'
      });
      return;
    }

    try {
      const response = await datasetAPI.captureToDataset(currentDataset.name);
      
      // Use backend screenshot data if available
      let thumbnailData = null;
      if (response.data && response.data.base64_image) {
        thumbnailData = `data:image/jpeg;base64,${response.data.base64_image}`;
      } else {
        // Fallback to getting frame from video element
        const frame = await getVideoFrame();
        if (frame) {
          thumbnailData = `data:image/jpeg;base64,${frame}`;
        }
      }
      
      const newImage = {
        id: Date.now(),
        path: response.data.screenshot_path || `screenshot_${Date.now()}.jpg`,
        filename: response.data.filename || `screenshot_${Date.now()}.jpg`,
        timestamp: response.data.timestamp || new Date().toISOString(),
        labels: null,
        thumbnail: thumbnailData
      };
      
      setCapturedImages(prev => [newImage, ...prev]);
      setCurrentStep(Math.max(currentStep, 3));
      
      onNotification({
        type: 'success',
        title: 'Screenshot Captured',
        message: 'Image saved successfully'
      });
      
      // Refresh dataset list to update image count
      await loadDatasets();
      
    } catch (error) {
      onNotification({
        type: 'info',
        title: 'Screenshot Saved',
        message: 'Screenshot saved to backend (thumbnail may not be available due to CORS)'
      });
      
      // Still refresh dataset list in case image was saved
      await loadDatasets();
    }
  };

  const getVideoFrame = async () => {
    return new Promise((resolve) => {
      const img = videoRef.current;
      if (!img) {
        resolve('');
        return;
      }

      try {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = img.naturalWidth || 704;
        canvas.height = img.naturalHeight || 480;
        
        // Create a new image with crossorigin to avoid CORS issues
        const crossOriginImg = new Image();
        crossOriginImg.crossOrigin = 'anonymous';
        
        crossOriginImg.onload = () => {
          try {
            ctx.drawImage(crossOriginImg, 0, 0, canvas.width, canvas.height);
            const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
            resolve(base64);
          } catch (canvasError) {
            console.warn('Canvas export failed due to CORS:', canvasError);
            resolve('');
          }
        };
        
        crossOriginImg.onerror = () => {
          console.warn('Cross-origin image load failed');
          resolve('');
        };
        
        crossOriginImg.src = img.src;
      } catch (error) {
        console.error('getVideoFrame error:', error);
        resolve('');
      }
    });
  };

  const handleDeleteImage = (imageId) => {
    setCapturedImages(prev => prev.filter(img => img.id !== imageId));
    onNotification({
      type: 'info',
      title: 'Image Deleted',
      message: 'Screenshot removed from dataset'
    });
  };

  const openLabelingDialog = (image) => {
    setSelectedImage(image);
    
    // Initialize labels based on dataset type
    let defaultLabels = {};
    
    switch (config.datasetType) {
      case DATASET_TYPES.OBJECT_DETECTION:
        defaultLabels = {
          boundingBoxes: [],
          notes: ''
        };
        break;
      case DATASET_TYPES.IMAGE_CLASSIFICATION:
        defaultLabels = {
          className: '',
          confidence: 100,
          notes: ''
        };
        break;
      case DATASET_TYPES.VISION_LLM:
      default:
        defaultLabels = {
          screen_type: '',
          app_name: '',
          ui_elements: [],
          visible_text: '',
          anomalies: [],
          navigation: {
            focused_element: '',
            can_navigate: { up: false, down: false, left: false, right: false }
          },
          notes: ''
        };
        break;
    }
    
    setCurrentLabels(image.labels || defaultLabels);
    setLabelingDialogOpen(true);
  };

  const handleViewImage = (image) => {
    setSelectedImage(image);
    setViewImageDialog(true);
  };

  const saveLabeledImage = async () => {
    if (!selectedImage) {
      onNotification({
        type: 'error',
        title: 'Save Error',
        message: 'No image selected for labeling'
      });
      return;
    }

    // If no dataset is selected, prompt user to create one
    if (!currentDataset) {
      const shouldCreateDataset = window.confirm(
        'No dataset selected. Would you like to create a new dataset to save these labels?'
      );
      
      if (shouldCreateDataset) {
        const datasetName = prompt('Enter dataset name:');
        if (!datasetName) {
          return;
        }
        
        try {
          const datasetTypeInfo = DATASET_TYPE_INFO[config.datasetType];
          const description = `${datasetTypeInfo.name} dataset for ${datasetTypeInfo.description.toLowerCase()}`;
          
          const response = await datasetAPI.createDataset(datasetName, description, {
            datasetType: config.datasetType,
            augmentationOptions: config.augmentationOptions,
            supportedFormats: datasetTypeInfo.supportedFormats
          });
          setCurrentDataset(response.data);
          
          onNotification({
            type: 'success',
            title: 'Dataset Created',
            message: `Dataset "${datasetName}" created successfully. Now saving labels...`
          });
        } catch (error) {
          let errorMessage = 'Unknown error occurred';
          
          if (error.response?.data?.detail) {
            if (Array.isArray(error.response.data.detail)) {
              errorMessage = error.response.data.detail.map(err => err.msg || err).join(', ');
            } else if (typeof error.response.data.detail === 'string') {
              errorMessage = error.response.data.detail;
            } else {
              errorMessage = JSON.stringify(error.response.data.detail);
            }
          } else if (error.message) {
            errorMessage = error.message;
          }
          
          onNotification({
            type: 'error',
            title: 'Dataset Creation Failed',
            message: errorMessage
          });
          return;
        }
      } else {
        onNotification({
          type: 'info',
          title: 'Save Cancelled',
          message: 'Please select or create a dataset to save labels'
        });
        return;
      }
    }
    
    // Validate labels based on dataset type
    const isValidLabels = () => {
      switch (config.datasetType) {
        case DATASET_TYPES.OBJECT_DETECTION:
          return currentLabels.boundingBoxes && currentLabels.boundingBoxes.length > 0;
        case DATASET_TYPES.IMAGE_CLASSIFICATION:
          return currentLabels.className && currentLabels.className.length > 0;
        case DATASET_TYPES.VISION_LLM:
          return currentLabels.screen_type && currentLabels.screen_type.length > 0;
        default:
          return true;
      }
    };

    if (!isValidLabels()) {
      onNotification({
        type: 'warning',
        title: 'Incomplete Labels',
        message: 'Please complete the required labeling fields'
      });
      return;
    }
    
    try {
      // Prepare labels in the format expected by the backend
      const labelData = {
        datasetType: config.datasetType,
        labels: currentLabels,
        augmentationOptions: config.augmentationOptions
      };
      
      // Determine screen_type based on dataset type
      let screenType = '';
      if (config.datasetType === DATASET_TYPES.IMAGE_CLASSIFICATION) {
        screenType = currentLabels.className || '';
      } else if (config.datasetType === DATASET_TYPES.VISION_LLM) {
        screenType = currentLabels.screen_type || '';
      } else {
        // For object detection, use a default or generic screen type
        screenType = 'other';
      }
      
      // Save to backend with enhanced data
      const response = await datasetAPI.labelImage(
        currentDataset.name,
        selectedImage.path.split('/').pop(),
        screenType,
        currentLabels.notes || '',
        labelData
      );
      
      // Update local state
      const updatedImages = capturedImages.map(img => 
        img.id === selectedImage.id 
          ? { ...img, labels: currentLabels, datasetType: config.datasetType }
          : img
      );
      setCapturedImages(updatedImages);
      setCurrentStep(Math.max(currentStep, 4));
      
      // Close the dialog after successful save
      setLabelingDialogOpen(false);
      setSelectedImage(null);
      setCurrentLabels({
        screen_type: '',
        app_name: '',
        ui_elements: [],
        visible_text: '',
        anomalies: [],
        navigation: {
          focused_element: '',
          can_navigate: { up: false, down: false, left: false, right: false }
        },
        notes: ''
      });
      
      const labelCount = config.datasetType === DATASET_TYPES.OBJECT_DETECTION 
        ? currentLabels.boundingBoxes?.length || 0
        : 1;
      
      onNotification({
        type: 'success',
        title: 'Labels Saved',
        message: `Image labeled successfully with ${labelCount} ${config.datasetType.replace('_', ' ')} label${labelCount !== 1 ? 's' : ''}`
      });
    } catch (error) {
      let errorMessage = 'Unknown error occurred';
      
      if (error.response?.data?.detail) {
        // Handle FastAPI validation errors
        if (Array.isArray(error.response.data.detail)) {
          errorMessage = error.response.data.detail.map(err => err.msg || err).join(', ');
        } else if (typeof error.response.data.detail === 'string') {
          errorMessage = error.response.data.detail;
        } else {
          errorMessage = JSON.stringify(error.response.data.detail);
        }
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      onNotification({
        type: 'error',
        title: 'Labeling Failed',
        message: errorMessage
      });
    }
  };

  const createDataset = async () => {
    if (!config.datasetType) {
      onNotification({
        type: 'warning',
        title: 'Dataset Type Required',
        message: 'Please select a dataset type first by initializing the platform'
      });
      return;
    }
    
    setIsCreatingDataset(true);
    const datasetName = prompt('Enter dataset name:');
    
    if (!datasetName) {
      setIsCreatingDataset(false);
      return;
    }
    
    try {
      const datasetTypeInfo = DATASET_TYPE_INFO[config.datasetType];
      const description = `${datasetTypeInfo.name} dataset for ${datasetTypeInfo.description.toLowerCase()}`;
      
      const response = await datasetAPI.createDataset(datasetName, description, {
        datasetType: config.datasetType,
        augmentationOptions: config.augmentationOptions,
        supportedFormats: datasetTypeInfo.supportedFormats
      });
      setCurrentDataset(response.data);
      await loadDatasets();
      
      onNotification({
        type: 'success',
        title: 'Dataset Created',
        message: `${datasetTypeInfo.icon} ${datasetTypeInfo.name} dataset "${datasetName}" created successfully`
      });
    } catch (error) {
      let errorMessage = 'Unknown error occurred';
      
      if (error.response?.data?.detail) {
        if (Array.isArray(error.response.data.detail)) {
          errorMessage = error.response.data.detail.map(err => err.msg || err).join(', ');
        } else if (typeof error.response.data.detail === 'string') {
          errorMessage = error.response.data.detail;
        } else {
          errorMessage = JSON.stringify(error.response.data.detail);
        }
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      onNotification({
        type: 'error',
        title: 'Dataset Creation Failed',
        message: errorMessage
      });
    } finally {
      setIsCreatingDataset(false);
    }
  };

  const generateTrainingDataset = async () => {
    if (!currentDataset) {
      onNotification({
        type: 'warning',
        title: 'No Dataset Selected',
        message: 'Please select a dataset first'
      });
      return;
    }

    const labeledCount = capturedImages.filter(img => img.labels).length;
    if (labeledCount === 0) {
      onNotification({
        type: 'warning',
        title: 'No Labeled Images',
        message: 'Please label some images before generating training dataset'
      });
      return;
    }

    setIsGeneratingDataset(true);
    try {
      // Determine format based on dataset type
      let format = 'yolo';
      if (config.datasetType === DATASET_TYPES.IMAGE_CLASSIFICATION) {
        format = 'folder_structure';
      } else if (config.datasetType === DATASET_TYPES.VISION_LLM) {
        format = 'llava';
      }

      onNotification({
        type: 'info',
        title: 'Generating Training Dataset',
        message: `Creating ${format.toUpperCase()} format dataset with augmentation...`
      });

      const response = await datasetAPI.generateTrainingDataset(
        currentDataset.name,
        format,
        0.8, // 80% train, 20% validation
        true, // Enable augmentation
        3 // 3x augmentation factor
      );

      onNotification({
        type: 'success',
        title: 'Training Dataset Generated',
        message: `${response.data.zip_file} created with ${response.data.train_images} training and ${response.data.val_images} validation images`
      });

      // Mark the Generate Dataset step as completed
      setCurrentStep(5);

    } catch (error) {
      console.error('Failed to generate training dataset:', error);
      onNotification({
        type: 'error',
        title: 'Generation Failed',
        message: error.response?.data?.detail || 'Failed to generate training dataset'
      });
    } finally {
      setIsGeneratingDataset(false);
    }
  };

  // Copy/Paste and Multi-selection functions
  const handleCopyAnnotations = (annotations) => {
    setCopiedAnnotations(annotations);
    onNotification({
      type: 'success',
      title: 'Annotations Copied',
      message: `Copied ${annotations.boundingBoxes?.length || 0} bounding boxes from ${annotations.imageInfo?.imageName || 'image'}`
    });
  };

  const handleImageSelect = (imageId, isSelected) => {
    setSelectedImages(prev => {
      const newSet = new Set(prev);
      if (isSelected) {
        newSet.add(imageId);
      } else {
        newSet.delete(imageId);
      }
      return newSet;
    });
  };

  const handleSelectAllImages = () => {
    if (selectedImages.size === capturedImages.length) {
      setSelectedImages(new Set()); // Deselect all
    } else {
      setSelectedImages(new Set(capturedImages.map(img => img.id))); // Select all
    }
  };

  const handleBulkPasteAnnotations = async () => {
    if (!copiedAnnotations || selectedImages.size === 0) return;

    try {
      setIsGeneratingDataset(true); // Reuse existing loading state
      
      const imagesToUpdate = capturedImages.filter(img => selectedImages.has(img.id));
      let successCount = 0;
      let failCount = 0;

      for (const image of imagesToUpdate) {
        try {
          // Create label data with pasted annotations
          const labelData = {
            boundingBoxes: copiedAnnotations.boundingBoxes.map(box => ({
              ...box,
              id: Date.now() + Math.random() // New unique ID
            }))
          };

          // Save to backend
          await datasetAPI.labelImage(
            currentDataset.name,
            image.path.split('/').pop(),
            'other', // Default screen type for bulk operations
            `Bulk pasted from ${copiedAnnotations.imageInfo?.imageName || 'source image'}`,
            labelData
          );

          successCount++;
        } catch (error) {
          console.error(`Failed to paste annotations to image ${image.filename}:`, error);
          failCount++;
        }
      }

      // Update local state for successful updates
      const updatedImages = capturedImages.map(img => {
        if (selectedImages.has(img.id)) {
          return {
            ...img,
            labels: {
              ...currentLabels,
              boundingBoxes: copiedAnnotations.boundingBoxes.map(box => ({
                ...box,
                id: Date.now() + Math.random()
              })),
              notes: `Bulk pasted from ${copiedAnnotations.imageInfo?.imageName || 'source image'}`
            },
            datasetType: config.datasetType
          };
        }
        return img;
      });
      setCapturedImages(updatedImages);

      // Clear selections
      setSelectedImages(new Set());
      setIsMultiSelectMode(false);
      setShowBulkPasteDialog(false);

      const totalProcessed = successCount + failCount;
      onNotification({
        type: successCount === totalProcessed ? 'success' : 'warning',
        title: 'Bulk Paste Complete',
        message: `Successfully pasted annotations to ${successCount} images${failCount > 0 ? `, ${failCount} failed` : ''}`
      });

    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Bulk Paste Failed',
        message: 'Failed to paste annotations to selected images'
      });
    } finally {
      setIsGeneratingDataset(false);
    }
  };

  const steps = [
    'Configure Platform',
    'Start Video Stream',
    'Lock Device',
    'Capture Screenshots',
    'Label Images',
    'Generate Dataset'
  ];

  return (
    <Box sx={{ p: 3, maxWidth: '100%', height: 'calc(100vh - 150px)', overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h4" fontWeight="bold">
              Dataset Creation
            </Typography>
            {(isInitialized || capturedImages.length > 0) && (
              <Chip 
                label="Session Active" 
                size="small" 
                color="success" 
                variant="outlined"
                icon={<SaveIcon sx={{ fontSize: 16 }} />}
                sx={{ 
                  animation: 'pulse 2s infinite',
                  '& .MuiChip-label': { fontSize: '0.75rem' }
                }}
              />
            )}
          </Box>
          <Typography variant="body1" color="text.secondary">
            Create labeled datasets for TV/STB vision model training
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            size="small"
            startIcon={<ResetIcon />}
            onClick={resetPanelSizes}
            title="Reset panel sizes to default"
          >
            Reset Layout
          </Button>
          {isInitialized && (
            <Button
              variant="outlined"
              size="small"
              color="error"
              onClick={() => {
                if (confirm('This will clear your current session including all captured images. Are you sure?')) {
                  clearSession();
                  onNotification({
                    type: 'info',
                    title: 'Session Cleared',
                    message: 'Dataset creation session has been reset'
                  });
                }
              }}
              title="Clear current session"
            >
              New Session
            </Button>
          )}
        </Box>
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '820px' }}>
        {/* Top Row - Configuration and Video Stream */}
        <Box sx={{ display: 'flex', gap: 2, mb: 1 }}>
          <Box sx={{ flex: '0 0 400px', minWidth: '300px', maxWidth: '500px' }}>
          <Card sx={{ height: `${panelSizes.configHeight}px`, display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1, overflow: 'auto' }}>
              <Typography variant="h6" gutterBottom>
                Platform Configuration
              </Typography>

              {/* Dataset Type Display */}
              {config.datasetType && (
                <Alert 
                  severity="info" 
                  sx={{ mb: 2 }}
                  icon={<Chip label={DATASET_TYPE_INFO[config.datasetType].icon} size="small" />}
                >
                  <Typography variant="body2">
                    <strong>Dataset Type:</strong> {DATASET_TYPE_INFO[config.datasetType].name}
                  </Typography>
                  <Typography variant="caption" display="block">
                    {DATASET_TYPE_INFO[config.datasetType].description}
                  </Typography>
                </Alert>
              )}
              
              {/* Configuration Form */}
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  label="Video Device ID"
                  value={config.deviceId}
                  onChange={(e) => setConfig(prev => ({ ...prev, deviceId: e.target.value }))}
                  margin="normal"
                  size="small"
                />
                <TextField
                  fullWidth
                  label="Video Outlet"
                  value={config.outlet}
                  onChange={(e) => setConfig(prev => ({ ...prev, outlet: e.target.value }))}
                  margin="normal"
                  size="small"
                />
                <FormControl fullWidth margin="normal" size="small">
                  <InputLabel>Resolution</InputLabel>
                  <Select
                    value={config.resolution}
                    onChange={(e) => setConfig(prev => ({ ...prev, resolution: e.target.value }))}
                  >
                    <MenuItem value="704x480">704x480 (SD)</MenuItem>
                    <MenuItem value="1280x720">1280x720 (HD)</MenuItem>
                    <MenuItem value="1920x1080">1920x1080 (Full HD)</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  fullWidth
                  label="Device MAC Address"
                  value={config.macAddress}
                  onChange={(e) => setConfig(prev => ({ ...prev, macAddress: e.target.value }))}
                  margin="normal"
                  size="small"
                  placeholder="XX:XX:XX:XX:XX:XX"
                />
                <FormControl fullWidth margin="normal" size="small">
                  <InputLabel>Remote Key Set</InputLabel>
                  <Select
                    value={config.keySet}
                    onChange={(e) => setConfig(prev => ({ ...prev, keySet: e.target.value }))}
                  >
                    {KEY_SETS.map((keySet) => (
                      <MenuItem key={keySet} value={keySet}>
                        {keySet}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                  <Button
                    variant="contained"
                    fullWidth
                    onClick={handleInitializePlatform}
                    disabled={isInitialized && streamActive}
                  >
                    Initialize Platform
                  </Button>
                  {isInitialized && (
                    <IconButton
                      color="primary"
                      onClick={handleRestreamVideo}
                      title="Restream with new settings"
                    >
                      <RefreshIcon />
                    </IconButton>
                  )}
                </Box>
              </Box>

              {/* Device Control */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Device Control
                </Typography>
                
                <Button
                  variant={deviceLocked ? "contained" : "outlined"}
                  fullWidth
                  startIcon={isLockingDevice ? <CircularProgress size={16} color="inherit" /> : (deviceLocked ? <LockIcon /> : <UnlockIcon />)}
                  onClick={handleDeviceLock}
                  disabled={!isInitialized || isLockingDevice}
                  color={deviceLocked ? "primary" : "inherit"}
                  sx={{ mb: 2 }}
                >
                  {isLockingDevice ? 'Processing...' : (deviceLocked ? 'Device Locked' : 'Lock Device')}
                </Button>

                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="outlined"
                    startIcon={<PowerIcon />}
                    onClick={() => handleDeviceControl('power_on')}
                    disabled={!deviceLocked}
                    size="small"
                  >
                    Power On
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<PowerIcon />}
                    onClick={() => handleDeviceControl('power_off')}
                    disabled={!deviceLocked}
                    size="small"
                  >
                    Power Off
                  </Button>
                </Box>

                {/* Navigation Keys */}
                <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                  Navigation Keys
                </Typography>
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1 }}>
                  <div />
                  <IconButton
                    onClick={() => handleDeviceControl('key', `${config.keySet}:UP`)}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <KeyboardArrowUp />
                  </IconButton>
                  <div />
                  
                  <IconButton
                    onClick={() => handleDeviceControl('key', `${config.keySet}:LEFT`)}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <KeyboardArrowLeft />
                  </IconButton>
                  <IconButton
                    onClick={() => handleDeviceControl('key', `${config.keySet}:OK`)}
                    disabled={!deviceLocked}
                    color="primary"
                    sx={{ bgcolor: 'primary.main', color: 'white', '&:hover': { bgcolor: 'primary.dark' } }}
                  >
                    <OKIcon />
                  </IconButton>
                  <IconButton
                    onClick={() => handleDeviceControl('key', `${config.keySet}:RIGHT`)}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <KeyboardArrowRight />
                  </IconButton>
                  
                  <div />
                  <IconButton
                    onClick={() => handleDeviceControl('key', `${config.keySet}:DOWN`)}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <KeyboardArrowDown />
                  </IconButton>
                  <div />
                </Box>

                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                  <IconButton
                    onClick={() => handleDeviceControl('key', `${config.keySet}:HOME`)}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <HomeIcon />
                  </IconButton>
                  <IconButton
                    onClick={() => handleDeviceControl('key', `${config.keySet}:LAST`)}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <BackIcon />
                  </IconButton>
                </Box>
              </Box>
            </CardContent>
          </Card>
          </Box>

          {/* This handle doesn't make sense here - remove it */}
          
          {/* Video Stream */}
          <Box sx={{ flexGrow: 1, minWidth: '400px' }}>
          <Card sx={{ height: `${panelSizes.videoHeight}px`, display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Live Video Stream
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  {videoInfo && videoInfo.current_resolution && (
                    <Chip
                      label={videoInfo.current_resolution}
                      color="primary"
                      size="small"
                      variant="outlined"
                    />
                  )}
                  {streamActive && (
                    <Chip
                      label="LIVE"
                      color="success"
                      size="small"
                      sx={{ animation: 'pulse 2s infinite' }}
                    />
                  )}
                  <Button
                    variant="contained"
                    startIcon={<CameraIcon />}
                    onClick={captureScreenshot}
                    disabled={!streamActive}
                    size="small"
                  >
                    Capture
                  </Button>
                </Box>
              </Box>

              <Box 
                sx={{ 
                  flexGrow: 1,
                  bgcolor: 'black',
                  borderRadius: 1,
                  overflow: 'hidden',
                  position: 'relative',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {streamActive ? (
                  <img
                    ref={videoRef}
                    src={streamUrl}
                    alt="Video Stream"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain',
                    }}
                    onError={(e) => {
                      console.error('Video stream error:', e);
                    }}
                    onLoad={() => {
                      console.log('Video stream loaded successfully');
                    }}
                  />
                ) : (
                  <Typography color="white" variant="h6">
                    No Video Stream
                  </Typography>
                )}
              </Box>

              
              {/* Video Info Debug Panel */}
              {videoInfo && streamActive && (
                <Box sx={{ mt: 2, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                  <Typography variant="caption" display="block">
                    <strong>Stream Info:</strong> {videoInfo.current_resolution || 'Unknown'} • 
                    FPS: {videoInfo.fps_actual?.toFixed(1) || 'N/A'} • 
                    Status: {videoInfo.status}
                    {videoInfo.resolution && ` • Frame: ${videoInfo.resolution[0]}x${videoInfo.resolution[1]}`}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
          </Box>
        </Box>

        {/* Horizontal Resize Handle */}
        <Box sx={{ py: 1 }}>
          <DragHandle panelType="datasetHeight" orientation="horizontal" />
        </Box>

        {/* Bottom Row - Dataset Management and Captured Images */}
        <Box sx={{ display: 'flex', gap: 2, height: `${Math.max(panelSizes.datasetHeight, panelSizes.imagesHeight)}px` }}>
          <Box sx={{ flex: 1, minWidth: '300px' }}>
          <Box sx={{ height: `${panelSizes.datasetHeight}px`, display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Dataset Selection */}
            <Card sx={{ height: '250px' }}>
              <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'auto' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Dataset Management
                </Typography>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={createDataset}
                  disabled={isCreatingDataset}
                >
                  Create New
                </Button>
              </Box>

              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Select Dataset</InputLabel>
                <Select
                  value={currentDataset && datasets.find(d => d.id === currentDataset.id) ? currentDataset.id : ''}
                  onChange={(e) => {
                    const dataset = datasets.find(d => d.id === e.target.value);
                    setCurrentDataset(dataset || null);
                  }}
                  displayEmpty
                >
                  <MenuItem value="">
                    <em>No dataset selected</em>
                  </MenuItem>
                  {datasets.map((dataset) => (
                    <MenuItem key={dataset.id} value={dataset.id}>
                      {dataset.name} ({dataset.image_count || 0} images)
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {currentDataset && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Training Data Export
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                    <Button
                      variant="contained"
                      startIcon={isGeneratingDataset ? <CircularProgress size={16} color="inherit" /> : <DownloadIcon />}
                      onClick={generateTrainingDataset}
                      disabled={isGeneratingDataset}
                      size="small"
                      color="primary"
                    >
                      {isGeneratingDataset ? 'Generating...' : 'Generate Training Dataset'}
                    </Button>
                    <Chip
                      label={`${capturedImages.filter(img => img.labels).length} labeled`}
                      color="success"
                      size="small"
                    />
                  </Box>
                </Box>
              )}
              
              {!currentDataset && datasets.length > 0 && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                  Select a dataset above to export training data
                </Typography>
              )}
            </CardContent>
          </Card>

            {/* Progress Steps */}
            <Card sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flexGrow: 1, overflow: 'auto' }}>
                <Typography variant="h6" gutterBottom>
                  Progress
                </Typography>
                <Stepper activeStep={currentStep} orientation="vertical">
                  {steps.map((label, index) => (
                    <Step key={label}>
                      <StepLabel>{label}</StepLabel>
                    </Step>
                  ))}
                </Stepper>
              </CardContent>
            </Card>
          </Box>
          </Box>

          {/* Vertical Panel Divider (for visual separation) */}
          <Box sx={{ width: '2px', backgroundColor: 'divider' }} />

          {/* Captured Images */}
          <Box sx={{ flex: 1, minWidth: '300px' }}>
          <Card sx={{ height: `${panelSizes.imagesHeight}px`, display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', minHeight: 0, pb: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="h6">
                  Captured Images
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  <Chip 
                    label={`${capturedImages.length} total`} 
                    size="small" 
                    color="primary"
                    variant="outlined"
                  />
                  {capturedImages.filter(img => img.labels).length > 0 && (
                    <Chip 
                      label={`${capturedImages.filter(img => img.labels).length} labeled`} 
                      size="small" 
                      color="success"
                    />
                  )}
                  {capturedImages.length > 0 && capturedImages.some(img => !img.thumbnail && img.filename) && (
                    <Button
                      size="small"
                      startIcon={<RefreshIcon />}
                      onClick={reloadImagesFromDataset}
                      variant="outlined"
                    >
                      Reload Images
                    </Button>
                  )}
                </Box>
              </Box>

              {/* Copy/Paste and Multi-Select Controls - Only show for Object Detection */}
              {config.datasetType === DATASET_TYPES.OBJECT_DETECTION && capturedImages.length > 0 && (
                <Box sx={{ mb: 2, p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                      <Button
                        size="small"
                        variant={isMultiSelectMode ? "contained" : "outlined"}
                        startIcon={isMultiSelectMode ? <DeselectIcon /> : <SelectAllIcon />}
                        onClick={() => {
                          setIsMultiSelectMode(!isMultiSelectMode);
                          if (isMultiSelectMode) {
                            setSelectedImages(new Set()); // Clear selections when exiting multi-select
                          }
                        }}
                      >
                        {isMultiSelectMode ? 'Exit Select' : 'Multi-Select'}
                      </Button>
                      
                      {isMultiSelectMode && (
                        <>
                          <Button
                            size="small"
                            startIcon={<CheckedIcon />}
                            onClick={handleSelectAllImages}
                            disabled={capturedImages.length === 0}
                          >
                            {selectedImages.size === capturedImages.length ? 'Deselect All' : 'Select All'}
                          </Button>
                          
                          {selectedImages.size > 0 && (
                            <Chip 
                              label={`${selectedImages.size} selected`}
                              size="small"
                              color="primary"
                            />
                          )}
                        </>
                      )}
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button
                        size="small"
                        startIcon={<PasteIcon />}
                        onClick={() => setShowBulkPasteDialog(true)}
                        disabled={!copiedAnnotations || selectedImages.size === 0 || !isMultiSelectMode}
                        color="secondary"
                      >
                        Paste to Selected ({selectedImages.size})
                      </Button>
                      
                      {copiedAnnotations && (
                        <Chip 
                          label={`${copiedAnnotations.boundingBoxes?.length || 0} boxes copied`}
                          size="small"
                          color="success"
                          variant="outlined"
                          icon={<CopyIcon />}
                        />
                      )}
                    </Box>
                  </Box>
                </Box>
              )}

              {capturedImages.length === 0 ? (
                <Box
                  sx={{
                    flexGrow: 1,
                    bgcolor: 'grey.50',
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: 150,
                  }}
                >
                  <Typography color="text.secondary">
                    Capture screenshots to start labeling
                  </Typography>
                </Box>
              ) : (
                <Box sx={{ 
                  flexGrow: 1, 
                  overflow: 'auto',
                  maxHeight: '320px', // Fixed height for scrolling
                  pr: 1, // Padding for scrollbar
                  '&::-webkit-scrollbar': {
                    width: '8px',
                  },
                  '&::-webkit-scrollbar-track': {
                    bgcolor: 'grey.100',
                    borderRadius: '4px',
                  },
                  '&::-webkit-scrollbar-thumb': {
                    bgcolor: 'grey.400',
                    borderRadius: '4px',
                    '&:hover': {
                      bgcolor: 'grey.500',
                    },
                  },
                }}>
                  <Grid container spacing={1}> {/* Reduced spacing */}
                    {capturedImages.map((image) => {
                      const { thumbnailUrl } = getImageUrls(image);
                      return (
                        <Grid item xs={4} sm={3} md={3} lg={2} key={image.id}> {/* Smaller grid items */}
                          <Card
                            sx={{
                              cursor: 'pointer',
                              border: selectedImages.has(image.id) ? '3px solid' : 
                                      image.labels ? '2px solid' : '1px solid',
                              borderColor: selectedImages.has(image.id) ? 'primary.main' :
                                          image.labels ? 'success.main' : 'divider',
                              '&:hover': { boxShadow: 2 },
                              height: '100%',
                              opacity: isMultiSelectMode && !selectedImages.has(image.id) ? 0.7 : 1,
                            }}
                            onClick={() => {
                              if (isMultiSelectMode) {
                                handleImageSelect(image.id, !selectedImages.has(image.id));
                              }
                            }}
                          >
                            <Box
                              sx={{
                                position: 'relative',
                                paddingTop: '56.25%', // 16:9 aspect ratio (smaller)
                                bgcolor: 'grey.100',
                                backgroundImage: thumbnailUrl ? `url(${thumbnailUrl})` : 'none',
                                backgroundSize: 'cover',
                              backgroundPosition: 'center',
                              backgroundRepeat: 'no-repeat',
                            }}
                          >
                            {/* Selection checkbox in multi-select mode */}
                            {isMultiSelectMode && (
                              <IconButton
                                size="small"
                                sx={{ 
                                  position: 'absolute', 
                                  top: 4, 
                                  left: 4,
                                  bgcolor: 'rgba(255, 255, 255, 0.9)',
                                  '&:hover': { bgcolor: 'rgba(255, 255, 255, 1)' }
                                }}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleImageSelect(image.id, !selectedImages.has(image.id));
                                }}
                              >
                                {selectedImages.has(image.id) ? 
                                  <CheckedIcon color="primary" /> : 
                                  <UncheckedIcon />
                                }
                              </IconButton>
                            )}
                            
                            {/* Label indicator */}
                            {image.labels && !isMultiSelectMode && (
                              <Chip
                                label="✓"
                                color="success"
                                size="small"
                                sx={{ 
                                  position: 'absolute', 
                                  top: 2, 
                                  right: 2,
                                  minWidth: '24px',
                                  height: '20px',
                                  fontSize: '12px'
                                }}
                              />
                            )}
                            
                            {/* Show both selection and label in multi-select */}
                            {image.labels && isMultiSelectMode && (
                              <Chip
                                label="✓"
                                color="success"
                                size="small"
                                sx={{ 
                                  position: 'absolute', 
                                  top: 2, 
                                  right: 2,
                                  minWidth: '24px',
                                  height: '20px',
                                  fontSize: '12px'
                                }}
                              />
                            )}
                          </Box>
                          <Box sx={{ p: 0.5 }}> {/* Reduced padding */}
                            <Typography variant="caption" display="block" sx={{ fontSize: '10px', mb: 0.5 }}>
                              {new Date(image.timestamp).toLocaleTimeString()}
                            </Typography>
                            {!isMultiSelectMode && (
                              <Box sx={{ display: 'flex', gap: 0.25, justifyContent: 'center' }}>
                                <IconButton
                                  size="small"
                                  onClick={() => openLabelingDialog(image)}
                                  title={image.labels ? "Edit Labels" : "Label Image"}
                                  sx={{ padding: '4px' }}
                                >
                                  {image.labels ? (
                                    <EditIcon sx={{ fontSize: '16px' }} />
                                  ) : (
                                    <LabelIcon sx={{ fontSize: '16px' }} />
                                  )}
                                </IconButton>
                                <IconButton 
                                  size="small"
                                  onClick={() => handleViewImage(image)}
                                  title="View Image"
                                  sx={{ padding: '4px' }}
                                >
                                  <ViewIcon sx={{ fontSize: '16px' }} />
                                </IconButton>
                                <IconButton 
                                  size="small" 
                                  color="error"
                                  onClick={() => handleDeleteImage(image.id)}
                                  title="Delete Image"
                                  sx={{ padding: '4px' }}
                                >
                                  <DeleteIcon sx={{ fontSize: '16px' }} />
                                </IconButton>
                              </Box>
                            )}
                          </Box>
                        </Card>
                      </Grid>
                      );
                    })}
                  </Grid>
                </Box>
              )}
            </CardContent>
          </Card>
          </Box>
        </Box>
      </Box>

      {/* Labeling Dialog */}
      <Dialog
        open={labelingDialogOpen}
        onClose={() => setLabelingDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {config.datasetType && DATASET_TYPE_INFO[config.datasetType] && (
                <Chip
                  label={`${DATASET_TYPE_INFO[config.datasetType].icon} ${DATASET_TYPE_INFO[config.datasetType].name}`}
                  color="primary"
                  variant="outlined"
                />
              )}
              Label Image
            </Box>
            
            {config.datasetType === DATASET_TYPES.OBJECT_DETECTION && copiedAnnotations && (
              <Chip 
                label={`${copiedAnnotations.boundingBoxes?.length || 0} boxes copied`}
                size="small"
                color="success"
                variant="outlined"
                icon={<CopyIcon />}
              />
            )}
          </Box>
        </DialogTitle>
        
        {/* Dataset Selection for Labeling */}
        <Box sx={{ px: 3, py: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
          <FormControl size="small" sx={{ minWidth: 300 }}>
            <InputLabel>Save to Dataset</InputLabel>
            <Select
              value={currentDataset && datasets.find(d => d.id === currentDataset.id) ? currentDataset.id : ''}
              onChange={(e) => {
                const dataset = datasets.find(d => d.id === e.target.value);
                setCurrentDataset(dataset || null);
              }}
              displayEmpty
            >
              <MenuItem value="">
                <em>No dataset selected</em>
              </MenuItem>
              {datasets.map((dataset) => (
                <MenuItem key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.image_count || 0} images)
                </MenuItem>
              ))}
            </Select>
            <Box sx={{ mt: 1, display: 'flex', gap: 1, alignItems: 'center' }}>
              <Button
                variant="outlined"
                size="small"
                onClick={createDataset}
                disabled={isCreatingDataset}
              >
                {isCreatingDataset ? 'Creating...' : 'Create New Dataset'}
              </Button>
              {!currentDataset && (
                <Typography variant="caption" color="error">
                  Please select or create a dataset to save labels
                </Typography>
              )}
            </Box>
          </FormControl>
        </Box>
        
        <DialogContent sx={{ minHeight: 500 }}>
          {selectedImage && (
            <Box>
              {/* Render appropriate labeling interface based on dataset type */}
              {config.datasetType === DATASET_TYPES.OBJECT_DETECTION && selectedImage && (
                <ObjectDetectionLabeler
                  image={getImageUrls(selectedImage).fullImageUrl}
                  labels={currentLabels}
                  onLabelsChange={setCurrentLabels}
                  copiedAnnotations={copiedAnnotations}
                  onCopyAnnotations={handleCopyAnnotations}
                  showCopyPaste={true}
                />
              )}
              
              {config.datasetType === DATASET_TYPES.IMAGE_CLASSIFICATION && selectedImage && (
                <ImageClassificationLabeler
                  image={getImageUrls(selectedImage).fullImageUrl}
                  labels={currentLabels}
                  onLabelsChange={setCurrentLabels}
                />
              )}
              
              {config.datasetType === DATASET_TYPES.VISION_LLM && selectedImage && (
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <img
                      src={getImageUrls(selectedImage).fullImageUrl}
                      alt="Screenshot"
                      style={{
                        width: '100%',
                        height: 'auto',
                        borderRadius: 8,
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth margin="normal" size="small">
                      <InputLabel>Screen Type</InputLabel>
                      <Select
                        value={currentLabels.screen_type}
                        onChange={(e) => setCurrentLabels(prev => ({ ...prev, screen_type: e.target.value }))}
                      >
                        {Object.entries(SCREEN_STATES).map(([key, label]) => (
                          <MenuItem key={key} value={key}>{label}</MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <FormControl fullWidth margin="normal" size="small">
                      <InputLabel>App Name</InputLabel>
                      <Select
                        value={currentLabels.app_name}
                        onChange={(e) => setCurrentLabels(prev => ({ ...prev, app_name: e.target.value }))}
                      >
                        <MenuItem value="">None</MenuItem>
                        {COMMON_APPS.map((app) => (
                          <MenuItem key={app} value={app}>{app}</MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                      UI Elements Present
                    </Typography>
                    <FormGroup row>
                      {UI_ELEMENTS.map((element) => (
                        <FormControlLabel
                          key={element}
                          control={
                            <Checkbox
                              checked={currentLabels.ui_elements.includes(element)}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setCurrentLabels(prev => ({
                                    ...prev,
                                    ui_elements: [...prev.ui_elements, element]
                                  }));
                                } else {
                                  setCurrentLabels(prev => ({
                                    ...prev,
                                    ui_elements: prev.ui_elements.filter(el => el !== element)
                                  }));
                                }
                              }}
                              size="small"
                            />
                          }
                          label={element}
                        />
                      ))}
                    </FormGroup>

                    <TextField
                      fullWidth
                      multiline
                      rows={2}
                      label="Notes"
                      value={currentLabels.notes}
                      onChange={(e) => setCurrentLabels(prev => ({ ...prev, notes: e.target.value }))}
                      margin="normal"
                      size="small"
                    />
                  </Grid>
                </Grid>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLabelingDialogOpen(false)}>Cancel</Button>
          <Button onClick={saveLabeledImage} variant="contained">
            Save Labels
          </Button>
        </DialogActions>
      </Dialog>

      {/* View Image Dialog */}
      <Dialog
        open={viewImageDialog}
        onClose={() => setViewImageDialog(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>View Image</DialogTitle>
        <DialogContent>
          {selectedImage && (
            <Box sx={{ textAlign: 'center' }}>
              <img
                src={getImageUrls(selectedImage).fullImageUrl}
                alt="Screenshot"
                style={{
                  maxWidth: '100%',
                  height: 'auto',
                  borderRadius: 8,
                }}
              />
              {selectedImage.labels && (
                <Box sx={{ mt: 2, textAlign: 'left' }}>
                  <Typography variant="h6">Labels:</Typography>
                  <Typography>Screen Type: {SCREEN_STATES[selectedImage.labels.screen_type] || 'Unknown'}</Typography>
                  <Typography>App: {selectedImage.labels.app_name || 'None'}</Typography>
                  <Typography>UI Elements: {selectedImage.labels.ui_elements?.join(', ') || 'None'}</Typography>
                  <Typography>Notes: {selectedImage.labels.notes || 'None'}</Typography>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewImageDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Dataset Type Selector Dialog */}
      <DatasetTypeSelector
        open={showDatasetTypeSelector}
        onClose={() => setShowDatasetTypeSelector(false)}
        onSelect={handleDatasetTypeSelection}
        currentConfig={config}
      />

      {/* Bulk Paste Confirmation Dialog */}
      <Dialog
        open={showBulkPasteDialog}
        onClose={() => setShowBulkPasteDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Bulk Paste Annotations</DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            You are about to paste <strong>{copiedAnnotations?.boundingBoxes?.length || 0} bounding boxes</strong> 
            to <strong>{selectedImages.size} selected images</strong>.
          </Alert>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            This will:
            • Add the copied annotations to all selected images
            • Save the changes automatically to your dataset
            • Allow you to fine-tune each image individually afterwards
          </Typography>
          
          {copiedAnnotations?.imageInfo && (
            <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
              <Typography variant="subtitle2">Source Image:</Typography>
              <Typography variant="body2">{copiedAnnotations.imageInfo.imageName}</Typography>
              <Typography variant="caption" color="text.secondary">
                {copiedAnnotations.boundingBoxes?.length || 0} bounding boxes
              </Typography>
            </Box>
          )}
          
          {isGeneratingDataset && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress />
              <Typography variant="caption" color="text.secondary">
                Pasting annotations to selected images...
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setShowBulkPasteDialog(false)}
            disabled={isGeneratingDataset}
          >
            Cancel
          </Button>
          <Button
            onClick={handleBulkPasteAnnotations}
            variant="contained"
            color="secondary"
            disabled={isGeneratingDataset || selectedImages.size === 0}
            startIcon={isGeneratingDataset ? <CircularProgress size={16} /> : <PasteIcon />}
          >
            {isGeneratingDataset ? 'Pasting...' : `Paste to ${selectedImages.size} Images`}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DatasetCreation;