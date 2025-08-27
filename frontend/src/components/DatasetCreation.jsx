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
} from '@mui/material';
import {
  PhotoCamera as CameraIcon,
  PowerSettingsNew as PowerIcon,
  Lock as LockIcon,
  LockOpen as UnlockIcon,
  Save as SaveIcon,
  Label as LabelIcon,
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
} from '@mui/icons-material';
import { videoAPI, deviceAPI, datasetAPI } from '../services/api.jsx';

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

const DatasetCreation = ({ onNotification }) => {
  // Configuration state
  const [config, setConfig] = useState({
    deviceId: '',
    outlet: '5',
    resolution: '1920x1080',
    macAddress: '',
    keySet: 'SKYQ',
  });

  // Platform state
  const [isInitialized, setIsInitialized] = useState(false);
  const [deviceLocked, setDeviceLocked] = useState(false);
  const [streamActive, setStreamActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // Video stream
  const videoRef = useRef(null);
  const [streamUrl, setStreamUrl] = useState('');
  const [capturedImages, setCapturedImages] = useState([]);
  const [videoInfo, setVideoInfo] = useState(null);

  // Dataset management
  const [currentDataset, setCurrentDataset] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);

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

  // Load datasets on component mount
  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await datasetAPI.listDatasets();
      setDatasets(response.data?.datasets || []);
    } catch (error) {
      console.error('Failed to load datasets:', error);
      setDatasets([]); // Ensure datasets is always an array
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

  const handleInitializePlatform = async () => {
    if (!config.deviceId || !config.macAddress) {
      onNotification({
        type: 'error',
        title: 'Configuration Error',
        message: 'Please provide Device ID and MAC Address'
      });
      return;
    }

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
          response = await deviceAPI.pressKey(key);
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
    try {
      const response = await videoAPI.captureScreenshot();
      
      // Get the actual screenshot image
      const frame = await getVideoFrame();
      
      const newImage = {
        id: Date.now(),
        path: response.data.screenshot_path,
        timestamp: response.data.timestamp,
        labels: null,
        thumbnail: frame ? `data:image/jpeg;base64,${frame}` : null
      };
      
      setCapturedImages(prev => [newImage, ...prev]);
      setCurrentStep(Math.max(currentStep, 3));
      
      onNotification({
        type: 'success',
        title: 'Screenshot Captured',
        message: 'Image saved for labeling'
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Capture Failed',
        message: error.message
      });
    }
  };

  const getVideoFrame = async () => {
    return new Promise((resolve) => {
      const img = videoRef.current;
      if (!img) {
        resolve('');
        return;
      }

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      canvas.width = img.naturalWidth || 704;
      canvas.height = img.naturalHeight || 480;
      
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL('image/jpeg', 0.8).split(',')[1]);
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
    setCurrentLabels(image.labels || {
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
    setLabelingDialogOpen(true);
  };

  const handleViewImage = (image) => {
    setSelectedImage(image);
    setViewImageDialog(true);
  };

  const saveLabeledImage = async () => {
    if (!currentDataset || !selectedImage) return;
    
    try {
      // Save to backend
      await datasetAPI.labelImage(
        currentDataset.name,
        selectedImage.path.split('/').pop(),
        currentLabels.screen_type,
        currentLabels.notes
      );
      
      // Update local state
      const updatedImages = capturedImages.map(img => 
        img.id === selectedImage.id 
          ? { ...img, labels: currentLabels }
          : img
      );
      setCapturedImages(updatedImages);
      setCurrentStep(Math.max(currentStep, 4));
      
      setLabelingDialogOpen(false);
      onNotification({
        type: 'success',
        title: 'Labels Saved',
        message: 'Image labeled successfully'
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Labeling Failed',
        message: error.message
      });
    }
  };

  const createDataset = async () => {
    setIsCreatingDataset(true);
    const datasetName = prompt('Enter dataset name:');
    
    if (!datasetName) {
      setIsCreatingDataset(false);
      return;
    }
    
    try {
      const response = await datasetAPI.createDataset(datasetName, 'TV/STB Screen Dataset');
      setCurrentDataset(response.data);
      await loadDatasets();
      
      onNotification({
        type: 'success',
        title: 'Dataset Created',
        message: `Dataset "${datasetName}" created successfully`
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Dataset Creation Failed',
        message: error.response?.data?.detail || error.message
      });
    } finally {
      setIsCreatingDataset(false);
    }
  };

  const exportDataset = async () => {
    if (!currentDataset || capturedImages.length === 0) {
      onNotification({
        type: 'warning',
        title: 'No Data to Export',
        message: 'Please capture and label some images first'
      });
      return;
    }
    
    // Create JSON export
    const exportData = {
      dataset: currentDataset,
      images: capturedImages.filter(img => img.labels),
      created_at: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentDataset.name}_export_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    onNotification({
      type: 'success',
      title: 'Dataset Exported',
      message: 'Training data exported successfully'
    });
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
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Dataset Creation
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Create labeled datasets for TV/STB vision model training
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ minHeight: '720px' }}>
        {/* Top Row - Configuration and Video Stream */}
        <Grid item xs={12} md={4} sx={{ height: '400px' }}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1, overflow: 'auto' }}>
              <Typography variant="h6" gutterBottom>
                Platform Configuration
              </Typography>

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
                  startIcon={deviceLocked ? <LockIcon /> : <UnlockIcon />}
                  onClick={handleDeviceLock}
                  disabled={!isInitialized}
                  color={deviceLocked ? "primary" : "inherit"}
                  sx={{ mb: 2 }}
                >
                  {deviceLocked ? 'Device Locked' : 'Lock Device'}
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

        </Grid>

        {/* Top Row - Video Stream */}
        <Grid item xs={12} md={8} sx={{ height: '400px' }}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
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

              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<CameraIcon />}
                  onClick={captureScreenshot}
                  disabled={!streamActive || !deviceLocked}
                  size="large"
                >
                  Capture Screenshot
                </Button>
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
        </Grid>

        {/* Bottom Row - Dataset Management */}
        <Grid item xs={12} md={6} sx={{ height: '300px' }}>
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Dataset Selection */}
            <Card sx={{ height: '140px' }}>
              <CardContent>
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
                  value={currentDataset?.id || ''}
                  onChange={(e) => {
                    const dataset = datasets.find(d => d.id === e.target.value);
                    setCurrentDataset(dataset);
                  }}
                >
                  {datasets.map((dataset) => (
                    <MenuItem key={dataset.id} value={dataset.id}>
                      {dataset.name} ({dataset.image_count || 0} images)
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {currentDataset && (
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    onClick={exportDataset}
                    size="small"
                  >
                    Export Training Data
                  </Button>
                  <Chip
                    label={`${capturedImages.filter(img => img.labels).length} labeled`}
                    color="success"
                    size="small"
                  />
                </Box>
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
        </Grid>

        {/* Bottom Row - Captured Images */}
        <Grid item xs={12} md={6} sx={{ height: '300px' }}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <Typography variant="h6" gutterBottom>
                Captured Images ({capturedImages.length})
              </Typography>

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
                  overflow: capturedImages.length > 15 ? 'auto' : 'visible',
                  maxHeight: capturedImages.length > 15 ? '100%' : 'none'
                }}>
                  <Grid container spacing={2}>
                    {capturedImages.map((image) => (
                      <Grid item xs={6} sm={4} md={6} key={image.id}>
                        <Card
                          sx={{
                            cursor: 'pointer',
                            border: image.labels ? '2px solid' : '1px solid',
                            borderColor: image.labels ? 'success.main' : 'divider',
                            '&:hover': { boxShadow: 3 },
                          }}
                        >
                          <Box
                            sx={{
                              position: 'relative',
                              paddingTop: '75%', // 4:3 aspect ratio
                              bgcolor: 'grey.100',
                              backgroundImage: image.thumbnail ? `url(${image.thumbnail})` : 'none',
                              backgroundSize: 'cover',
                              backgroundPosition: 'center',
                              backgroundRepeat: 'no-repeat',
                            }}
                          >
                            {image.labels && (
                              <Chip
                                label="Labeled"
                                color="success"
                                size="small"
                                sx={{ position: 'absolute', top: 4, right: 4 }}
                              />
                            )}
                          </Box>
                          <Box sx={{ p: 1 }}>
                            <Typography variant="caption" display="block" gutterBottom>
                              {new Date(image.timestamp).toLocaleTimeString()}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 0.5 }}>
                              <IconButton
                                size="small"
                                onClick={() => openLabelingDialog(image)}
                                title="Label Image"
                              >
                                <LabelIcon />
                              </IconButton>
                              <IconButton 
                                size="small"
                                onClick={() => handleViewImage(image)}
                                title="View Image"
                              >
                                <ViewIcon />
                              </IconButton>
                              <IconButton 
                                size="small" 
                                color="error"
                                onClick={() => handleDeleteImage(image.id)}
                                title="Delete Image"
                              >
                                <DeleteIcon />
                              </IconButton>
                            </Box>
                          </Box>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Labeling Dialog */}
      <Dialog
        open={labelingDialogOpen}
        onClose={() => setLabelingDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Label TV Screen</DialogTitle>
        <DialogContent>
          {selectedImage && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <img
                  src={selectedImage.thumbnail}
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
                src={selectedImage.thumbnail}
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
                  <Typography>UI Elements: {selectedImage.labels.ui_elements.join(', ') || 'None'}</Typography>
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
    </Box>
  );
};

export default DatasetCreation;