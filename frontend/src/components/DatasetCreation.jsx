import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  Button,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  LinearProgress,
  Alert,
  Divider,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Tooltip,
  Badge,
  Stepper,
  Step,
  StepLabel,
  StepContent,
} from '@mui/material';
import {
  VideoCall as VideoIcon,
  Lock as LockIcon,
  LockOpen as UnlockIcon,
  Power as PowerIcon,
  Screenshot as ScreenshotIcon,
  Label as LabelIcon,
  Save as SaveIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Visibility as ViewIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  KeyboardArrowUp,
  KeyboardArrowDown,
  KeyboardArrowLeft,
  KeyboardArrowRight,
  RadioButtonChecked as OKIcon,
  Home as HomeIcon,
  ArrowBack as BackIcon,
} from '@mui/icons-material';
import { videoAPI, deviceAPI, datasetAPI } from '../services/api.jsx';

// Screen state definitions
const SCREEN_STATES = {
  'home': 'Home Screen',
  'app_rail': 'App Selection Rail',
  'app_loading': 'App Loading',
  'app_content': 'App Content',
  'login': 'Login Screen',
  'profile_selection': 'Profile Selection',
  'settings': 'Settings Menu',
  'guide': 'TV Guide/EPG',
  'error': 'Error Screen',
  'no_signal': 'No Signal',
  'black_screen': 'Black Screen',
  'buffering': 'Buffering',
  'video_playing': 'Video Playing',
  'search': 'Search Interface',
  'details': 'Content Details'
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

const DatasetCreation = ({ onNotification }) => {
  // Configuration state
  const [config, setConfig] = useState({
    deviceId: '',
    outlet: '5',
    resolution: '1920x1080',
    macAddress: '',
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

  // Dataset management
  const [currentDataset, setCurrentDataset] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);

  // Labeling state
  const [selectedImage, setSelectedImage] = useState(null);
  const [labelingDialogOpen, setLabelingDialogOpen] = useState(false);
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
      // Start video stream
      const streamUrl = videoAPI.getStreamUrl(config.deviceId, config.outlet, config.resolution);
      setStreamUrl(streamUrl);
      setStreamActive(true);
      
      setIsInitialized(true);
      setCurrentStep(1);
      
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
      onNotification({
        type: 'error',
        title: 'Device Control Failed',
        message: error.response?.data?.detail || error.message
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
      const newImage = {
        id: Date.now(),
        path: response.data.screenshot_path,
        timestamp: response.data.timestamp,
        labels: null,
        thumbnail: `data:image/jpeg;base64,${await getVideoFrame()}`
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
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      if (videoRef.current) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);
        resolve(canvas.toDataURL('image/jpeg', 0.8).split(',')[1]);
      } else {
        resolve('');
      }
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

  const saveLabeledImage = async () => {
    if (!selectedImage || !currentDataset) {
      onNotification({
        type: 'error',
        title: 'Save Failed',
        message: 'Please select a dataset first'
      });
      return;
    }

    try {
      await datasetAPI.addImage(currentDataset.id, selectedImage.thumbnail, currentLabels);
      
      // Update local state
      setCapturedImages(prev =>
        prev.map(img =>
          img.id === selectedImage.id
            ? { ...img, labels: currentLabels }
            : img
        )
      );
      
      setLabelingDialogOpen(false);
      setCurrentStep(Math.max(currentStep, 4));
      
      onNotification({
        type: 'success',
        title: 'Image Labeled',
        message: 'Labels saved to dataset'
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Save Failed',
        message: error.message
      });
    }
  };

  const createNewDataset = async () => {
    const name = prompt('Enter dataset name:');
    if (!name) return;

    const description = prompt('Enter dataset description (optional):') || '';

    try {
      const response = await datasetAPI.createDataset(name, description);
      const newDataset = response.data;
      
      setDatasets(prev => [newDataset, ...prev]);
      setCurrentDataset(newDataset);
      setIsCreatingDataset(false);
      
      onNotification({
        type: 'success',
        title: 'Dataset Created',
        message: `Dataset "${name}" created successfully`
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Creation Failed',
        message: error.message
      });
    }
  };

  const exportDataset = async () => {
    if (!currentDataset) return;

    try {
      const response = await datasetAPI.exportDataset(currentDataset.id, 'llava');
      
      // Download the exported dataset
      const blob = new Blob([JSON.stringify(response.data, null, 2)], {
        type: 'application/json'
      });
      
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${currentDataset.name}_dataset.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      onNotification({
        type: 'success',
        title: 'Dataset Exported',
        message: 'Training dataset downloaded'
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Export Failed',
        message: error.message
      });
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
    <Box sx={{ p: 3, maxWidth: '100%', overflow: 'hidden' }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Dataset Creation
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Create labeled datasets for TV/STB vision model training
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Left Panel - Configuration & Controls */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
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
                    <MenuItem value="1920x1080">1920x1080 (Full HD)</MenuItem>
                    <MenuItem value="1280x720">1280x720 (HD)</MenuItem>
                    <MenuItem value="704x480">704x480 (SD)</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  fullWidth
                  label="Device MAC Address"
                  value={config.macAddress}
                  onChange={(e) => setConfig(prev => ({ ...prev, macAddress: e.target.value }))}
                  margin="normal"
                  size="small"
                />

                <Button
                  fullWidth
                  variant="contained"
                  onClick={handleInitializePlatform}
                  disabled={isInitialized}
                  sx={{ mt: 2 }}
                >
                  {isInitialized ? 'Platform Ready' : 'Initialize Platform'}
                </Button>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Device Controls */}
              <Typography variant="h6" gutterBottom>
                Device Controls
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button
                  fullWidth
                  variant={deviceLocked ? 'contained' : 'outlined'}
                  color={deviceLocked ? 'error' : 'success'}
                  startIcon={deviceLocked ? <UnlockIcon /> : <LockIcon />}
                  onClick={handleDeviceLock}
                  disabled={!isInitialized}
                >
                  {deviceLocked ? 'Unlock Device' : 'Lock Device'}
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
                    onClick={() => handleDeviceControl('key', 'UP')}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <KeyboardArrowUp />
                  </IconButton>
                  <div />
                  
                  <IconButton
                    onClick={() => handleDeviceControl('key', 'LEFT')}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <KeyboardArrowLeft />
                  </IconButton>
                  <IconButton
                    onClick={() => handleDeviceControl('key', 'OK')}
                    disabled={!deviceLocked}
                    color="primary"
                    sx={{ bgcolor: 'primary.main', color: 'white', '&:hover': { bgcolor: 'primary.dark' } }}
                  >
                    <OKIcon />
                  </IconButton>
                  <IconButton
                    onClick={() => handleDeviceControl('key', 'RIGHT')}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <KeyboardArrowRight />
                  </IconButton>
                  
                  <div />
                  <IconButton
                    onClick={() => handleDeviceControl('key', 'DOWN')}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <KeyboardArrowDown />
                  </IconButton>
                  <div />
                </Box>

                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                  <IconButton
                    onClick={() => handleDeviceControl('key', 'HOME')}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <HomeIcon />
                  </IconButton>
                  <IconButton
                    onClick={() => handleDeviceControl('key', 'BACK')}
                    disabled={!deviceLocked}
                    color="primary"
                  >
                    <BackIcon />
                  </IconButton>
                </Box>
              </Box>
            </CardContent>
          </Card>

          {/* Progress Steps */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
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
        </Grid>

        {/* Right Panel - Video Stream & Images */}
        <Grid item xs={12} md={8}>
          {/* Video Stream */}
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Live Video Stream
                </Typography>
                <Box>
                  <Chip
                    label={streamActive ? 'Live' : 'Offline'}
                    color={streamActive ? 'success' : 'error'}
                    size="small"
                    icon={streamActive ? <PlayIcon /> : <StopIcon />}
                  />
                  <Button
                    variant="contained"
                    startIcon={<ScreenshotIcon />}
                    onClick={captureScreenshot}
                    disabled={!streamActive}
                    sx={{ ml: 2 }}
                  >
                    Capture
                  </Button>
                </Box>
              </Box>

              {/* Video Player */}
              {streamUrl ? (
                <Box
                  sx={{
                    position: 'relative',
                    width: '100%',
                    height: 400,
                    bgcolor: 'black',
                    borderRadius: 1,
                    overflow: 'hidden',
                  }}
                >
                  <img
                    ref={videoRef}
                    src={streamUrl}
                    alt="Live Stream"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain',
                    }}
                    onLoad={() => setStreamActive(true)}
                    onError={() => setStreamActive(false)}
                  />
                </Box>
              ) : (
                <Box
                  sx={{
                    height: 400,
                    bgcolor: 'grey.100',
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography color="text.secondary">
                    Initialize platform to start video stream
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Dataset Management */}
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Dataset Management</Typography>
                <Box>
                  <Button
                    variant="outlined"
                    startIcon={<UploadIcon />}
                    sx={{ mr: 1 }}
                    size="small"
                  >
                    Import
                  </Button>
                  <Button
                    variant="contained"
                    startIcon={<SaveIcon />}
                    onClick={createNewDataset}
                    size="small"
                  >
                    New Dataset
                  </Button>
                </Box>
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

          {/* Captured Images */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Captured Images ({capturedImages.length})
              </Typography>

              {capturedImages.length === 0 ? (
                <Box
                  sx={{
                    height: 200,
                    bgcolor: 'grey.50',
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography color="text.secondary">
                    Capture screenshots to start labeling
                  </Typography>
                </Box>
              ) : (
                <Grid container spacing={2}>
                  {capturedImages.map((image) => (
                    <Grid item xs={6} sm={4} md={3} key={image.id}>
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
                            height: 120,
                            bgcolor: 'grey.100',
                            backgroundImage: `url(${image.thumbnail})`,
                            backgroundSize: 'cover',
                            backgroundPosition: 'center',
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
                            >
                              <LabelIcon />
                            </IconButton>
                            <IconButton size="small">
                              <ViewIcon />
                            </IconButton>
                            <IconButton size="small" color="error">
                              <DeleteIcon />
                            </IconButton>
                          </Box>
                        </Box>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
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

                <TextField
                  fullWidth
                  label="Visible Text"
                  value={currentLabels.visible_text}
                  onChange={(e) => setCurrentLabels(prev => ({ ...prev, visible_text: e.target.value }))}
                  margin="normal"
                  size="small"
                  multiline
                  rows={2}
                />

                <TextField
                  fullWidth
                  label="Notes"
                  value={currentLabels.notes}
                  onChange={(e) => setCurrentLabels(prev => ({ ...prev, notes: e.target.value }))}
                  margin="normal"
                  size="small"
                  multiline
                  rows={2}
                />

                <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                  Navigation Hints
                </Typography>
                {Object.entries({up: 'Up', down: 'Down', left: 'Left', right: 'Right'}).map(([key, label]) => (
                  <FormControlLabel
                    key={key}
                    control={
                      <Switch
                        checked={currentLabels.navigation.can_navigate[key]}
                        onChange={(e) => setCurrentLabels(prev => ({
                          ...prev,
                          navigation: {
                            ...prev.navigation,
                            can_navigate: {
                              ...prev.navigation.can_navigate,
                              [key]: e.target.checked
                            }
                          }
                        }))}
                      />
                    }
                    label={`Can navigate ${label}`}
                  />
                ))}
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLabelingDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={saveLabeledImage}
            variant="contained"
            disabled={!currentLabels.screen_type}
          >
            Save Labels
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DatasetCreation;