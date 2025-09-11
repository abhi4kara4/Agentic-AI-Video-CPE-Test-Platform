import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  LinearProgress,
  Chip,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  FormControlLabel,
  Checkbox,
  Table,
  TableBody,
  TableRow,
  TableCell,
  Menu,
  Divider,
} from '@mui/material';
import {
  ModelTraining as TrainingIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  PlayArrow as ResumeIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  Delete as DeleteIcon,
  ExpandMore as ExpandMoreIcon,
  Timeline as MetricsIcon,
  Storage as DatasetIcon,
  Settings as ConfigIcon,
  Assessment as TestIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import { trainingAPI, datasetAPI, wsManager } from '../services/api.jsx';
import { DATASET_TYPES, DATASET_TYPE_INFO } from '../constants/datasetTypes.js';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

// Training Progress Charts Component
const TrainingProgressCharts = ({ job }) => {
  if (!job.metrics || !job.metrics.length) {
    return (
      <Alert severity="info">
        No training metrics data available yet. Metrics will appear as training progresses.
      </Alert>
    );
  }

  // Prepare data for charts
  const chartData = job.metrics.map((metric, index) => ({
    epoch: metric.epoch || index + 1,
    loss: metric.loss,
    accuracy: metric.accuracy,
    mAP: metric.mAP,
    val_loss: metric.val_loss,
    val_accuracy: metric.val_accuracy,
    val_mAP: metric.val_mAP,
    learning_rate: metric.learning_rate,
    time: new Date(metric.timestamp).getTime()
  }));

  return (
    <Grid container spacing={3}>
      {/* Loss Chart */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Training Loss
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="loss" 
                  stroke="#1976d2" 
                  strokeWidth={2}
                  name="Training Loss"
                />
                {chartData.some(d => d.val_loss !== undefined) && (
                  <Line 
                    type="monotone" 
                    dataKey="val_loss" 
                    stroke="#dc004e" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Validation Loss"
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Accuracy/mAP Chart */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {job.config?.datasetType === 'object_detection' ? 'Mean Average Precision (mAP)' : 'Accuracy'}
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis domain={[0, 1]} />
                <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                <Legend />
                {job.config?.datasetType === 'object_detection' ? (
                  <>
                    <Area 
                      type="monotone" 
                      dataKey="mAP" 
                      stackId="1"
                      stroke="#4caf50" 
                      fill="#4caf50" 
                      fillOpacity={0.3}
                      name="mAP"
                    />
                    {chartData.some(d => d.val_mAP !== undefined) && (
                      <Area 
                        type="monotone" 
                        dataKey="val_mAP" 
                        stackId="2"
                        stroke="#ff9800" 
                        fill="#ff9800" 
                        fillOpacity={0.3}
                        strokeDasharray="5 5"
                        name="Validation mAP"
                      />
                    )}
                  </>
                ) : (
                  <>
                    <Area 
                      type="monotone" 
                      dataKey="accuracy" 
                      stackId="1"
                      stroke="#4caf50" 
                      fill="#4caf50" 
                      fillOpacity={0.3}
                      name="Training Accuracy"
                    />
                    {chartData.some(d => d.val_accuracy !== undefined) && (
                      <Area 
                        type="monotone" 
                        dataKey="val_accuracy" 
                        stackId="2"
                        stroke="#ff9800" 
                        fill="#ff9800" 
                        fillOpacity={0.3}
                        strokeDasharray="5 5"
                        name="Validation Accuracy"
                      />
                    )}
                  </>
                )}
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Learning Rate Chart */}
      {chartData.some(d => d.learning_rate !== undefined) && (
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Learning Rate Schedule
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis scale="log" domain={['dataMin', 'dataMax']} />
                  <Tooltip formatter={(value) => value.toExponential(3)} />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="learning_rate" 
                    stroke="#9c27b0" 
                    strokeWidth={2}
                    name="Learning Rate"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      )}

      {/* Training Time Chart */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Training Progress Timeline
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="time" 
                  type="number"
                  scale="time"
                  domain={['dataMin', 'dataMax']}
                  tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(time) => new Date(time).toLocaleString()}
                  formatter={(value, name) => [
                    name === 'loss' ? value?.toFixed(4) : `${(value * 100)?.toFixed(1)}%`,
                    name
                  ]}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="loss" 
                  stroke="#1976d2" 
                  strokeWidth={2}
                  name="Loss"
                  yAxisId="loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

const ModelTraining = ({ onNotification }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [datasets, setDatasets] = useState([]);
  const [trainingJobs, setTrainingJobs] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [trainingConfig, setTrainingConfig] = useState({
    modelName: '',
    baseModel: 'yolo11n',
    epochs: 100,
    batchSize: 16,
    learningRate: 0.001,
    imageSize: 640,
    patience: 10,
    device: 'auto',
    // PaddleOCR specific parameters
    language: 'en',
    trainType: 'det', // 'det', 'rec', 'cls'
  });
  const [startTrainingDialog, setStartTrainingDialog] = useState(false);
  const [isStartingTraining, setIsStartingTraining] = useState(false);
  const [modelDetailsDialog, setModelDetailsDialog] = useState(false);
  const [selectedModelForDetails, setSelectedModelForDetails] = useState(null);
  const [modelDetails, setModelDetails] = useState(null);
  const [detailsActiveTab, setDetailsActiveTab] = useState(0);
  const [downloadMenuAnchor, setDownloadMenuAnchor] = useState(null);
  const [selectedModelForDownload, setSelectedModelForDownload] = useState(null);

  // Update base model when PaddleOCR parameters change
  useEffect(() => {
    const selectedDatasetType = datasets.find(d => d.id === selectedDataset)?.dataset_type;
    if (selectedDatasetType === DATASET_TYPES.PADDLEOCR) {
      const availableModels = getPaddleOCRModels(trainingConfig.language, trainingConfig.trainType);
      if (availableModels.length > 0) {
        setTrainingConfig(prev => ({ ...prev, baseModel: availableModels[0].value }));
      }
    }
  }, [trainingConfig.language, trainingConfig.trainType, selectedDataset, datasets]);

  // Load data on component mount
  useEffect(() => {
    loadDatasets();
    loadTrainingJobs();
    loadModels();

    // Set up WebSocket listeners for training updates
    const handleTrainingStarted = (data) => {
      console.log('Training started:', data);
      loadTrainingJobs();
      onNotification({
        type: 'info',
        title: 'Training Started',
        message: `Training started for ${data.model_name}`
      });
    };

    const handleTrainingProgress = (data) => {
      console.log('Training progress received:', data);
      
      // Update the specific job's progress in real-time without full API refresh
      setTrainingJobs(prevJobs => 
        prevJobs.map(job => {
          // Match by job_id, job_name, or model_name from the WebSocket payload
          const isMatchingJob = job.id === data.job_id || 
                               job.id === data.job_name ||
                               job.modelName === data.model_name;
          
          if (isMatchingJob) {
            console.log(`Updating progress for job ${job.id}:`, data.progress);
            
            return {
              ...job,
              progress: {
                current_epoch: data.progress?.current_epoch || data.current_epoch,
                total_epochs: data.progress?.total_epochs || data.total_epochs,
                percentage: data.progress?.percentage || 
                           (data.current_epoch && data.total_epochs ? 
                            Math.round((data.current_epoch / data.total_epochs) * 100) : 0),
                loss: data.progress?.loss || data.loss,
                mAP: data.progress?.mAP || data.mAP,
                accuracy: data.progress?.accuracy || data.accuracy
              },
              status: data.status || 'running' // Use status from WebSocket or default to running
            };
          }
          return job;
        })
      );
    };

    const handleTrainingCompleted = (data) => {
      console.log('Training completed:', data);
      loadTrainingJobs();
      loadModels();
      onNotification({
        type: 'success',
        title: 'Training Completed',
        message: `Model ${data.model_name} training completed successfully!`
      });
    };

    const handleTrainingFailed = (data) => {
      console.log('Training failed:', data);
      loadTrainingJobs();
      onNotification({
        type: 'error',
        title: 'Training Failed',
        message: `Training failed: ${data.error}`
      });
    };

    const handleTrainingStopped = (data) => {
      console.log('Training stopped:', data);
      loadTrainingJobs();
      onNotification({
        type: 'info',
        title: 'Training Stopped',
        message: `Training for ${data.job_name} has been stopped`
      });
    };

    // Add event listeners
    wsManager.on('training_started', handleTrainingStarted);
    wsManager.on('training_progress', handleTrainingProgress);
    wsManager.on('training_completed', handleTrainingCompleted);
    wsManager.on('training_failed', handleTrainingFailed);
    wsManager.on('training_stopped', handleTrainingStopped);

    // Cleanup on unmount
    return () => {
      wsManager.off('training_started', handleTrainingStarted);
      wsManager.off('training_progress', handleTrainingProgress);
      wsManager.off('training_completed', handleTrainingCompleted);
      wsManager.off('training_failed', handleTrainingFailed);
      wsManager.off('training_stopped', handleTrainingStopped);
    };
  }, [onNotification]);

  const loadDatasets = async () => {
    try {
      const response = await datasetAPI.listDatasets();
      setDatasets(response.data?.datasets || []);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const loadTrainingJobs = async () => {
    try {
      const response = await trainingAPI.listTrainingJobs();
      setTrainingJobs(response.data?.jobs || []);
    } catch (error) {
      console.error('Failed to load training jobs:', error);
    }
  };

  const loadModels = async () => {
    try {
      const response = await trainingAPI.listModels();
      setModels(response.data?.models || []);
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };


  const handleStartTraining = async () => {
    if (!selectedDataset || !trainingConfig.modelName) {
      onNotification({
        type: 'warning',
        title: 'Missing Information',
        message: 'Please select a dataset and provide a model name'
      });
      return;
    }

    setIsStartingTraining(true);
    try {
      const dataset = datasets.find(d => d.id === selectedDataset);
      const config = {
        ...trainingConfig,
        datasetId: selectedDataset,
        datasetType: dataset?.dataset_type,
      };

      const response = await trainingAPI.startTraining(config);
      
      onNotification({
        type: 'success',
        title: 'Training Started',
        message: `Training job ${response.data.jobId} started successfully`
      });

      setStartTrainingDialog(false);
      await loadTrainingJobs();
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Training Failed',
        message: error.response?.data?.detail || 'Failed to start training'
      });
    } finally {
      setIsStartingTraining(false);
    }
  };

  const handleStopTraining = async (jobId) => {
    try {
      await trainingAPI.stopTraining(jobId);
      onNotification({
        type: 'info',
        title: 'Training Stopped',
        message: `Training job ${jobId} has been stopped`
      });
      await loadTrainingJobs();
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Stop Failed',
        message: 'Failed to stop training job'
      });
    }
  };

  const handleResumeTraining = async (jobId) => {
    try {
      await trainingAPI.resumeTraining(jobId);
      onNotification({
        type: 'success',
        title: 'Training Resumed',
        message: `Training job ${jobId} has been resumed`
      });
      await loadTrainingJobs();
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Resume Failed',
        message: error.response?.data?.detail || 'Failed to resume training job'
      });
    }
  };

  const handleDeleteTraining = async (jobId) => {
    if (!confirm(`Are you sure you want to delete training job ${jobId}? This action cannot be undone.`)) {
      return;
    }
    
    try {
      await trainingAPI.deleteTrainingJob(jobId);
      onNotification({
        type: 'success',
        title: 'Job Deleted',
        message: `Training job ${jobId} has been deleted`
      });
      await loadTrainingJobs();
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Delete Failed',
        message: error.response?.data?.detail || 'Failed to delete training job'
      });
    }
  };

  const handleDownloadMenuOpen = (event, modelName) => {
    setDownloadMenuAnchor(event.currentTarget);
    setSelectedModelForDownload(modelName);
  };

  const handleDownloadMenuClose = () => {
    setDownloadMenuAnchor(null);
    setSelectedModelForDownload(null);
  };

  const handleDownloadModel = async (modelName, fileType = 'zip') => {
    try {
      let response;
      let filename;
      
      if (fileType === 'zip') {
        // Download complete model as zip
        response = await trainingAPI.downloadModelZip(modelName);
        filename = `${modelName}_complete.zip`;
      } else {
        // Download individual weight file
        response = await trainingAPI.downloadModelFile(modelName, fileType);
        filename = `${modelName}_${fileType}.pt`;
      }
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      onNotification({
        type: 'success',
        title: 'Download Started',
        message: `Downloading ${filename}`
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Download Failed',
        message: error.response?.data?.detail || 'Failed to download model'
      });
    }
    
    handleDownloadMenuClose();
  };

  const handleViewModelDetails = async (model) => {
    try {
      setSelectedModelForDetails(model);
      setModelDetailsDialog(true);
      
      const response = await trainingAPI.getModelDetails(model.name);
      setModelDetails(response.data);
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Failed to Load Details',
        message: error.response?.data?.detail || 'Failed to load model details'
      });
      setModelDetails({
        name: model.name,
        type: model.type,
        size: 'N/A',
        accuracy: 'N/A',
        training_time: 'N/A',
        error: 'Could not load detailed metrics'
      });
    }
  };

  const handleDeleteModel = async (modelName) => {
    if (!confirm(`Are you sure you want to delete model "${modelName}"? This action cannot be undone.`)) {
      return;
    }
    
    try {
      await trainingAPI.deleteModel(modelName);
      
      onNotification({
        type: 'success',
        title: 'Model Deleted',
        message: `Model ${modelName} has been deleted successfully`
      });
      
      // Refresh the models list
      await loadModels();
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Delete Failed',
        message: error.response?.data?.detail || 'Failed to delete model'
      });
    }
  };

  const getDatasetTypeColor = (type) => {
    switch (type) {
      case DATASET_TYPES.OBJECT_DETECTION: return 'primary';
      case DATASET_TYPES.IMAGE_CLASSIFICATION: return 'secondary';
      case DATASET_TYPES.VISION_LLM: return 'success';
      case DATASET_TYPES.PADDLEOCR: return 'warning';
      default: return 'default';
    }
  };

  const getJobStatusColor = (status) => {
    switch (status) {
      case 'running': return 'primary';
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'stopped': return 'warning';
      default: return 'default';
    }
  };

  const getPaddleOCRModels = (language, trainType) => {
    const models = [];
    const langPrefix = language === 'en' ? 'en' : language === 'ch' ? 'ch' : language;
    
    switch (trainType) {
      case 'det':
        models.push(
          { value: `${langPrefix}_PP-OCRv4_det`, label: `PaddleOCR v4 Detection (${language.toUpperCase()})` },
          { value: `${langPrefix}_PP-OCRv3_det`, label: `PaddleOCR v3 Detection (${language.toUpperCase()})` }
        );
        break;
      case 'rec':
        models.push(
          { value: `${langPrefix}_PP-OCRv4_rec`, label: `PaddleOCR v4 Recognition (${language.toUpperCase()})` },
          { value: `${langPrefix}_PP-OCRv3_rec`, label: `PaddleOCR v3 Recognition (${language.toUpperCase()})` }
        );
        break;
      case 'cls':
        models.push(
          { value: `${langPrefix}_ppocr_mobile_v2.0_cls`, label: `PaddleOCR Classification (${language.toUpperCase()})` }
        );
        break;
    }
    
    return models;
  };

  const getBaseModels = (datasetType) => {
    switch (datasetType) {
      case DATASET_TYPES.OBJECT_DETECTION:
        return [
          { value: 'yolo11n', label: 'YOLO11 Nano (Fast)' },
          { value: 'yolo11s', label: 'YOLO11 Small' },
          { value: 'yolo11m', label: 'YOLO11 Medium' },
          { value: 'yolo11l', label: 'YOLO11 Large' },
          { value: 'yolo11x', label: 'YOLO11 Extra Large' },
        ];
      case DATASET_TYPES.IMAGE_CLASSIFICATION:
        return [
          { value: 'resnet50', label: 'ResNet50' },
          { value: 'efficientnet_b0', label: 'EfficientNet B0' },
          { value: 'mobilenet_v3', label: 'MobileNet V3' },
          { value: 'vit_base', label: 'Vision Transformer Base' },
        ];
      case DATASET_TYPES.VISION_LLM:
        return [
          { value: 'llava-7b', label: 'LLaVA 7B' },
          { value: 'llava-13b', label: 'LLaVA 13B' },
          { value: 'moondream2', label: 'Moondream2' },
          { value: 'blip2', label: 'BLIP2' },
        ];
      case DATASET_TYPES.PADDLEOCR:
        return getPaddleOCRModels(trainingConfig.language, trainingConfig.trainType);
      default:
        return [];
    }
  };


  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Model Training
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Train and fine-tune AI models for TV/STB recognition
        </Typography>
      </Box>

      <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)} sx={{ mb: 3 }}>
        <Tab label="Start Training" icon={<StartIcon />} />
        <Tab label="Training Jobs" icon={<TrainingIcon />} />
        <Tab label="Trained Models" icon={<ConfigIcon />} />
      </Tabs>

      {/* Start Training Tab */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <DatasetIcon sx={{ mr: 1, verticalAlign: 'bottom' }} />
                  Available Datasets
                </Typography>
                
                {datasets.length === 0 ? (
                  <Alert severity="info">
                    No datasets found. Create and label datasets first in the Dataset Creation tab.
                  </Alert>
                ) : (
                  <List>
                    {datasets.map((dataset) => (
                      <ListItem
                        key={dataset.id}
                        button
                        selected={selectedDataset === dataset.id}
                        onClick={() => setSelectedDataset(dataset.id)}
                      >
                        <ListItemText
                          primary={dataset.name}
                          secondary={`${dataset.dataset_type} • ${dataset.image_count || 0} images`}
                        />
                        <ListItemSecondaryAction>
                          <Chip
                            label={DATASET_TYPE_INFO[dataset.dataset_type]?.name || dataset.dataset_type}
                            color={getDatasetTypeColor(dataset.dataset_type)}
                            size="small"
                          />
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>
                )}

                <Button
                  variant="contained"
                  startIcon={<StartIcon />}
                  onClick={() => setStartTrainingDialog(true)}
                  disabled={!selectedDataset}
                  fullWidth
                  sx={{ mt: 2 }}
                >
                  Configure Training
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <MetricsIcon sx={{ mr: 1, verticalAlign: 'bottom' }} />
                  Quick Start Guide
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" color="primary" gutterBottom>
                    1. Select Dataset
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Choose a dataset with labeled images from the list
                  </Typography>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" color="primary" gutterBottom>
                    2. Configure Training
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Set model name, base model, and training parameters
                  </Typography>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" color="primary" gutterBottom>
                    3. Monitor Progress
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Watch training metrics and logs in real-time
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="subtitle2" color="primary" gutterBottom>
                    4. Test Model
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Evaluate performance and deploy for inference
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Training Jobs Tab */}
      {activeTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Training Jobs ({trainingJobs.length})
                </Typography>
                
                {trainingJobs.length === 0 ? (
                  <Alert severity="info">
                    No training jobs found. Start a new training job to see progress here.
                  </Alert>
                ) : (
                  <Grid container spacing={2}>
                    {trainingJobs.map((job) => (
                      <Grid item xs={12} key={job.id}>
                        <Card variant="outlined">
                          <CardContent>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                              <Box sx={{ flex: 1 }}>
                                <Typography variant="h6" gutterBottom>
                                  {job.modelName}
                                </Typography>
                                <Typography variant="body2" color="text.secondary" gutterBottom>
                                  Dataset: {job.datasetName} • Started: {new Date(job.startTime).toLocaleString()}
                                </Typography>
                                {job.config && (
                                  <Typography variant="caption" color="text.secondary">
                                    Base Model: {job.config.baseModel || 'N/A'} • 
                                    Epochs: {job.config.epochs || 'N/A'} • 
                                    Batch Size: {job.config.batch_size || job.config.batchSize || 'N/A'}
                                  </Typography>
                                )}
                              </Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Chip
                                  label={job.status}
                                  color={getJobStatusColor(job.status)}
                                  size="small"
                                />
                              </Box>
                            </Box>

                            {/* Debug info - can be removed later */}
                            {console.log(`Job ${job.id} status: ${job.status}, progress:`, job.progress)}
                            
                            {/* Progress Section for Running Jobs */}
                            {(job.status === 'running' || job.status === 'training') && (
                              <Box sx={{ mb: 2 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
                                  {job.progress ? (
                                    <>
                                      <CircularProgress 
                                        variant="determinate" 
                                        value={job.progress.percentage || 0}
                                        size={40}
                                      />
                                      <Box sx={{ flex: 1 }}>
                                        <Typography variant="body2" color="text.primary">
                                          Epoch {job.progress.current_epoch || 0} / {job.progress.total_epochs || 0}
                                        </Typography>
                                        <LinearProgress 
                                          variant="determinate" 
                                          value={job.progress.percentage || 0} 
                                          sx={{ height: 8, borderRadius: 4 }}
                                        />
                                      </Box>
                                      <Typography variant="body2" color="text.secondary">
                                        {job.progress.percentage || 0}%
                                      </Typography>
                                    </>
                                  ) : (
                                    <>
                                      <CircularProgress 
                                        variant="indeterminate"
                                        size={40}
                                        color="primary"
                                      />
                                      <Box sx={{ flex: 1 }}>
                                        <Typography variant="body2" color="text.primary">
                                          Training starting...
                                        </Typography>
                                        <LinearProgress variant="indeterminate" sx={{ height: 8, borderRadius: 4 }} />
                                      </Box>
                                    </>
                                  )}
                                </Box>
                                
                                {job.progress && (
                                  <Box sx={{ display: 'flex', gap: 3, mt: 1 }}>
                                    {job.progress.loss && (
                                      <Typography variant="caption" color="text.secondary">
                                        Loss: {job.progress.loss}
                                      </Typography>
                                    )}
                                    {job.progress.mAP && (
                                      <Typography variant="caption" color="text.secondary">
                                        mAP: {job.progress.mAP}
                                      </Typography>
                                    )}
                                    {job.progress.accuracy && (
                                      <Typography variant="caption" color="text.secondary">
                                        Accuracy: {(job.progress.accuracy * 100).toFixed(1)}%
                                      </Typography>
                                    )}
                                    {job.progress.perplexity && (
                                      <Typography variant="caption" color="text.secondary">
                                        Perplexity: {job.progress.perplexity}
                                      </Typography>
                                    )}
                                  </Box>
                                )}
                              </Box>
                            )}

                            {/* Training Progress Charts - Expandable */}
                            {(job.status === 'running' || job.status === 'completed') && job.metrics && job.metrics.length > 0 && (
                              <Accordion sx={{ mt: 2, mb: 2 }}>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                  <Typography variant="subtitle2">
                                    <MetricsIcon sx={{ mr: 1, verticalAlign: 'bottom' }} />
                                    Training Progress Charts ({job.metrics.length} data points)
                                  </Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                  <TrainingProgressCharts job={job} />
                                </AccordionDetails>
                              </Accordion>
                            )}

                            {/* Action Buttons */}
                            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                              {job.status === 'pending' && (
                                <>
                                  <Button
                                    startIcon={<ResumeIcon />}
                                    onClick={() => handleResumeTraining(job.id)}
                                    color="primary"
                                    size="small"
                                    variant="contained"
                                  >
                                    Resume
                                  </Button>
                                  <Button
                                    startIcon={<DeleteIcon />}
                                    onClick={() => handleDeleteTraining(job.id)}
                                    color="error"
                                    size="small"
                                    variant="outlined"
                                  >
                                    Delete
                                  </Button>
                                </>
                              )}
                              
                              {job.status === 'running' && (
                                <Button
                                  startIcon={<StopIcon />}
                                  onClick={() => handleStopTraining(job.id)}
                                  color="warning"
                                  size="small"
                                  variant="contained"
                                >
                                  Stop
                                </Button>
                              )}
                              
                              {(job.status === 'completed' || job.status === 'failed' || job.status === 'stopped') && (
                                <Button
                                  startIcon={<DeleteIcon />}
                                  onClick={() => handleDeleteTraining(job.id)}
                                  color="error"
                                  size="small"
                                  variant="outlined"
                                >
                                  Delete
                                </Button>
                              )}
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Trained Models Tab */}
      {activeTab === 2 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Trained Models
            </Typography>
            
            {models.length === 0 ? (
              <Alert severity="info">
                No trained models found. Complete training jobs will appear here.
              </Alert>
            ) : (
              <List>
                {models.map((model) => (
                  <ListItem key={model.name}>
                    <ListItemText
                      primary={model.name}
                      secondary={`Type: ${model.type} • Created: ${new Date(model.createdAt).toLocaleString()}`}
                    />
                    <ListItemSecondaryAction>
                      <IconButton 
                        onClick={(e) => handleDownloadMenuOpen(e, model.name)}
                        title="Download Model"
                      >
                        <DownloadIcon />
                      </IconButton>
                      <IconButton 
                        onClick={() => handleViewModelDetails(model)}
                        title="View Model Details"
                      >
                        <ViewIcon />
                      </IconButton>
                      <IconButton 
                        onClick={() => handleDeleteModel(model.name)} 
                        color="error"
                        title="Delete Model"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                ))}
              </List>
            )}
          </CardContent>
        </Card>
      )}

      {/* Download Menu */}
      <Menu
        anchorEl={downloadMenuAnchor}
        open={Boolean(downloadMenuAnchor)}
        onClose={handleDownloadMenuClose}
      >
        <MenuItem onClick={() => handleDownloadModel(selectedModelForDownload, 'zip')}>
          <DownloadIcon sx={{ mr: 1 }} />
          Download Complete Model (ZIP)
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => handleDownloadModel(selectedModelForDownload, 'best')}>
          <DownloadIcon sx={{ mr: 1 }} />
          Download Best Weights (.pt)
        </MenuItem>
        <MenuItem onClick={() => handleDownloadModel(selectedModelForDownload, 'last')}>
          <DownloadIcon sx={{ mr: 1 }} />
          Download Last Weights (.pt)
        </MenuItem>
      </Menu>

      {/* Training Configuration Dialog */}
      <Dialog
        open={startTrainingDialog}
        onClose={() => setStartTrainingDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Configure Training Parameters</DialogTitle>
        <DialogContent>
          {selectedDataset && (
            <>
              <TextField
                fullWidth
                label="Model Name"
                value={trainingConfig.modelName}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, modelName: e.target.value })}
                margin="normal"
                placeholder="my-tv-detection-model"
              />

              <FormControl fullWidth margin="normal">
                <InputLabel>Base Model</InputLabel>
                <Select
                  value={trainingConfig.baseModel}
                  onChange={(e) => setTrainingConfig({ ...trainingConfig, baseModel: e.target.value })}
                >
                  {getBaseModels(datasets.find(d => d.id === selectedDataset)?.dataset_type).map((model) => (
                    <MenuItem key={model.value} value={model.value}>
                      {model.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Accordion sx={{ mt: 2 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Advanced Parameters</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Epochs"
                        type="number"
                        value={trainingConfig.epochs}
                        onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: parseInt(e.target.value) })}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Batch Size"
                        type="number"
                        value={trainingConfig.batchSize}
                        onChange={(e) => setTrainingConfig({ ...trainingConfig, batchSize: parseInt(e.target.value) })}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Learning Rate"
                        type="number"
                        step="0.0001"
                        value={trainingConfig.learningRate}
                        onChange={(e) => setTrainingConfig({ ...trainingConfig, learningRate: parseFloat(e.target.value) })}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Image Size"
                        type="number"
                        value={trainingConfig.imageSize}
                        onChange={(e) => setTrainingConfig({ ...trainingConfig, imageSize: parseInt(e.target.value) })}
                      />
                    </Grid>
                    
                    {/* PaddleOCR specific parameters */}
                    {datasets.find(d => d.id === selectedDataset)?.dataset_type === DATASET_TYPES.PADDLEOCR && (
                      <>
                        <Grid item xs={6}>
                          <FormControl fullWidth>
                            <InputLabel>Language</InputLabel>
                            <Select
                              value={trainingConfig.language}
                              onChange={(e) => setTrainingConfig({ ...trainingConfig, language: e.target.value })}
                            >
                              <MenuItem value="en">English</MenuItem>
                              <MenuItem value="ch">Chinese</MenuItem>
                              <MenuItem value="ka">Korean</MenuItem>
                              <MenuItem value="japan">Japanese</MenuItem>
                              <MenuItem value="fr">French</MenuItem>
                              <MenuItem value="german">German</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>
                        <Grid item xs={6}>
                          <FormControl fullWidth>
                            <InputLabel>Training Type</InputLabel>
                            <Select
                              value={trainingConfig.trainType}
                              onChange={(e) => setTrainingConfig({ ...trainingConfig, trainType: e.target.value })}
                            >
                              <MenuItem value="det">Text Detection</MenuItem>
                              <MenuItem value="rec">Text Recognition</MenuItem>
                              <MenuItem value="cls">Text Classification</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>
                      </>
                    )}
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStartTrainingDialog(false)}>Cancel</Button>
          <Button
            onClick={handleStartTraining}
            variant="contained"
            disabled={isStartingTraining}
            startIcon={isStartingTraining ? <CircularProgress size={16} /> : <StartIcon />}
          >
            {isStartingTraining ? 'Starting...' : 'Start Training'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Enhanced Model Details Dialog */}
      <Dialog
        open={modelDetailsDialog}
        onClose={() => setModelDetailsDialog(false)}
        maxWidth="xl"
        fullWidth
        PaperProps={{
          sx: { height: '90vh' }
        }}
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h5" fontWeight="bold">
              {selectedModelForDetails?.name}
            </Typography>
            {modelDetails && (
              <Chip 
                label={DATASET_TYPE_INFO[modelDetails.dataset_type]?.name || modelDetails.dataset_type}
                color="primary"
                size="small"
              />
            )}
          </Box>
        </DialogTitle>
        <DialogContent sx={{ p: 0 }}>
          {modelDetails ? (
            <Box sx={{ height: '100%' }}>
              {/* Navigation Tabs */}
              <Tabs 
                value={detailsActiveTab} 
                onChange={(e, newValue) => setDetailsActiveTab(newValue)}
                variant="scrollable"
                scrollButtons="auto"
                sx={{ borderBottom: 1, borderColor: 'divider', px: 2 }}
              >
                <Tab label="Overview" icon={<ConfigIcon />} />
                <Tab label="Training Charts" icon={<MetricsIcon />} />
                <Tab label="Performance Analysis" icon={<AssessmentIcon />} />
                <Tab label="Dataset Info" icon={<DatasetIcon />} />
                <Tab label="Training Logs" icon={<TimelineIcon />} />
              </Tabs>

              {/* Tab Content */}
              <Box sx={{ p: 3, height: 'calc(100% - 48px)', overflow: 'auto' }}>
                {/* Overview Tab */}
                {detailsActiveTab === 0 && (
                  <Grid container spacing={3}>
                    {/* Training Summary */}
                    <Grid item xs={12}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Training Summary
                          </Typography>
                          <Grid container spacing={2}>
                            <Grid item xs={12} md={6}>
                              <Table size="small">
                                <TableBody>
                                  <TableRow>
                                    <TableCell><strong>Model Name</strong></TableCell>
                                    <TableCell>{modelDetails.name}</TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell><strong>Base Model</strong></TableCell>
                                    <TableCell>{modelDetails.base_model || 'N/A'}</TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell><strong>Dataset</strong></TableCell>
                                    <TableCell>{modelDetails.dataset_name || 'N/A'}</TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell><strong>Training Status</strong></TableCell>
                                    <TableCell>
                                      <Chip 
                                        label={modelDetails.status || 'Unknown'}
                                        color={modelDetails.status === 'completed' ? 'success' : 'warning'}
                                        size="small"
                                      />
                                    </TableCell>
                                  </TableRow>
                                </TableBody>
                              </Table>
                            </Grid>
                            <Grid item xs={12} md={6}>
                              <Table size="small">
                                <TableBody>
                                  <TableRow>
                                    <TableCell><strong>Created</strong></TableCell>
                                    <TableCell>{new Date(modelDetails.created_at).toLocaleString()}</TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell><strong>Training Time</strong></TableCell>
                                    <TableCell>
                                      {modelDetails.metrics?.training_time ? 
                                        `${Math.round(modelDetails.metrics.training_time)}s` : 'N/A'}
                                    </TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell><strong>Epochs Completed</strong></TableCell>
                                    <TableCell>{modelDetails.metrics?.epochs_completed || 'N/A'}</TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell><strong>Final mAP@0.5</strong></TableCell>
                                    <TableCell>
                                      {modelDetails.metrics?.final_map ? 
                                        `${(modelDetails.metrics.final_map * 100).toFixed(2)}%` : 'N/A'}
                                    </TableCell>
                                  </TableRow>
                                </TableBody>
                              </Table>
                            </Grid>
                          </Grid>
                        </CardContent>
                      </Card>
                    </Grid>

                    {/* Key Metrics */}
                    {modelDetails.epoch_metrics && modelDetails.epoch_metrics.length > 0 && (
                      <Grid item xs={12}>
                        <Card>
                          <CardContent>
                            <Typography variant="h6" gutterBottom>
                              Training Progress
                            </Typography>
                            <Box sx={{ height: 300 }}>
                              <ResponsiveContainer width="100%" height={300}>
                                <LineChart data={modelDetails.epoch_metrics.slice(-20)}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                {modelDetails.epoch_metrics[0]?.['train/box_loss'] !== undefined && (
                                  <Line type="monotone" dataKey="train/box_loss" stroke="#8884d8" name="Box Loss" />
                                )}
                                {modelDetails.epoch_metrics[0]?.['train/cls_loss'] !== undefined && (
                                  <Line type="monotone" dataKey="train/cls_loss" stroke="#82ca9d" name="Class Loss" />
                                )}
                                {modelDetails.epoch_metrics[0]?.['val/mAP50'] !== undefined && (
                                  <Line type="monotone" dataKey="val/mAP50" stroke="#ffc658" name="mAP@0.5" />
                                )}
                                </LineChart>
                              </ResponsiveContainer>
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                    )}
                  </Grid>
                )}

                {/* Training Charts Tab */}
                {detailsActiveTab === 1 && (
                  <Grid container spacing={3}>
                    {modelDetails.artifacts?.training_plots?.map((plot, index) => (
                      <Grid item xs={12} md={6} key={index}>
                        <Card>
                          <CardContent>
                            <Typography variant="h6" gutterBottom>
                              {plot.type.title}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                              {plot.type.description}
                            </Typography>
                            <CardMedia
                              component="img"
                              image={`/api/models/${modelDetails.name}/artifacts/${plot.name}`}
                              alt={plot.type.title}
                              sx={{ 
                                maxHeight: 400,
                                objectFit: 'contain',
                                border: '1px solid #e0e0e0',
                                borderRadius: 1
                              }}
                            />
                            <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Chip 
                                label={plot.type.category}
                                size="small"
                                variant="outlined"
                              />
                              <Typography variant="caption" color="text.secondary">
                                {(plot.size / 1024).toFixed(1)} KB
                              </Typography>
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                    {(!modelDetails.artifacts?.training_plots || modelDetails.artifacts.training_plots.length === 0) && (
                      <Grid item xs={12}>
                        <Alert severity="info">
                          No training charts available for this model.
                        </Alert>
                      </Grid>
                    )}
                  </Grid>
                )}

                {/* Other tabs will be implemented here */}
                {detailsActiveTab > 1 && (
                  <Alert severity="info">
                    This section is under development. More detailed analysis coming soon!
                  </Alert>
                )}
              </Box>
            </Box>
          ) : (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setModelDetailsDialog(false)}>Close</Button>
          {selectedModelForDetails && (
            <>
              <Button
                startIcon={<DownloadIcon />}
                onClick={(e) => handleDownloadMenuOpen(e, selectedModelForDetails.name)}
                variant="contained"
              >
                Download Model
              </Button>
              <Button
                startIcon={<DeleteIcon />}
                onClick={() => {
                  setModelDetailsDialog(false);
                  handleDeleteModel(selectedModelForDetails.name);
                }}
                color="error"
                variant="outlined"
              >
                Delete Model
              </Button>
            </>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ModelTraining;