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
} from '@mui/icons-material';
import { trainingAPI, datasetAPI, testingAPI, wsManager } from '../services/api.jsx';
import { DATASET_TYPES, DATASET_TYPE_INFO } from '../constants/datasetTypes.js';

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
  });
  const [startTrainingDialog, setStartTrainingDialog] = useState(false);
  const [isStartingTraining, setIsStartingTraining] = useState(false);
  
  // Testing state
  const [testHistory, setTestHistory] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [testPrompt, setTestPrompt] = useState('Describe what you see on this TV screen');
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [benchmarkConfig, setBenchmarkConfig] = useState({
    iterations: 5,
    prompt: 'Describe what you see on this TV screen'
  });
  const [compareModels, setCompareModels] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);

  // Load data on component mount
  useEffect(() => {
    loadDatasets();
    loadTrainingJobs();
    loadModels();
    loadTestHistory();

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
      console.log('Training progress:', data);
      loadTrainingJobs(); // Refresh job list to show progress
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

  const loadTestHistory = async () => {
    try {
      const response = await testingAPI.getTestHistory();
      setTestHistory(response.data?.tests || []);
    } catch (error) {
      console.error('Failed to load test history:', error);
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

  const getDatasetTypeColor = (type) => {
    switch (type) {
      case DATASET_TYPES.OBJECT_DETECTION: return 'primary';
      case DATASET_TYPES.IMAGE_CLASSIFICATION: return 'secondary';
      case DATASET_TYPES.VISION_LLM: return 'success';
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
      default:
        return [];
    }
  };

  // Testing handlers
  const handleSingleModelTest = async () => {
    if (!selectedModel) {
      onNotification({
        type: 'warning',
        title: 'No Model Selected',
        message: 'Please select a model to test'
      });
      return;
    }

    setIsTesting(true);
    try {
      const response = await testingAPI.testModel(selectedModel, testPrompt);
      setTestResult(response.data);
      await loadTestHistory();
      
      onNotification({
        type: 'success',
        title: 'Test Completed',
        message: `Model ${selectedModel} analysis completed`
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Test Failed',
        message: error.response?.data?.detail || 'Failed to test model'
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleBenchmarkTest = async () => {
    if (!selectedModel) {
      onNotification({
        type: 'warning',
        title: 'No Model Selected',
        message: 'Please select a model to benchmark'
      });
      return;
    }

    setIsTesting(true);
    try {
      const response = await testingAPI.benchmarkModel(
        selectedModel, 
        benchmarkConfig.iterations, 
        benchmarkConfig.prompt
      );
      setTestResult(response.data);
      await loadTestHistory();
      
      onNotification({
        type: 'success',
        title: 'Benchmark Completed',
        message: `Model ${selectedModel} benchmark completed with ${benchmarkConfig.iterations} iterations`
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Benchmark Failed',
        message: error.response?.data?.detail || 'Failed to benchmark model'
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleCompareModels = async () => {
    if (compareModels.length < 2) {
      onNotification({
        type: 'warning',
        title: 'Insufficient Models',
        message: 'Please select at least 2 models to compare'
      });
      return;
    }

    setIsTesting(true);
    try {
      const response = await testingAPI.compareModels(compareModels, testPrompt);
      setComparisonResult(response.data);
      await loadTestHistory();
      
      onNotification({
        type: 'success',
        title: 'Comparison Completed',
        message: `Compared ${compareModels.length} models successfully`
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Comparison Failed',
        message: error.response?.data?.detail || 'Failed to compare models'
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleClearTestHistory = async () => {
    try {
      await testingAPI.clearTestHistory();
      setTestHistory([]);
      setTestResult(null);
      setComparisonResult(null);
      
      onNotification({
        type: 'info',
        title: 'History Cleared',
        message: 'Test history has been cleared'
      });
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Clear Failed',
        message: 'Failed to clear test history'
      });
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
        <Tab label="Model Testing" icon={<TestIcon />} />
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

                            {/* Progress Section for Running Jobs */}
                            {job.status === 'running' && job.progress && (
                              <Box sx={{ mb: 2 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
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
                                </Box>
                                
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
                              </Box>
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
                      <IconButton onClick={() => console.log('Download', model.name)}>
                        <DownloadIcon />
                      </IconButton>
                      <IconButton onClick={() => console.log('View', model.name)}>
                        <ViewIcon />
                      </IconButton>
                      <IconButton onClick={() => console.log('Delete', model.name)} color="error">
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

      {/* Model Testing Tab */}
      {activeTab === 3 && (
        <Grid container spacing={3}>
          {/* Testing Controls */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <TestIcon sx={{ mr: 1, verticalAlign: 'bottom' }} />
                  Model Testing
                </Typography>
                
                <FormControl fullWidth margin="normal">
                  <InputLabel>Select Model</InputLabel>
                  <Select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    {models.map((model) => (
                      <MenuItem key={model.name} value={model.name}>
                        {model.name} ({model.type})
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <TextField
                  fullWidth
                  label="Test Prompt"
                  value={testPrompt}
                  onChange={(e) => setTestPrompt(e.target.value)}
                  margin="normal"
                  multiline
                  rows={2}
                  placeholder="Describe what you see on this TV screen"
                />

                <Button
                  variant="contained"
                  onClick={handleSingleModelTest}
                  disabled={isTesting || !selectedModel}
                  fullWidth
                  sx={{ mt: 2, mb: 1 }}
                  startIcon={isTesting ? <CircularProgress size={16} /> : <TestIcon />}
                >
                  {isTesting ? 'Testing...' : 'Run Single Test'}
                </Button>

                <Button
                  variant="outlined"
                  onClick={handleBenchmarkTest}
                  disabled={isTesting || !selectedModel}
                  fullWidth
                  sx={{ mb: 1 }}
                  startIcon={<MetricsIcon />}
                >
                  Run Benchmark
                </Button>

                <Accordion sx={{ mt: 2 }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Benchmark Config</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <TextField
                      fullWidth
                      label="Iterations"
                      type="number"
                      value={benchmarkConfig.iterations}
                      onChange={(e) => setBenchmarkConfig({
                        ...benchmarkConfig,
                        iterations: parseInt(e.target.value) || 5
                      })}
                      margin="normal"
                      inputProps={{ min: 1, max: 20 }}
                    />
                  </AccordionDetails>
                </Accordion>
              </CardContent>
            </Card>
          </Grid>

          {/* Test Results */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Test Results</Typography>
                  <Button
                    onClick={handleClearTestHistory}
                    color="warning"
                    size="small"
                    startIcon={<DeleteIcon />}
                  >
                    Clear History
                  </Button>
                </Box>

                {testResult && (
                  <Box sx={{ mb: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      Latest Test Result
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Model: {testResult.model} | {new Date(testResult.timestamp).toLocaleString()}
                    </Typography>
                    <Typography variant="body1" sx={{ mt: 1 }}>
                      {testResult.response}
                    </Typography>
                    {testResult.confidence && (
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Confidence: {(testResult.confidence * 100).toFixed(1)}%
                      </Typography>
                    )}
                    {testResult.processing_time_seconds && (
                      <Typography variant="body2" color="text.secondary">
                        Processing Time: {testResult.processing_time_seconds.toFixed(2)}s
                      </Typography>
                    )}
                    {testResult.statistics && (
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          Benchmark: {testResult.iterations} iterations, 
                          Avg: {testResult.statistics.average_processing_time.toFixed(2)}s, 
                          Total: {testResult.statistics.total_time.toFixed(2)}s
                        </Typography>
                      </Box>
                    )}
                  </Box>
                )}

                {comparisonResult && (
                  <Box sx={{ mb: 3, p: 2, bgcolor: 'blue.50', borderRadius: 1 }}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      Model Comparison
                    </Typography>
                    {comparisonResult.results?.map((result, index) => (
                      <Box key={index} sx={{ mb: 2, p: 1, bgcolor: 'white', borderRadius: 1 }}>
                        <Typography variant="body2" fontWeight="bold">
                          {result.model}
                        </Typography>
                        <Typography variant="body2">
                          {result.response}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Time: {result.processing_time_seconds.toFixed(2)}s
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                )}

                <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>
                  Test History ({testHistory.length})
                </Typography>
                
                {testHistory.length === 0 ? (
                  <Alert severity="info">
                    No tests have been run yet. Start by selecting a model and running a test.
                  </Alert>
                ) : (
                  <List sx={{ maxHeight: 300, overflow: 'auto' }}>
                    {testHistory.slice(0, 10).map((test, index) => (
                      <ListItem key={test.test_id || index} divider>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="body2">
                                {test.test_type === 'comparison' ? 
                                  `Comparison: ${test.models_tested?.join(', ')}` :
                                  test.test_type === 'benchmark' ? 
                                    `Benchmark: ${test.model} (${test.iterations} iterations)` :
                                    `Single Test: ${test.model}`
                                }
                              </Typography>
                              <Chip 
                                label={test.test_type} 
                                size="small" 
                                color={test.test_type === 'comparison' ? 'secondary' : 
                                       test.test_type === 'benchmark' ? 'primary' : 'default'} 
                              />
                            </Box>
                          }
                          secondary={new Date(test.created_at).toLocaleString()}
                        />
                      </ListItem>
                    ))}
                  </List>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Model Comparison */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <ViewIcon sx={{ mr: 1, verticalAlign: 'bottom' }} />
                  Model Comparison
                </Typography>
                
                <FormControl fullWidth margin="normal">
                  <InputLabel>Select Models to Compare</InputLabel>
                  <Select
                    multiple
                    value={compareModels}
                    onChange={(e) => setCompareModels(e.target.value)}
                    renderValue={(selected) => (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selected.map((value) => (
                          <Chip key={value} label={value} size="small" />
                        ))}
                      </Box>
                    )}
                  >
                    {models.map((model) => (
                      <MenuItem key={model.name} value={model.name}>
                        <Checkbox checked={compareModels.indexOf(model.name) > -1} />
                        <ListItemText primary={`${model.name} (${model.type})`} />
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Button
                  variant="contained"
                  onClick={handleCompareModels}
                  disabled={isTesting || compareModels.length < 2}
                  sx={{ mt: 2 }}
                  startIcon={isTesting ? <CircularProgress size={16} /> : <ViewIcon />}
                >
                  {isTesting ? 'Comparing...' : `Compare ${compareModels.length} Models`}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

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
    </Box>
  );
};

export default ModelTraining;