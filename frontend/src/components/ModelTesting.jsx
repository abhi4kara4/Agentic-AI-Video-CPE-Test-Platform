import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Tabs,
  Tab,
  LinearProgress,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Switch,
  FormControlLabel,
  Divider,
  Paper,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
} from '@mui/material';
import {
  Science as TestingIcon,
  Speed as BenchmarkIcon,
  Compare as CompareIcon,
  CloudUpload as UploadIcon,
  PhotoCamera as CameraIcon,
  Videocam as VideoIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  Assessment as AnalyticsIcon,
} from '@mui/icons-material';
import { trainingAPI, testingAPI, videoAPI, wsManager } from '../services/api.jsx';
import { DATASET_TYPES, DATASET_TYPE_INFO } from '../constants/datasetTypes.js';

const ModelTesting = ({ onNotification }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedModelType, setSelectedModelType] = useState('');
  const [testHistory, setTestHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  
  // Input source states
  const [inputType, setInputType] = useState('stream'); // 'stream', 'upload', 'video'
  const [uploadedImages, setUploadedImages] = useState([]);
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [testResults, setTestResults] = useState([]);
  const [isLiveTesting, setIsLiveTesting] = useState(false);
  
  // Model-specific states
  const [objectDetectionSettings, setObjectDetectionSettings] = useState({
    confidenceThreshold: 0.5,
    showBoundingBoxes: true,
    showLabels: true,
    showConfidence: true,
  });
  
  const [classificationSettings, setClassificationSettings] = useState({
    topK: 5,
    showProbabilities: true,
  });
  
  const [visionLLMSettings, setVisionLLMSettings] = useState({
    prompt: 'Describe what you see on this TV/STB screen. What is the current state and what UI elements are visible?',
    temperature: 0.7,
    maxTokens: 200,
  });
  
  // File input refs
  const imageUploadRef = useRef(null);
  const videoUploadRef = useRef(null);

  useEffect(() => {
    loadModels();
    loadTestHistory();
  }, []);

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

  const handleModelSelect = (modelName) => {
    const model = models.find(m => m.name === modelName);
    setSelectedModel(modelName);
    setSelectedModelType(model?.type || 'object_detection');
    
    // Set appropriate tab based on model type
    if (model?.type === 'object_detection') {
      setActiveTab(0);
    } else if (model?.type === 'image_classification') {
      setActiveTab(1);
    } else if (model?.type === 'vision_llm') {
      setActiveTab(2);
    }
  };

  const handleImageUpload = (event) => {
    const files = Array.from(event.target.files);
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    const imagePromises = imageFiles.map(file => {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve({
          file,
          name: file.name,
          url: e.target.result,
          size: file.size
        });
        reader.readAsDataURL(file);
      });
    });

    Promise.all(imagePromises).then(images => {
      setUploadedImages(prev => [...prev, ...images]);
    });
  };

  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedVideo({
          file,
          name: file.name,
          url: e.target.result,
          size: file.size
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const runObjectDetectionTest = async () => {
    if (!selectedModel) {
      onNotification({
        type: 'warning',
        title: 'No Model Selected',
        message: 'Please select an object detection model to test'
      });
      return;
    }

    if (inputType === 'upload' && uploadedImages.length === 0) {
      onNotification({
        type: 'warning',
        title: 'No Images Uploaded',
        message: 'Please upload at least one image to test'
      });
      return;
    }

    setLoading(true);
    try {
      let testData = null;
      
      if (inputType === 'stream') {
        // Test with current video stream
        const response = await testingAPI.testModel(selectedModel);
        testData = response.data;
      } else if (inputType === 'upload' && uploadedImages.length > 0) {
        // Test with first uploaded image
        const response = await testingAPI.testModelWithUpload(
          selectedModel, 
          uploadedImages[0].file
        );
        testData = response.data;
      } else if (inputType === 'video' && uploadedVideo) {
        // For video, use stream endpoint (could be extended to process video frames)
        const response = await testingAPI.testModel(selectedModel);
        testData = response.data;
      }
      
      if (testData) {
        setTestResults(prev => [{
          id: Date.now(),
          modelName: selectedModel,
          modelType: 'object_detection',
          inputType,
          timestamp: new Date(),
          results: testData,
          settings: objectDetectionSettings
        }, ...prev]);
        
        onNotification({
          type: 'success',
          title: 'Object Detection Complete',
          message: `Found ${testData.detections?.length || 0} objects`
        });
      }
      
      await loadTestHistory();
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Test Failed',
        message: error.response?.data?.detail || 'Failed to run object detection test'
      });
    } finally {
      setLoading(false);
    }
  };

  const runClassificationTest = async () => {
    if (!selectedModel) {
      onNotification({
        type: 'warning',
        title: 'No Model Selected',
        message: 'Please select a classification model to test'
      });
      return;
    }

    if (inputType === 'upload' && uploadedImages.length === 0) {
      onNotification({
        type: 'warning',
        title: 'No Images Uploaded',
        message: 'Please upload at least one image to test'
      });
      return;
    }

    setLoading(true);
    try {
      let testData = null;
      
      if (inputType === 'stream') {
        const response = await testingAPI.testModel(selectedModel);
        testData = response.data;
      } else if (inputType === 'upload' && uploadedImages.length > 0) {
        const response = await testingAPI.testModelWithUpload(
          selectedModel, 
          uploadedImages[0].file
        );
        testData = response.data;
      } else if (inputType === 'video' && uploadedVideo) {
        const response = await testingAPI.testModel(selectedModel);
        testData = response.data;
      }
      
      if (testData) {
        setTestResults(prev => [{
          id: Date.now(),
          modelName: selectedModel,
          modelType: 'image_classification',
          inputType,
          timestamp: new Date(),
          results: testData,
          settings: classificationSettings
        }, ...prev]);
        
        onNotification({
          type: 'success',
          title: 'Classification Complete',
          message: `Top prediction: ${testData.predictions?.[0]?.label || testData.top_prediction?.label || 'Unknown'}`
        });
      }
      
      await loadTestHistory();
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Test Failed',
        message: error.response?.data?.detail || 'Failed to run classification test'
      });
    } finally {
      setLoading(false);
    }
  };

  const runVisionLLMTest = async () => {
    if (!selectedModel) {
      onNotification({
        type: 'warning',
        title: 'No Model Selected',
        message: 'Please select a vision LLM model to test'
      });
      return;
    }

    if (inputType === 'upload' && uploadedImages.length === 0) {
      onNotification({
        type: 'warning',
        title: 'No Images Uploaded',
        message: 'Please upload at least one image to test'
      });
      return;
    }

    setLoading(true);
    try {
      let testData = null;
      
      if (inputType === 'stream') {
        const response = await testingAPI.testModel(selectedModel, visionLLMSettings.prompt);
        testData = response.data;
      } else if (inputType === 'upload' && uploadedImages.length > 0) {
        const response = await testingAPI.testModelWithUpload(
          selectedModel, 
          uploadedImages[0].file,
          visionLLMSettings.prompt
        );
        testData = response.data;
      } else if (inputType === 'video' && uploadedVideo) {
        const response = await testingAPI.testModel(selectedModel, visionLLMSettings.prompt);
        testData = response.data;
      }
      
      if (testData) {
        setTestResults(prev => [{
          id: Date.now(),
          modelName: selectedModel,
          modelType: 'vision_llm',
          inputType,
          timestamp: new Date(),
          results: testData,
          settings: visionLLMSettings
        }, ...prev]);
        
        onNotification({
          type: 'success',
          title: 'Vision LLM Analysis Complete',
          message: 'Generated response successfully'
        });
      }
      
      await loadTestHistory();
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Test Failed',
        message: error.response?.data?.detail || 'Failed to run vision LLM test'
      });
    } finally {
      setLoading(false);
    }
  };

  const startLiveTesting = async () => {
    setIsLiveTesting(true);
    // This would start continuous testing on the video stream
    onNotification({
      type: 'info',
      title: 'Live Testing Started',
      message: 'Testing will run continuously on the video stream'
    });
  };

  const stopLiveTesting = async () => {
    setIsLiveTesting(false);
    onNotification({
      type: 'info',
      title: 'Live Testing Stopped',
      message: 'Continuous testing has been stopped'
    });
  };

  const clearResults = () => {
    setTestResults([]);
    onNotification({
      type: 'info',
      title: 'Results Cleared',
      message: 'Test results have been cleared'
    });
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Model Testing
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Test trained models with images, videos, and live streams
        </Typography>
      </Box>

      {/* Model Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Model Selection
          </Typography>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Select Model to Test</InputLabel>
                <Select
                  value={selectedModel}
                  onChange={(e) => handleModelSelect(e.target.value)}
                >
                  {models.map((model) => (
                    <MenuItem key={model.name} value={model.name}>
                      {model.name} ({DATASET_TYPE_INFO[model.type]?.name || model.type})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            {selectedModel && (
              <Grid item xs={12} md={6}>
                <Alert severity="info">
                  Selected: <strong>{selectedModel}</strong> - {DATASET_TYPE_INFO[selectedModelType]?.name || selectedModelType}
                </Alert>
              </Grid>
            )}
          </Grid>
        </CardContent>
      </Card>

      {/* Input Source Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Input Source
          </Typography>
          <Grid container spacing={2}>
            <Grid item>
              <Button
                variant={inputType === 'stream' ? 'contained' : 'outlined'}
                startIcon={<VideoIcon />}
                onClick={() => setInputType('stream')}
              >
                Live Stream
              </Button>
            </Grid>
            <Grid item>
              <Button
                variant={inputType === 'upload' ? 'contained' : 'outlined'}
                startIcon={<UploadIcon />}
                onClick={() => setInputType('upload')}
              >
                Upload Images
              </Button>
            </Grid>
            <Grid item>
              <Button
                variant={inputType === 'video' ? 'contained' : 'outlined'}
                startIcon={<CameraIcon />}
                onClick={() => setInputType('video')}
              >
                Upload Video
              </Button>
            </Grid>
          </Grid>

          {/* Input Type Specific Controls */}
          {inputType === 'upload' && (
            <Box sx={{ mt: 2 }}>
              <input
                type="file"
                multiple
                accept="image/*"
                onChange={handleImageUpload}
                style={{ display: 'none' }}
                ref={imageUploadRef}
              />
              <Button
                variant="outlined"
                startIcon={<UploadIcon />}
                onClick={() => imageUploadRef.current?.click()}
              >
                Select Images
              </Button>
              {uploadedImages.length > 0 && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {uploadedImages.length} image(s) selected
                </Typography>
              )}
            </Box>
          )}

          {inputType === 'video' && (
            <Box sx={{ mt: 2 }}>
              <input
                type="file"
                accept="video/*"
                onChange={handleVideoUpload}
                style={{ display: 'none' }}
                ref={videoUploadRef}
              />
              <Button
                variant="outlined"
                startIcon={<CameraIcon />}
                onClick={() => videoUploadRef.current?.click()}
              >
                Select Video
              </Button>
              {uploadedVideo && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Video: {uploadedVideo.name}
                </Typography>
              )}
            </Box>
          )}
        </CardContent>
      </Card>

      {!selectedModel ? (
        <Alert severity="info">
          Please select a model to start testing
        </Alert>
      ) : (
        <>
          {/* Model Type Specific Testing Interfaces */}
          <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)} sx={{ mb: 3 }}>
            {selectedModelType === 'object_detection' && <Tab label="Object Detection" />}
            {selectedModelType === 'image_classification' && <Tab label="Image Classification" />}
            {selectedModelType === 'vision_llm' && <Tab label="Vision LLM" />}
            <Tab label="Results" />
            <Tab label="History" />
          </Tabs>

          {/* Object Detection Testing */}
          {activeTab === 0 && selectedModelType === 'object_detection' && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Object Detection Settings
                    </Typography>
                    
                    <TextField
                      fullWidth
                      label="Confidence Threshold"
                      type="number"
                      value={objectDetectionSettings.confidenceThreshold}
                      onChange={(e) => setObjectDetectionSettings(prev => ({
                        ...prev,
                        confidenceThreshold: parseFloat(e.target.value)
                      }))}
                      inputProps={{ min: 0, max: 1, step: 0.1 }}
                      margin="normal"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={objectDetectionSettings.showBoundingBoxes}
                          onChange={(e) => setObjectDetectionSettings(prev => ({
                            ...prev,
                            showBoundingBoxes: e.target.checked
                          }))}
                        />
                      }
                      label="Show Bounding Boxes"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={objectDetectionSettings.showLabels}
                          onChange={(e) => setObjectDetectionSettings(prev => ({
                            ...prev,
                            showLabels: e.target.checked
                          }))}
                        />
                      }
                      label="Show Labels"
                    />
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Run Detection
                    </Typography>
                    
                    <Button
                      fullWidth
                      variant="contained"
                      startIcon={loading ? <CircularProgress size={16} /> : <TestingIcon />}
                      onClick={runObjectDetectionTest}
                      disabled={loading || !selectedModel}
                      sx={{ mb: 2 }}
                    >
                      {loading ? 'Detecting...' : 'Run Object Detection'}
                    </Button>
                    
                    {inputType === 'stream' && (
                      <>
                        <Button
                          fullWidth
                          variant={isLiveTesting ? 'contained' : 'outlined'}
                          color={isLiveTesting ? 'error' : 'primary'}
                          startIcon={isLiveTesting ? <StopIcon /> : <PlayIcon />}
                          onClick={isLiveTesting ? stopLiveTesting : startLiveTesting}
                          disabled={loading || !selectedModel}
                        >
                          {isLiveTesting ? 'Stop Live Testing' : 'Start Live Testing'}
                        </Button>
                      </>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {/* Image Classification Testing */}
          {activeTab === 0 && selectedModelType === 'image_classification' && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Classification Settings
                    </Typography>
                    
                    <TextField
                      fullWidth
                      label="Top K Predictions"
                      type="number"
                      value={classificationSettings.topK}
                      onChange={(e) => setClassificationSettings(prev => ({
                        ...prev,
                        topK: parseInt(e.target.value)
                      }))}
                      inputProps={{ min: 1, max: 10 }}
                      margin="normal"
                    />
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={classificationSettings.showProbabilities}
                          onChange={(e) => setClassificationSettings(prev => ({
                            ...prev,
                            showProbabilities: e.target.checked
                          }))}
                        />
                      }
                      label="Show Probabilities"
                    />
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Run Classification
                    </Typography>
                    
                    <Button
                      fullWidth
                      variant="contained"
                      startIcon={loading ? <CircularProgress size={16} /> : <TestingIcon />}
                      onClick={runClassificationTest}
                      disabled={loading || !selectedModel}
                      sx={{ mb: 2 }}
                    >
                      {loading ? 'Classifying...' : 'Run Classification'}
                    </Button>
                    
                    {inputType === 'stream' && (
                      <Button
                        fullWidth
                        variant={isLiveTesting ? 'contained' : 'outlined'}
                        color={isLiveTesting ? 'error' : 'primary'}
                        startIcon={isLiveTesting ? <StopIcon /> : <PlayIcon />}
                        onClick={isLiveTesting ? stopLiveTesting : startLiveTesting}
                        disabled={loading || !selectedModel}
                      >
                        {isLiveTesting ? 'Stop Live Testing' : 'Start Live Testing'}
                      </Button>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {/* Vision LLM Testing */}
          {activeTab === 0 && selectedModelType === 'vision_llm' && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Vision LLM Settings
                    </Typography>
                    
                    <TextField
                      fullWidth
                      label="Prompt"
                      value={visionLLMSettings.prompt}
                      onChange={(e) => setVisionLLMSettings(prev => ({
                        ...prev,
                        prompt: e.target.value
                      }))}
                      multiline
                      rows={4}
                      margin="normal"
                    />
                    
                    <TextField
                      fullWidth
                      label="Temperature"
                      type="number"
                      value={visionLLMSettings.temperature}
                      onChange={(e) => setVisionLLMSettings(prev => ({
                        ...prev,
                        temperature: parseFloat(e.target.value)
                      }))}
                      inputProps={{ min: 0, max: 2, step: 0.1 }}
                      margin="normal"
                    />
                    
                    <TextField
                      fullWidth
                      label="Max Tokens"
                      type="number"
                      value={visionLLMSettings.maxTokens}
                      onChange={(e) => setVisionLLMSettings(prev => ({
                        ...prev,
                        maxTokens: parseInt(e.target.value)
                      }))}
                      inputProps={{ min: 50, max: 500 }}
                      margin="normal"
                    />
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Run Analysis
                    </Typography>
                    
                    <Button
                      fullWidth
                      variant="contained"
                      startIcon={loading ? <CircularProgress size={16} /> : <TestingIcon />}
                      onClick={runVisionLLMTest}
                      disabled={loading || !selectedModel}
                      sx={{ mb: 2 }}
                    >
                      {loading ? 'Analyzing...' : 'Analyze with Vision LLM'}
                    </Button>
                    
                    {inputType === 'stream' && (
                      <Button
                        fullWidth
                        variant={isLiveTesting ? 'contained' : 'outlined'}
                        color={isLiveTesting ? 'error' : 'primary'}
                        startIcon={isLiveTesting ? <StopIcon /> : <PlayIcon />}
                        onClick={isLiveTesting ? stopLiveTesting : startLiveTesting}
                        disabled={loading || !selectedModel}
                      >
                        {isLiveTesting ? 'Stop Live Testing' : 'Start Live Testing'}
                      </Button>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {/* Results Tab */}
          {(activeTab === 1 || (activeTab === 0 && !['object_detection', 'image_classification', 'vision_llm'].includes(selectedModelType))) && (
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Test Results ({testResults.length})
                  </Typography>
                  <Button
                    startIcon={<DeleteIcon />}
                    onClick={clearResults}
                    color="warning"
                    size="small"
                  >
                    Clear Results
                  </Button>
                </Box>
                
                {testResults.length === 0 ? (
                  <Alert severity="info">
                    No test results yet. Run a test to see results here.
                  </Alert>
                ) : (
                  <List>
                    {testResults.map((result) => (
                      <ListItem key={result.id} divider>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="subtitle1">
                                {result.modelName} - {result.modelType}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {result.timestamp.toLocaleString()}
                              </Typography>
                            </Box>
                          }
                          secondary={
                            <Box sx={{ mt: 1 }}>
                              <Typography variant="body2" color="text.secondary">
                                Input: {result.inputType} | 
                                Processing Time: {result.results.processing_time_seconds?.toFixed(2) || 'N/A'}s
                              </Typography>
                              {result.results.response && (
                                <Typography variant="body2" sx={{ mt: 1 }}>
                                  {result.results.response}
                                </Typography>
                              )}
                            </Box>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                )}
              </CardContent>
            </Card>
          )}

          {/* History Tab */}
          {activeTab === 2 && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Test History
                </Typography>
                
                {testHistory.length === 0 ? (
                  <Alert severity="info">
                    No test history available.
                  </Alert>
                ) : (
                  <List>
                    {testHistory.slice(0, 20).map((test, index) => (
                      <ListItem key={test.test_id || index} divider>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="body2">
                                {test.test_type === 'comparison' ? 
                                  `Comparison: ${test.models_tested?.join(', ')}` :
                                  test.test_type === 'benchmark' ? 
                                    `Benchmark: ${test.model}` :
                                    `Test: ${test.model}`
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
          )}
        </>
      )}
    </Box>
  );
};

export default ModelTesting;