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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CardMedia,
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
  ExpandMore as ExpandMoreIcon,
  Code as CodeIcon,
  Image as ImageIcon,
  BarChart as ChartIcon,
} from '@mui/icons-material';
import { 
  PieChart, 
  Pie, 
  Cell, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { trainingAPI, testingAPI, videoAPI, wsManager } from '../services/api.jsx';
import { DATASET_TYPES, DATASET_TYPE_INFO } from '../constants/datasetTypes.js';

// Component to draw bounding boxes on images
const BoundingBoxVisualization = ({ imageUrl, detections, settings = {} }) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const containerRef = useRef(null);
  
  // Zoom and pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!imageUrl || !detections || detections.length === 0) return;

    const canvas = canvasRef.current;
    const image = imageRef.current;
    
    if (!canvas || !image) return;

    const ctx = canvas.getContext('2d');
    
    const drawBoundingBoxes = () => {
      const containerWidth = containerRef.current?.clientWidth || 800;
      const containerHeight = containerRef.current?.clientHeight || 600;
      
      // Set canvas size to container size
      canvas.width = containerWidth;
      canvas.height = containerHeight;
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Calculate scaling to fit image in container
      const scaleX = containerWidth / image.naturalWidth;
      const scaleY = containerHeight / image.naturalHeight;
      const baseScale = Math.min(scaleX, scaleY);
      
      // Apply zoom and pan transformations
      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.scale(baseScale * zoom, baseScale * zoom);
      ctx.translate(pan.x, pan.y);
      ctx.translate(-image.naturalWidth / 2, -image.naturalHeight / 2);
      
      // Draw image
      ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight);
      
      // Draw bounding boxes
      detections.forEach((detection, index) => {
        const { bbox, class: className, confidence } = detection;
        const [x, y, width, height] = bbox;
        
        // Generate color based on class index for consistency
        const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', 
                       '#ffa500', '#800080', '#008080', '#ffc0cb', '#a52a2a', '#dda0dd',
                       '#20b2aa', '#87ceeb', '#98fb98', '#f0e68c', '#deb887'];
        const color = colors[index % colors.length];
        
        // Draw bounding box outline
        ctx.strokeStyle = color;
        ctx.lineWidth = settings.outlineOnly ? 2 : 3;
        ctx.strokeRect(x, y, width, height);
        
        // Only draw filled label background if not in outline-only mode
        if (!settings.outlineOnly && settings.showLabels !== false) {
          // Draw label background
          ctx.fillStyle = color;
          const labelText = `${className} ${(confidence * 100).toFixed(1)}%`;
          const textWidth = ctx.measureText(labelText).width;
          ctx.fillRect(x, y - 25, textWidth + 10, 25);
          
          // Draw label text
          ctx.fillStyle = '#ffffff';
          ctx.font = '16px Arial';
          ctx.fillText(labelText, x + 5, y - 5);
        } else if (settings.outlineOnly && settings.showLabels !== false) {
          // In outline-only mode, draw label with outline/shadow for better visibility
          const labelText = `${className} ${(confidence * 100).toFixed(1)}%`;
          ctx.font = '16px Arial';
          
          // Draw text shadow/outline for better readability
          ctx.strokeStyle = '#000000';
          ctx.lineWidth = 3;
          ctx.strokeText(labelText, x + 5, y - 5);
          
          // Draw label text
          ctx.fillStyle = color;
          ctx.fillText(labelText, x + 5, y - 5);
        }
      });
      
      // Restore canvas transformation
      ctx.restore();
    };

    if (image.complete) {
      drawBoundingBoxes();
    } else {
      image.onload = drawBoundingBoxes;
    }
  }, [imageUrl, detections, settings, zoom, pan]);

  // Zoom and pan handlers
  const handleWheel = (e) => {
    e.preventDefault();
    const zoomSpeed = 0.1;
    const delta = e.deltaY > 0 ? -zoomSpeed : zoomSpeed;
    setZoom(prevZoom => Math.max(0.5, Math.min(5, prevZoom + delta)));
  };

  const handleMouseDown = (e) => {
    if (zoom > 1) {
      setIsPanning(true);
      const rect = e.currentTarget.getBoundingClientRect();
      setLastPanPoint({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      });
    }
  };

  const handleMouseMove = (e) => {
    if (isPanning && zoom > 1) {
      const rect = e.currentTarget.getBoundingClientRect();
      const currentPoint = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      };
      
      const deltaX = currentPoint.x - lastPanPoint.x;
      const deltaY = currentPoint.y - lastPanPoint.y;
      
      setPan(prevPan => ({
        x: prevPan.x + deltaX / zoom,
        y: prevPan.y + deltaY / zoom
      }));
      
      setLastPanPoint(currentPoint);
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handleZoomIn = () => {
    setZoom(prevZoom => Math.min(5, prevZoom + 0.5));
  };

  const handleZoomOut = () => {
    setZoom(prevZoom => Math.max(0.5, prevZoom - 0.5));
  };

  const handleResetZoom = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  if (!imageUrl || !detections || detections.length === 0) {
    return <Alert severity="info">No detections to display</Alert>;
  }

  return (
    <Box sx={{ position: 'relative', maxWidth: '100%' }}>
      {/* Zoom Controls */}
      <Box sx={{ 
        position: 'absolute', 
        top: 8, 
        right: 8, 
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
        bgcolor: 'background.paper',
        borderRadius: 1,
        boxShadow: 2,
        p: 1
      }}>
        <Button size="small" onClick={handleZoomIn} disabled={zoom >= 5}>
          🔍+
        </Button>
        <Typography variant="caption" sx={{ textAlign: 'center', minWidth: 40 }}>
          {Math.round(zoom * 100)}%
        </Typography>
        <Button size="small" onClick={handleZoomOut} disabled={zoom <= 0.5}>
          🔍-
        </Button>
        <Button size="small" onClick={handleResetZoom} disabled={zoom === 1 && pan.x === 0 && pan.y === 0}>
          ↺
        </Button>
      </Box>
      
      {/* Instructions */}
      {zoom > 1 && (
        <Box sx={{
          position: 'absolute',
          bottom: 8,
          left: 8,
          zIndex: 1000,
          bgcolor: 'background.paper',
          borderRadius: 1,
          boxShadow: 2,
          px: 1.5,
          py: 0.5
        }}>
          <Typography variant="caption" color="text.secondary">
            Drag to pan • Mouse wheel to zoom
          </Typography>
        </Box>
      )}

      <img
        ref={imageRef}
        src={imageUrl}
        alt="Test result"
        style={{ width: '100%', height: 'auto', display: 'none' }}
        crossOrigin="anonymous"
      />
      
      <div
        ref={containerRef}
        style={{ 
          width: '100%', 
          height: '500px',
          border: '1px solid #ccc',
          borderRadius: '4px',
          overflow: 'hidden',
          cursor: zoom > 1 ? (isPanning ? 'grabbing' : 'grab') : 'default'
        }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <canvas
          ref={canvasRef}
          style={{ 
            display: 'block',
            width: '100%',
            height: '100%'
          }}
        />
      </div>
    </Box>
  );
};

// Component for classification confidence bars
const ClassificationVisualization = ({ predictions }) => {
  if (!predictions || predictions.length === 0) {
    return <Alert severity="info">No predictions to display</Alert>;
  }

  return (
    <Box>
      {predictions.slice(0, 5).map((prediction, index) => (
        <Box key={index} sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
            <Typography variant="body2">{prediction.label}</Typography>
            <Typography variant="body2" color="text.secondary">
              {(prediction.confidence * 100).toFixed(1)}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={prediction.confidence * 100}
            sx={{ 
              height: 8, 
              borderRadius: 4,
              backgroundColor: 'grey.200',
              '& .MuiLinearProgress-bar': {
                backgroundColor: index === 0 ? 'success.main' : 'primary.main'
              }
            }}
          />
        </Box>
      ))}
    </Box>
  );
};

// Model Performance Analysis Component
const ModelPerformanceAnalysis = ({ testResults }) => {
  if (!testResults || testResults.length === 0) {
    return (
      <Alert severity="info">
        No test results available for performance analysis. Run some tests first.
      </Alert>
    );
  }

  // Calculate performance metrics
  const performanceMetrics = testResults.reduce((acc, result) => {
    if (!acc[result.modelName]) {
      acc[result.modelName] = {
        modelName: result.modelName,
        modelType: result.modelType,
        totalTests: 0,
        avgProcessingTime: 0,
        avgConfidence: 0,
        detectionCounts: [],
        processingTimes: [],
        confidenceScores: []
      };
    }

    const metrics = acc[result.modelName];
    metrics.totalTests++;
    
    if (result.results.processing_time_seconds) {
      metrics.processingTimes.push(result.results.processing_time_seconds);
    }

    // Object Detection metrics
    if (result.modelType === 'object_detection' && result.results.detections) {
      metrics.detectionCounts.push(result.results.detections.length);
      result.results.detections.forEach(det => {
        if (det.confidence) metrics.confidenceScores.push(det.confidence);
      });
    }

    // Classification metrics
    if (result.modelType === 'image_classification' && result.results.predictions) {
      result.results.predictions.forEach(pred => {
        if (pred.confidence) metrics.confidenceScores.push(pred.confidence);
      });
    }

    return acc;
  }, {});

  // Calculate averages
  Object.values(performanceMetrics).forEach(metrics => {
    metrics.avgProcessingTime = metrics.processingTimes.length > 0 
      ? metrics.processingTimes.reduce((a, b) => a + b, 0) / metrics.processingTimes.length 
      : 0;
    metrics.avgConfidence = metrics.confidenceScores.length > 0
      ? metrics.confidenceScores.reduce((a, b) => a + b, 0) / metrics.confidenceScores.length
      : 0;
    metrics.avgDetections = metrics.detectionCounts.length > 0
      ? metrics.detectionCounts.reduce((a, b) => a + b, 0) / metrics.detectionCounts.length
      : 0;
  });

  const modelsData = Object.values(performanceMetrics);

  // Confidence distribution data for pie chart
  const confidenceDistribution = testResults.reduce((acc, result) => {
    let confidenceScores = [];
    
    if (result.results.detections) {
      confidenceScores = result.results.detections.map(d => d.confidence);
    } else if (result.results.predictions) {
      confidenceScores = result.results.predictions.map(p => p.confidence);
    }

    confidenceScores.forEach(score => {
      if (score >= 0.9) acc.high++;
      else if (score >= 0.7) acc.medium++;
      else if (score >= 0.5) acc.low++;
      else acc.veryLow++;
    });

    return acc;
  }, { high: 0, medium: 0, low: 0, veryLow: 0 });

  const pieData = [
    { name: 'High (≥90%)', value: confidenceDistribution.high, color: '#4caf50' },
    { name: 'Medium (70-89%)', value: confidenceDistribution.medium, color: '#ff9800' },
    { name: 'Low (50-69%)', value: confidenceDistribution.low, color: '#f44336' },
    { name: 'Very Low (<50%)', value: confidenceDistribution.veryLow, color: '#757575' }
  ].filter(item => item.value > 0);

  return (
    <Grid container spacing={3}>
      {/* Overall Performance Summary */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Model Performance Summary
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center', backgroundColor: 'primary.main', color: 'white' }}>
                  <Typography variant="h4">{testResults.length}</Typography>
                  <Typography variant="body2">Total Tests</Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center', backgroundColor: 'success.main', color: 'white' }}>
                  <Typography variant="h4">{modelsData.length}</Typography>
                  <Typography variant="body2">Models Tested</Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center', backgroundColor: 'info.main', color: 'white' }}>
                  <Typography variant="h4">
                    {modelsData.reduce((sum, m) => sum + m.avgProcessingTime, 0) > 0
                      ? (modelsData.reduce((sum, m) => sum + m.avgProcessingTime, 0) / modelsData.length).toFixed(2)
                      : '0.00'
                    }s
                  </Typography>
                  <Typography variant="body2">Avg Processing Time</Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center', backgroundColor: 'warning.main', color: 'white' }}>
                  <Typography variant="h4">
                    {modelsData.reduce((sum, m) => sum + m.avgConfidence, 0) > 0
                      ? (modelsData.reduce((sum, m) => sum + m.avgConfidence, 0) / modelsData.length * 100).toFixed(1)
                      : '0.0'
                    }%
                  </Typography>
                  <Typography variant="body2">Avg Confidence</Typography>
                </Paper>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>

      {/* Processing Time Comparison */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Processing Time Comparison
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="modelName" 
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis />
                <Tooltip formatter={(value) => `${value.toFixed(3)}s`} />
                <Legend />
                <Bar 
                  dataKey="avgProcessingTime" 
                  fill="#1976d2" 
                  name="Avg Processing Time (s)"
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Confidence Distribution */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Confidence Score Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Model Comparison Table */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Detailed Model Comparison
            </Typography>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model Name</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell align="right">Tests Run</TableCell>
                  <TableCell align="right">Avg Processing Time</TableCell>
                  <TableCell align="right">Avg Confidence</TableCell>
                  <TableCell align="right">Avg Detections</TableCell>
                  <TableCell align="center">Performance Rating</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {modelsData.map((model) => {
                  // Calculate performance rating
                  const processingScore = model.avgProcessingTime < 1 ? 3 : model.avgProcessingTime < 3 ? 2 : 1;
                  const confidenceScore = model.avgConfidence > 0.8 ? 3 : model.avgConfidence > 0.6 ? 2 : 1;
                  const overallRating = Math.round((processingScore + confidenceScore) / 2);
                  
                  const getRatingColor = (rating) => {
                    switch(rating) {
                      case 3: return 'success';
                      case 2: return 'warning'; 
                      case 1: return 'error';
                      default: return 'default';
                    }
                  };

                  const getRatingText = (rating) => {
                    switch(rating) {
                      case 3: return 'Excellent';
                      case 2: return 'Good';
                      case 1: return 'Needs Improvement';
                      default: return 'Unknown';
                    }
                  };

                  return (
                    <TableRow key={model.modelName}>
                      <TableCell component="th" scope="row">
                        <Typography variant="body2" fontWeight="bold">
                          {model.modelName}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={DATASET_TYPE_INFO[model.modelType]?.name || model.modelType}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell align="right">{model.totalTests}</TableCell>
                      <TableCell align="right">
                        {model.avgProcessingTime.toFixed(3)}s
                      </TableCell>
                      <TableCell align="right">
                        {(model.avgConfidence * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell align="right">
                        {model.avgDetections ? model.avgDetections.toFixed(1) : 'N/A'}
                      </TableCell>
                      <TableCell align="center">
                        <Chip 
                          label={getRatingText(overallRating)}
                          color={getRatingColor(overallRating)}
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

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
    outlineOnly: false,
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
        // Construct full image URL if relative URL is provided
        const fullResults = {
          ...testData,
          image_url: testData.image_url ? `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${testData.image_url}` : null
        };
        
        setTestResults(prev => [{
          id: Date.now(),
          modelName: selectedModel,
          modelType: 'object_detection',
          inputType,
          timestamp: new Date(),
          results: fullResults,
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
        // Construct full image URL if relative URL is provided
        const fullResults = {
          ...testData,
          image_url: testData.image_url ? `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${testData.image_url}` : null
        };
        
        setTestResults(prev => [{
          id: Date.now(),
          modelName: selectedModel,
          modelType: 'image_classification',
          inputType,
          timestamp: new Date(),
          results: fullResults,
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
        // Construct full image URL if relative URL is provided
        const fullResults = {
          ...testData,
          image_url: testData.image_url ? `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${testData.image_url}` : null
        };
        
        setTestResults(prev => [{
          id: Date.now(),
          modelName: selectedModel,
          modelType: 'vision_llm',
          inputType,
          timestamp: new Date(),
          results: fullResults,
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
            <Tab label="Performance Analysis" />
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
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={objectDetectionSettings.outlineOnly}
                          onChange={(e) => setObjectDetectionSettings(prev => ({
                            ...prev,
                            outlineOnly: e.target.checked
                          }))}
                        />
                      }
                      label={
                        <Box>
                          <Typography variant="body2">Outline Only Mode</Typography>
                          <Typography variant="caption" color="text.secondary">
                            Show only colored outlines without filled backgrounds for better visibility of overlapping detections
                          </Typography>
                        </Box>
                      }
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
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {testResults.map((result) => (
                      <Accordion key={result.id}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', mr: 2 }}>
                            <Box>
                              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                                {result.modelName}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {DATASET_TYPE_INFO[result.modelType]?.name || result.modelType} • 
                                Input: {result.inputType} • 
                                {result.timestamp.toLocaleString()}
                              </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              {result.results.processing_time_seconds && (
                                <Chip 
                                  label={`${result.results.processing_time_seconds.toFixed(2)}s`} 
                                  size="small" 
                                  color="primary" 
                                  icon={<ChartIcon />}
                                />
                              )}
                              {result.modelType === 'object_detection' && result.results.detections && (
                                <Chip 
                                  label={`${result.results.detections.length} objects`} 
                                  size="small" 
                                  color="success"
                                />
                              )}
                            </Box>
                          </Box>
                        </AccordionSummary>
                        
                        <AccordionDetails>
                          <Grid container spacing={3}>
                            {/* Visual Results Column */}
                            <Grid item xs={12} md={6}>
                              <Typography variant="h6" gutterBottom>
                                <ImageIcon sx={{ mr: 1, verticalAlign: 'bottom' }} />
                                Visual Results
                              </Typography>
                              
                              {result.modelType === 'object_detection' && result.results.detections && (
                                <BoundingBoxVisualization 
                                  imageUrl={result.results.image_url}
                                  detections={result.results.detections}
                                  settings={objectDetectionSettings}
                                />
                              )}
                              
                              {result.modelType === 'image_classification' && result.results.predictions && (
                                <Box>
                                  {result.results.image_url && (
                                    <CardMedia
                                      component="img"
                                      sx={{ 
                                        maxHeight: 300, 
                                        objectFit: 'contain',
                                        border: '1px solid #ccc',
                                        borderRadius: 1,
                                        mb: 2
                                      }}
                                      image={result.results.image_url}
                                      alt="Test image"
                                    />
                                  )}
                                  <ClassificationVisualization predictions={result.results.predictions} />
                                </Box>
                              )}
                              
                              {result.modelType === 'vision_llm' && (
                                <Box>
                                  {result.results.image_url && (
                                    <CardMedia
                                      component="img"
                                      sx={{ 
                                        maxHeight: 300, 
                                        objectFit: 'contain',
                                        border: '1px solid #ccc',
                                        borderRadius: 1,
                                        mb: 2
                                      }}
                                      image={result.results.image_url}
                                      alt="Test image"
                                    />
                                  )}
                                  {result.results.response && (
                                    <Paper sx={{ p: 2, backgroundColor: 'grey.50' }}>
                                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                                        {result.results.response}
                                      </Typography>
                                    </Paper>
                                  )}
                                </Box>
                              )}
                            </Grid>
                            
                            {/* Metadata Column */}
                            <Grid item xs={12} md={6}>
                              <Typography variant="h6" gutterBottom>
                                <CodeIcon sx={{ mr: 1, verticalAlign: 'bottom' }} />
                                Metadata & Results
                              </Typography>
                              
                              {/* Performance Metrics */}
                              <Box sx={{ mb: 2 }}>
                                <Typography variant="subtitle2" color="primary" gutterBottom>
                                  Performance Metrics
                                </Typography>
                                <Table size="small">
                                  <TableBody>
                                    {result.results.processing_time_seconds && (
                                      <TableRow>
                                        <TableCell>Processing Time</TableCell>
                                        <TableCell>{result.results.processing_time_seconds.toFixed(3)}s</TableCell>
                                      </TableRow>
                                    )}
                                    {result.modelType === 'object_detection' && result.results.detections && (
                                      <>
                                        <TableRow>
                                          <TableCell>Objects Detected</TableCell>
                                          <TableCell>{result.results.detections.length}</TableCell>
                                        </TableRow>
                                        {result.results.detections.length > 0 && (
                                          <TableRow>
                                            <TableCell>Avg Confidence</TableCell>
                                            <TableCell>
                                              {(result.results.detections.reduce((sum, det) => sum + det.confidence, 0) / result.results.detections.length * 100).toFixed(1)}%
                                            </TableCell>
                                          </TableRow>
                                        )}
                                      </>
                                    )}
                                    {result.modelType === 'image_classification' && result.results.predictions && result.results.predictions.length > 0 && (
                                      <TableRow>
                                        <TableCell>Top Prediction Confidence</TableCell>
                                        <TableCell>{(result.results.predictions[0].confidence * 100).toFixed(1)}%</TableCell>
                                      </TableRow>
                                    )}
                                  </TableBody>
                                </Table>
                              </Box>
                              
                              {/* Detailed Results */}
                              <Box>
                                <Typography variant="subtitle2" color="primary" gutterBottom>
                                  Detailed Results (JSON)
                                </Typography>
                                <Paper sx={{ 
                                  p: 2, 
                                  backgroundColor: 'grey.900', 
                                  color: 'grey.100',
                                  maxHeight: 400,
                                  overflow: 'auto',
                                  fontFamily: 'monospace',
                                  fontSize: '0.875rem'
                                }}>
                                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                    {JSON.stringify(result.results, null, 2)}
                                  </pre>
                                </Paper>
                              </Box>
                            </Grid>
                          </Grid>
                        </AccordionDetails>
                      </Accordion>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          )}

          {/* Performance Analysis Tab */}
          {activeTab === 2 && (
            <ModelPerformanceAnalysis testResults={testResults} />
          )}

          {/* History Tab */}
          {activeTab === 3 && (
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