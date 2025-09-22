import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
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
  Slider,
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
      
      // Create class-to-color mapping for consistent colors
      const classColors = {};
      const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', 
                     '#ffa500', '#800080', '#008080', '#ffc0cb', '#a52a2a', '#dda0dd',
                     '#20b2aa', '#87ceeb', '#98fb98', '#f0e68c', '#deb887'];
      
      // Assign colors to classes if using class-based coloring
      if (settings.colorByClass) {
        const uniqueClasses = [...new Set(detections.map(d => d.class))];
        uniqueClasses.forEach((className, index) => {
          classColors[className] = colors[index % colors.length];
        });
      }
      
      // Draw bounding boxes
      detections.forEach((detection, index) => {
        const { bbox, class: className, confidence } = detection;
        const [x, y, width, height] = bbox;
        
        // Choose color based on settings
        const color = settings.colorByClass 
          ? classColors[className] 
          : colors[index % colors.length];
        
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
  const location = useLocation();
  const [activeTab, setActiveTab] = useState(0);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedModelType, setSelectedModelType] = useState('');
  const [testHistory, setTestHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [testingProgress, setTestingProgress] = useState({ current: 0, total: 0 });
  
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
    colorByClass: true, // New option for class-based coloring
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
  
  const [paddleOCRSettings, setPaddleOCRSettings] = useState({
    language: 'en',
    taskType: 'det', // 'det', 'rec', or 'both'
    confidenceThreshold: 0.5,
    showBoundingBoxes: true,
    showText: true
  });
  
  // Video analysis settings
  const [videoAnalysisSettings, setVideoAnalysisSettings] = useState({
    frameSkipFrequency: 30,
    maxFramesToAnalyze: 10,
    selectedClasses: [],
    generateAnnotatedVideo: false,
  });

  // Model classes for filtering
  const [availableClasses, setAvailableClasses] = useState([]);
  const [videoMetadata, setVideoMetadata] = useState(null);
  
  // File input refs
  const imageUploadRef = useRef(null);
  const videoUploadRef = useRef(null);

  useEffect(() => {
    loadModels();
    loadTestHistory();
  }, []);

  // Handle model passed from navigation (from ModelTraining component)
  useEffect(() => {
    const passedModel = location.state?.selectedModel;
    if (passedModel && models.length > 0) {
      // Check if the passed model exists in the loaded models list
      const modelExists = models.find(m => m.name === passedModel.name);
      if (modelExists) {
        handleModelSelect(passedModel.name);
        
        // Set appropriate settings for PaddleOCR models
        if (passedModel.type === 'paddleocr') {
          setPaddleOCRSettings(prev => ({
            ...prev,
            language: passedModel.language || 'en',
            taskType: passedModel.trainType || 'det'
          }));
          setActiveTab(3); // PaddleOCR tab
        }
        
        // Show notification about the selected model
        if (onNotification) {
          onNotification({
            type: 'success',
            title: 'Model Selected',
            message: `${passedModel.name} has been selected for testing`
          });
        }
        
        // Clear the navigation state to prevent re-triggering
        window.history.replaceState({}, document.title);
      }
    }
  }, [models, location.state]); // Re-run when models are loaded or navigation state changes

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

  const loadAvailableClasses = async (modelName) => {
    try {
      const response = await trainingAPI.getModelClasses(modelName);
      setAvailableClasses(response.data?.classes || []);
    } catch (error) {
      console.error('Failed to load model classes:', error);
      setAvailableClasses([]);
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
    } else if (model?.type === 'paddleocr') {
      setActiveTab(3);
      
      // Set PaddleOCR settings from model info if available
      if (model.paddleocr_info) {
        setPaddleOCRSettings(prev => ({
          ...prev,
          language: model.paddleocr_info.language || 'en',
          taskType: model.paddleocr_info.train_type === 'det' ? 'det' : 
                   model.paddleocr_info.train_type === 'rec' ? 'rec' : 'both'
        }));
      }
    }

    // Load available classes for object detection models
    if (model?.type === 'object_detection') {
      loadAvailableClasses(modelName);
    }

    // Reset video analysis settings when model changes
    setVideoAnalysisSettings(prev => ({
      ...prev,
      selectedClasses: []
    }));
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
        // Create video element to extract metadata
        const videoElement = document.createElement('video');
        videoElement.src = e.target.result;
        
        videoElement.onloadedmetadata = () => {
          const metadata = {
            duration: videoElement.duration,
            width: videoElement.videoWidth,
            height: videoElement.videoHeight,
            fps: null, // FPS extraction requires more complex logic
          };
          
          // Try to estimate FPS (this is an approximation)
          // In a real scenario, you might need a library like ffprobe.js
          if (metadata.duration > 0) {
            // Estimate based on common frame rates
            const estimatedFrameCount = Math.round(metadata.duration * 30); // Assume 30 FPS
            metadata.estimatedFPS = 30;
          }
          
          setVideoMetadata(metadata);
          setUploadedVideo({
            file,
            name: file.name,
            url: e.target.result,
            size: file.size,
            duration: metadata.duration,
            width: metadata.width,
            height: metadata.height,
            estimatedFPS: metadata.estimatedFPS
          });
        };
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
        // Test with all uploaded images
        const allResults = [];
        setTestingProgress({ current: 0, total: uploadedImages.length });
        
        for (let i = 0; i < uploadedImages.length; i++) {
          const image = uploadedImages[i];
          setTestingProgress({ current: i + 1, total: uploadedImages.length });
          
          try {
            const response = await testingAPI.testModelWithUpload(
              selectedModel, 
              image.file
            );
            allResults.push({
              ...response.data,
              image_name: image.name,
              image_index: i
            });
          } catch (error) {
            console.error(`Error testing image ${image.name}:`, error);
            allResults.push({
              error: `Failed to test ${image.name}: ${error.message}`,
              image_name: image.name,
              image_index: i
            });
          }
        }
        
        setTestingProgress({ current: 0, total: 0 });
        
        // Combine results or use the first successful one for backward compatibility
        if (allResults.length === 1) {
          testData = allResults[0];
        } else {
          // For multiple images, create a combined result
          testData = {
            model_name: selectedModel,
            results: allResults,
            total_images: uploadedImages.length,
            successful_tests: allResults.filter(r => !r.error).length
          };
        }
      } else if (inputType === 'video' && uploadedVideo) {
        // Debug: Log current video analysis settings
        console.log('Current videoAnalysisSettings:', videoAnalysisSettings);
        
        // For video, use the video analysis endpoint with current settings
        const videoOptions = {
          skipFrequency: videoAnalysisSettings.frameSkipFrequency,
          maxFrames: videoAnalysisSettings.maxFramesToAnalyze,
          selectedClasses: videoAnalysisSettings.selectedClasses.join(','),
          generateVideo: videoAnalysisSettings.generateAnnotatedVideo,
          prompt: "Analyze TV/STB screen objects"
        };
        
        console.log('Video options being sent:', videoOptions);
        
        try {
          // Try async analysis first
          await handleAsyncVideoAnalysis(selectedModel, uploadedVideo.file, videoOptions, 'object_detection');
          return; // Exit early - async handler will manage the results
        } catch (error) {
          // Fallback to synchronous analysis
          console.log('Falling back to synchronous analysis');
          const response = await testingAPI.testModelWithVideo(
            selectedModel,
            uploadedVideo.file,
            videoOptions
          );
          testData = response.data;
        }
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
        
        // Different notification based on input type
        if (inputType === 'video') {
          const totalDetections = testData.summary?.total_detections || 0;
          const framesAnalyzed = testData.video_info?.analyzed_frames || 0;
          onNotification({
            type: 'success',
            title: 'Video Analysis Complete',
            message: `Analyzed ${framesAnalyzed} frames, found ${totalDetections} objects total`
          });
        } else {
          onNotification({
            type: 'success',
            title: 'Object Detection Complete',
            message: `Found ${testData.detections?.length || 0} objects`
          });
        }
      }
      
      await loadTestHistory();
    } catch (error) {
      console.error('Object detection test error:', error);
      
      // Handle different error types
      let errorMessage = 'Failed to run object detection test';
      let errorTitle = 'Test Failed';
      
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        errorTitle = 'Test Timeout';
        errorMessage = 'Video analysis is taking longer than expected. The analysis may still be running on the server. Check the History tab in a few minutes for results.';
      } else if (error.response?.status === 500) {
        errorMessage = error.response?.data?.detail || 'Server error during analysis';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      onNotification({
        type: error.code === 'ECONNABORTED' ? 'warning' : 'error',
        title: errorTitle,
        message: errorMessage
      });
    } finally {
      setLoading(false);
      
      // If there was a timeout, suggest checking history
      if (inputType === 'video') {
        setTimeout(() => {
          loadTestHistory(); // Refresh history in case results appeared
        }, 2000);
      }
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
        // Test with all uploaded images
        const allResults = [];
        for (let i = 0; i < uploadedImages.length; i++) {
          const image = uploadedImages[i];
          try {
            const response = await testingAPI.testModelWithUpload(
              selectedModel, 
              image.file
            );
            allResults.push({
              ...response.data,
              image_name: image.name,
              image_index: i
            });
          } catch (error) {
            console.error(`Error testing image ${image.name}:`, error);
            allResults.push({
              error: `Failed to test ${image.name}: ${error.message}`,
              image_name: image.name,
              image_index: i
            });
          }
        }
        
        // Combine results
        if (allResults.length === 1) {
          testData = allResults[0];
        } else {
          testData = {
            model_name: selectedModel,
            results: allResults,
            total_images: uploadedImages.length,
            successful_tests: allResults.filter(r => !r.error).length
          };
        }
      } else if (inputType === 'video' && uploadedVideo) {
        // Debug: Log current video analysis settings
        console.log('Current videoAnalysisSettings:', videoAnalysisSettings);
        
        // For video, use the video analysis endpoint with current settings
        const videoOptions = {
          skipFrequency: videoAnalysisSettings.frameSkipFrequency,
          maxFrames: videoAnalysisSettings.maxFramesToAnalyze,
          selectedClasses: videoAnalysisSettings.selectedClasses.join(','),
          generateVideo: videoAnalysisSettings.generateAnnotatedVideo,
          prompt: "Analyze TV/STB screen objects"
        };
        
        console.log('Video options being sent:', videoOptions);
        
        try {
          // Try async analysis first
          await handleAsyncVideoAnalysis(selectedModel, uploadedVideo.file, videoOptions, 'image_classification');
          return; // Exit early - async handler will manage the results
        } catch (error) {
          // Fallback to synchronous analysis
          console.log('Falling back to synchronous analysis');
          const response = await testingAPI.testModelWithVideo(
            selectedModel,
            uploadedVideo.file,
            videoOptions
          );
          testData = response.data;
        }
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
        // Test with all uploaded images
        const allResults = [];
        for (let i = 0; i < uploadedImages.length; i++) {
          const image = uploadedImages[i];
          try {
            const response = await testingAPI.testModelWithUpload(
              selectedModel, 
              image.file,
              visionLLMSettings.prompt
            );
            allResults.push({
              ...response.data,
              image_name: image.name,
              image_index: i
            });
          } catch (error) {
            console.error(`Error testing image ${image.name}:`, error);
            allResults.push({
              error: `Failed to test ${image.name}: ${error.message}`,
              image_name: image.name,
              image_index: i
            });
          }
        }
        
        // Combine results
        if (allResults.length === 1) {
          testData = allResults[0];
        } else {
          testData = {
            model_name: selectedModel,
            results: allResults,
            total_images: uploadedImages.length,
            successful_tests: allResults.filter(r => !r.error).length
          };
        }
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

  const runPaddleOCRTest = async () => {
    if (!selectedModel) {
      onNotification({
        type: 'warning',
        title: 'No Model Selected',
        message: 'Please select a PaddleOCR model to test'
      });
      return;
    }

    if (inputType === 'upload' && uploadedImages.length === 0) {
      onNotification({
        type: 'warning',
        title: 'No Image Selected',
        message: 'Please upload an image to test'
      });
      return;
    }

    setLoading(true);
    try {
      let testData;
      
      if (inputType === 'stream') {
        const response = await testingAPI.testPaddleOCRModel(selectedModel, null, paddleOCRSettings);
        testData = response.data;
      } else if (inputType === 'upload' && uploadedImages.length > 0) {
        // Test with all uploaded images
        const allResults = [];
        for (let i = 0; i < uploadedImages.length; i++) {
          const image = uploadedImages[i];
          try {
            const response = await testingAPI.testPaddleOCRModelWithUpload(
              selectedModel, 
              image.file,
              paddleOCRSettings
            );
            allResults.push({
              ...response.data,
              image_name: image.name,
              image_index: i
            });
          } catch (error) {
            console.error(`Error testing image ${image.name}:`, error);
            allResults.push({
              error: `Failed to test ${image.name}: ${error.message}`,
              image_name: image.name,
              image_index: i
            });
          }
        }
        
        // Combine results
        if (allResults.length === 1) {
          testData = allResults[0];
        } else {
          testData = {
            model_name: selectedModel,
            results: allResults,
            total_images: uploadedImages.length,
            successful_tests: allResults.filter(r => !r.error).length
          };
        }
      } else if (inputType === 'video' && uploadedVideo) {
        const videoOptions = {
          language: paddleOCRSettings.language,
          task_type: paddleOCRSettings.taskType,
          confidence_threshold: paddleOCRSettings.confidenceThreshold,
          skipFrequency: videoAnalysisSettings.frameSkipFrequency,
          maxFrames: videoAnalysisSettings.maxFramesToAnalyze,
          generateVideo: videoAnalysisSettings.generateVideo
        };
        
        const response = await testingAPI.testPaddleOCRModelWithVideo(
          selectedModel,
          uploadedVideo.file,
          videoOptions
        );
        testData = response.data;
      }

      if (testData) {
        const fullResults = {
          ...testData,
          confidenceThreshold: paddleOCRSettings.confidenceThreshold
        };
        
        setTestResults(prev => [{
          id: Date.now(),
          modelName: selectedModel,
          modelType: 'paddleocr',
          inputType,
          timestamp: new Date(),
          results: fullResults,
          settings: paddleOCRSettings
        }, ...prev]);

        onNotification({
          type: 'success',
          title: 'PaddleOCR Test Completed',
          message: `Detected ${testData.detections?.length || 0} text regions`
        });
      }
    } catch (error) {
      console.error('PaddleOCR test failed:', error);
      onNotification({
        type: 'error',
        title: 'Test Failed',
        message: error.response?.data?.detail || 'Failed to run PaddleOCR test'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleAsyncVideoAnalysis = async (modelName, videoFile, options, modelType) => {
    try {
      // First, try async video analysis
      const asyncResponse = await testingAPI.testModelWithVideoAsync(modelName, videoFile, options);
      const analysisId = asyncResponse.data.analysis_id;
      
      onNotification({
        type: 'info',
        title: 'Video Analysis Started',
        message: 'Analysis is running in the background. Results will appear when ready.'
      });
      
      // Start polling for results
      const pollForResults = async () => {
        const maxPolls = 120; // Poll for up to 10 minutes (5-second intervals)
        let pollCount = 0;
        
        const poll = async () => {
          try {
            const statusResponse = await testingAPI.getVideoAnalysisStatus(analysisId);
            const status = statusResponse.data.status;
            
            if (status === 'completed') {
              // Get the results
              const resultsResponse = await testingAPI.getVideoAnalysisResults(analysisId);
              const testData = resultsResponse.data;
              
              // Process and display results
              const fullResults = {
                ...testData,
                analysis_id: analysisId,
                async_analysis: true
              };
              
              setTestResults(prev => [{
                id: Date.now(),
                modelName: modelName,
                modelType: modelType,
                inputType: 'video',
                timestamp: new Date(),
                results: fullResults,
                settings: modelType === 'object_detection' ? objectDetectionSettings : 
                         modelType === 'image_classification' ? classificationSettings :
                         modelType === 'vision_llm' ? visionLLMSettings : paddleOCRSettings
              }, ...prev]);
              
              onNotification({
                type: 'success',
                title: 'Video Analysis Completed',
                message: `Analysis completed! Processed ${testData.total_frames_analyzed} frames.`
              });
              
              setLoading(false);
              return true; // Stop polling
              
            } else if (status === 'failed') {
              throw new Error(statusResponse.data.error || 'Analysis failed');
              
            } else if (status === 'running') {
              const progress = statusResponse.data.progress || {};
              if (progress.percentage) {
                onNotification({
                  type: 'info',
                  title: 'Analysis Progress',
                  message: `Processing... ${Math.round(progress.percentage)}% complete`
                });
              }
              
              // Continue polling
              pollCount++;
              if (pollCount < maxPolls) {
                setTimeout(poll, 5000); // Poll every 5 seconds
              } else {
                throw new Error('Analysis timed out after 10 minutes');
              }
            } else {
              // Still pending, continue polling
              pollCount++;
              if (pollCount < maxPolls) {
                setTimeout(poll, 5000);
              } else {
                throw new Error('Analysis timed out waiting to start');
              }
            }
          } catch (error) {
            console.error('Polling error:', error);
            onNotification({
              type: 'error',
              title: 'Analysis Failed',
              message: error.message || 'Failed to get analysis status'
            });
            setLoading(false);
          }
        };
        
        // Start polling after a short delay
        setTimeout(poll, 2000);
      };
      
      pollForResults();
      
    } catch (error) {
      console.error('Async video analysis failed:', error);
      
      // Fallback to synchronous analysis
      onNotification({
        type: 'warning',
        title: 'Falling back to synchronous analysis',
        message: 'Async analysis failed, trying synchronous method...'
      });
      
      throw error; // Re-throw to trigger fallback in calling function
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
                sx={{ mb: 2 }}
              >
                Select Video
              </Button>
              
              {uploadedVideo && (
                <Card sx={{ mt: 2, p: 2, bgcolor: 'background.paper' }}>
                  <Typography variant="h6" gutterBottom>
                    Video Information
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <Typography variant="body2"><strong>File:</strong> {uploadedVideo.name}</Typography>
                      <Typography variant="body2"><strong>Size:</strong> {(uploadedVideo.size / (1024 * 1024)).toFixed(2)} MB</Typography>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      {uploadedVideo.duration && (
                        <Typography variant="body2"><strong>Duration:</strong> {uploadedVideo.duration.toFixed(1)}s</Typography>
                      )}
                      {uploadedVideo.estimatedFPS && (
                        <Typography variant="body2"><strong>Est. FPS:</strong> {uploadedVideo.estimatedFPS}</Typography>
                      )}
                      {uploadedVideo.width && uploadedVideo.height && (
                        <Typography variant="body2"><strong>Resolution:</strong> {uploadedVideo.width}x{uploadedVideo.height}</Typography>
                      )}
                    </Grid>
                  </Grid>
                </Card>
              )}

              {/* Video Analysis Settings */}
              {uploadedVideo && selectedModelType === 'object_detection' && (
                <Card sx={{ mt: 2, p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Video Analysis Settings
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Frame Skip Frequency"
                        type="number"
                        value={videoAnalysisSettings.frameSkipFrequency}
                        onChange={(e) => setVideoAnalysisSettings(prev => ({
                          ...prev,
                          frameSkipFrequency: parseInt(e.target.value) || 1
                        }))}
                        inputProps={{ min: 1, max: 120 }}
                        helperText="Analyze every Nth frame"
                        size="small"
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Max Frames to Analyze"
                        type="number"
                        value={videoAnalysisSettings.maxFramesToAnalyze}
                        onChange={(e) => setVideoAnalysisSettings(prev => ({
                          ...prev,
                          maxFramesToAnalyze: parseInt(e.target.value) || 1
                        }))}
                        inputProps={{ min: 1, max: 100 }}
                        helperText="Maximum frames to process"
                        size="small"
                      />
                    </Grid>

                    <Grid item xs={12}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Select Classes to Detect (Optional)</InputLabel>
                        <Select
                          multiple
                          value={videoAnalysisSettings.selectedClasses}
                          onChange={(e) => setVideoAnalysisSettings(prev => ({
                            ...prev,
                            selectedClasses: e.target.value
                          }))}
                          renderValue={(selected) => (
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                              {selected.map((value) => (
                                <Chip key={value} label={value} size="small" />
                              ))}
                            </Box>
                          )}
                        >
                          {availableClasses.map((cls) => (
                            <MenuItem key={cls.id} value={cls.name}>
                              {cls.name}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>

                    <Grid item xs={12}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={videoAnalysisSettings.generateAnnotatedVideo}
                            onChange={(e) => setVideoAnalysisSettings(prev => ({
                              ...prev,
                              generateAnnotatedVideo: e.target.checked
                            }))}
                          />
                        }
                        label={
                          <Box>
                            <Typography variant="body2">Generate Annotated Video</Typography>
                            <Typography variant="caption" color="text.secondary">
                              Create MP4 output with bounding box overlays (slower processing)
                            </Typography>
                          </Box>
                        }
                      />
                    </Grid>
                  </Grid>
                </Card>
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
            {selectedModelType === 'paddleocr' && <Tab label="PaddleOCR" />}
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
                    
                    <FormControlLabel
                      control={
                        <Switch
                          checked={objectDetectionSettings.colorByClass}
                          onChange={(e) => setObjectDetectionSettings(prev => ({
                            ...prev,
                            colorByClass: e.target.checked
                          }))}
                        />
                      }
                      label={
                        <Box>
                          <Typography variant="body2">Color by Class</Typography>
                          <Typography variant="caption" color="text.secondary">
                            Use consistent colors for each class instead of different colors for each detection
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
                      {loading 
                        ? (testingProgress.total > 1 
                            ? `Detecting... (${testingProgress.current}/${testingProgress.total})` 
                            : 'Detecting...'
                          ) 
                        : 'Run Object Detection'
                      }
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
                      {loading 
                        ? (testingProgress.total > 1 
                            ? `Classifying... (${testingProgress.current}/${testingProgress.total})` 
                            : 'Classifying...'
                          ) 
                        : 'Run Classification'
                      }
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
                      {loading 
                        ? (testingProgress.total > 1 
                            ? `Analyzing... (${testingProgress.current}/${testingProgress.total})` 
                            : 'Analyzing...'
                          ) 
                        : 'Analyze with Vision LLM'
                      }
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

          {/* PaddleOCR Testing */}
          {activeTab === 0 && selectedModelType === 'paddleocr' && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      PaddleOCR Settings
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <FormControl fullWidth>
                          <InputLabel>Language</InputLabel>
                          <Select
                            value={paddleOCRSettings.language}
                            onChange={(e) => setPaddleOCRSettings(prev => ({ ...prev, language: e.target.value }))}
                          >
                            <MenuItem value="en">English</MenuItem>
                            <MenuItem value="ch">Chinese</MenuItem>
                            <MenuItem value="ka">Korean</MenuItem>
                            <MenuItem value="japan">Japanese</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <FormControl fullWidth>
                          <InputLabel>Task Type</InputLabel>
                          <Select
                            value={paddleOCRSettings.taskType}
                            onChange={(e) => setPaddleOCRSettings(prev => ({ ...prev, taskType: e.target.value }))}
                          >
                            <MenuItem value="det">Text Detection Only</MenuItem>
                            <MenuItem value="rec">Text Recognition Only</MenuItem>
                            <MenuItem value="both">Detection + Recognition</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Typography gutterBottom>
                          Confidence Threshold: {paddleOCRSettings.confidenceThreshold}
                        </Typography>
                        <Slider
                          value={paddleOCRSettings.confidenceThreshold}
                          onChange={(e, value) => setPaddleOCRSettings(prev => ({ ...prev, confidenceThreshold: value }))}
                          min={0.1}
                          max={1.0}
                          step={0.05}
                          marks
                          valueLabelDisplay="auto"
                        />
                      </Grid>
                      
                      <Grid item xs={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={paddleOCRSettings.showBoundingBoxes}
                              onChange={(e) => setPaddleOCRSettings(prev => ({ ...prev, showBoundingBoxes: e.target.checked }))}
                            />
                          }
                          label="Show Bounding Boxes"
                        />
                      </Grid>
                      
                      <Grid item xs={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={paddleOCRSettings.showText}
                              onChange={(e) => setPaddleOCRSettings(prev => ({ ...prev, showText: e.target.checked }))}
                            />
                          }
                          label="Show Recognized Text"
                        />
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Test Controls
                    </Typography>
                    
                    <Button
                      variant="contained"
                      fullWidth
                      startIcon={loading ? <CircularProgress size={16} /> : <TestingIcon />}
                      onClick={runPaddleOCRTest}
                      disabled={loading || !selectedModel}
                      sx={{ mb: 2 }}
                    >
                      {loading 
                        ? (testingProgress.total > 1 
                            ? `Processing... (${testingProgress.current}/${testingProgress.total})` 
                            : 'Processing...'
                          ) 
                        : 'Run Text Analysis'
                      }
                    </Button>

                    <Button
                      variant="outlined"
                      fullWidth
                      startIcon={isLiveTesting ? <StopIcon /> : <PlayIcon />}
                      onClick={isLiveTesting ? stopLiveTesting : startLiveTesting}
                      disabled={loading || !selectedModel}
                    >
                      {isLiveTesting ? 'Stop Live Testing' : 'Start Live Testing'}
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {/* Results Tab */}
          {(activeTab === 1 || (activeTab === 0 && !['object_detection', 'image_classification', 'vision_llm', 'paddleocr'].includes(selectedModelType))) && (
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

                              {/* Video Analysis Results */}
                              {result.results.video_info && (
                                <Box>
                                  <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                                    Video Analysis Results
                                  </Typography>
                                  
                                  {/* Video Information Card */}
                                  <Card sx={{ mb: 2, p: 2, bgcolor: 'background.default' }}>
                                    <Typography variant="subtitle1" gutterBottom>Video Information</Typography>
                                    <Grid container spacing={2}>
                                      <Grid item xs={6} sm={3}>
                                        <Typography variant="body2"><strong>File:</strong> {result.results.video_info.filename}</Typography>
                                      </Grid>
                                      <Grid item xs={6} sm={3}>
                                        <Typography variant="body2"><strong>FPS:</strong> {result.results.video_info.fps?.toFixed(1) || 'N/A'}</Typography>
                                      </Grid>
                                      <Grid item xs={6} sm={3}>
                                        <Typography variant="body2"><strong>Duration:</strong> {result.results.video_info.duration_seconds?.toFixed(1)}s</Typography>
                                      </Grid>
                                      <Grid item xs={6} sm={3}>
                                        <Typography variant="body2"><strong>Resolution:</strong> {result.results.video_info.width}x{result.results.video_info.height}</Typography>
                                      </Grid>
                                    </Grid>
                                  </Card>

                                  {/* Analysis Summary */}
                                  {result.results.summary && (
                                    <Card sx={{ mb: 2, p: 2, bgcolor: 'background.default' }}>
                                      <Typography variant="subtitle1" gutterBottom>Analysis Summary</Typography>
                                      <Grid container spacing={2}>
                                        <Grid item xs={6} sm={3}>
                                          <Typography variant="body2"><strong>Frames Analyzed:</strong> {result.results.video_info.analyzed_frames}</Typography>
                                        </Grid>
                                        <Grid item xs={6} sm={3}>
                                          <Typography variant="body2"><strong>Total Detections:</strong> {result.results.summary.total_detections}</Typography>
                                        </Grid>
                                        <Grid item xs={6} sm={3}>
                                          <Typography variant="body2"><strong>Avg per Frame:</strong> {result.results.summary.avg_detections_per_frame?.toFixed(1)}</Typography>
                                        </Grid>
                                        <Grid item xs={6} sm={3}>
                                          <Typography variant="body2"><strong>Processing Time:</strong> {result.results.summary.processing_time_seconds?.toFixed(1)}s</Typography>
                                        </Grid>
                                      </Grid>
                                    </Card>
                                  )}

                                  {/* Download Annotated Video */}
                                  {result.results.video_info.generate_video && result.results.video_info.output_video_path && (
                                    <Box sx={{ mb: 2 }}>
                                      <Button
                                        variant="contained"
                                        startIcon={<DownloadIcon />}
                                        onClick={async () => {
                                          try {
                                            const response = await testingAPI.downloadAnnotatedVideo(result.results.test_id);
                                            const blob = new Blob([response.data], { type: 'video/mp4' });
                                            const url = window.URL.createObjectURL(blob);
                                            const link = document.createElement('a');
                                            link.href = url;
                                            link.download = `annotated_video_${result.results.test_id}.mp4`;
                                            document.body.appendChild(link);
                                            link.click();
                                            document.body.removeChild(link);
                                            window.URL.revokeObjectURL(url);
                                            onNotification({
                                              type: 'success',
                                              title: 'Download Started',
                                              message: 'Annotated video download has begun'
                                            });
                                          } catch (error) {
                                            console.error('Failed to download video:', error);
                                            onNotification({
                                              type: 'error',
                                              title: 'Download Failed',
                                              message: 'Failed to download annotated video'
                                            });
                                          }
                                        }}
                                      >
                                        Download Annotated Video
                                      </Button>
                                    </Box>
                                  )}

                                  {/* Frame Analysis Results */}
                                  {result.results.frame_results && result.results.frame_results.length > 0 && (
                                    <Accordion>
                                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                        <Typography variant="subtitle1">
                                          Frame-by-Frame Analysis ({result.results.frame_results.length} frames)
                                        </Typography>
                                      </AccordionSummary>
                                      <AccordionDetails>
                                        <Grid container spacing={2}>
                                          {result.results.frame_results.map((frame, frameIndex) => (
                                            <Grid item xs={12} sm={6} md={4} key={frameIndex}>
                                              <Card sx={{ p: 1 }}>
                                                <Typography variant="caption" color="text.secondary">
                                                  Frame {frame.frame_number} ({frame.timestamp?.toFixed(1)}s)
                                                </Typography>
                                                {frame.frame_url && (
                                                  <CardMedia
                                                    component="img"
                                                    sx={{ maxHeight: 120, objectFit: 'contain', mt: 1, mb: 1 }}
                                                    image={frame.frame_url}
                                                    alt={`Frame ${frame.frame_number}`}
                                                    onError={(e) => {
                                                      console.error('Frame image failed to load:', frame.frame_url);
                                                      e.target.style.display = 'none';
                                                    }}
                                                  />
                                                )}
                                                <Typography variant="caption">
                                                  Detections: {frame.detections?.length || 0}
                                                </Typography>
                                                {frame.error && (
                                                  <Typography variant="caption" color="error.main">
                                                    Error: {frame.error}
                                                  </Typography>
                                                )}
                                              </Card>
                                            </Grid>
                                          ))}
                                        </Grid>
                                      </AccordionDetails>
                                    </Accordion>
                                  )}
                                </Box>
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
                              
                              {/* Detection Summary Table for Object Detection */}
                              {result.modelType === 'object_detection' && result.results.detections && result.results.detections.length > 0 && (
                                <Box sx={{ mb: 2 }}>
                                  <Typography variant="subtitle2" color="primary" gutterBottom>
                                    Detection Summary
                                  </Typography>
                                  <Table size="small">
                                    <TableHead>
                                      <TableRow>
                                        <TableCell>Class</TableCell>
                                        <TableCell align="right">Count</TableCell>
                                        <TableCell align="right">Avg Confidence</TableCell>
                                      </TableRow>
                                    </TableHead>
                                    <TableBody>
                                      {(() => {
                                        const classStats = result.results.detections.reduce((acc, detection) => {
                                          const className = detection.class;
                                          if (!acc[className]) {
                                            acc[className] = { count: 0, totalConfidence: 0 };
                                          }
                                          acc[className].count += 1;
                                          acc[className].totalConfidence += detection.confidence;
                                          return acc;
                                        }, {});
                                        
                                        return Object.entries(classStats)
                                          .sort(([,a], [,b]) => b.count - a.count)
                                          .map(([className, stats]) => (
                                            <TableRow key={className}>
                                              <TableCell>{className}</TableCell>
                                              <TableCell align="right">{stats.count}</TableCell>
                                              <TableCell align="right">
                                                {(stats.totalConfidence / stats.count * 100).toFixed(1)}%
                                              </TableCell>
                                            </TableRow>
                                          ));
                                      })()}
                                    </TableBody>
                                  </Table>
                                </Box>
                              )}
                              
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
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Test History
                  </Typography>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={loadTestHistory}
                    disabled={loading}
                  >
                    Refresh
                  </Button>
                </Box>
                
                {testHistory.length === 0 ? (
                  <Alert severity="info">
                    No test history available.
                  </Alert>
                ) : (
                  <Box sx={{ maxHeight: 600, overflow: 'auto' }}>
                    {testHistory.slice(0, 20).map((test, index) => (
                      <Accordion key={test.test_id || index} sx={{ mb: 1 }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', pr: 2 }}>
                            <Box>
                              <Typography variant="subtitle1">
                                {test.test_type === 'comparison' ? 
                                  `Model Comparison` :
                                  test.test_type === 'benchmark' ? 
                                    `Benchmark Test` :
                                    `Model Test`
                                }
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {test.test_type === 'comparison' ? 
                                  `${test.models_tested?.join(', ')}` :
                                  `${test.model || test.model_name || 'Unknown Model'}`
                                } • {new Date(test.created_at || test.timestamp).toLocaleString()}
                              </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', gap: 1 }}>
                              <Chip 
                                label={test.test_type || 'test'} 
                                size="small" 
                                color={test.test_type === 'comparison' ? 'secondary' : 
                                       test.test_type === 'benchmark' ? 'primary' : 'default'} 
                              />
                              {test.input_type && (
                                <Chip 
                                  label={test.input_type} 
                                  size="small" 
                                  variant="outlined"
                                />
                              )}
                            </Box>
                          </Box>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Grid container spacing={2}>
                            {/* Test Details */}
                            <Grid item xs={12} md={6}>
                              <Typography variant="subtitle2" gutterBottom>
                                Test Information
                              </Typography>
                              <Table size="small">
                                <TableBody>
                                  <TableRow>
                                    <TableCell><strong>Model</strong></TableCell>
                                    <TableCell>{test.model || test.model_name || 'N/A'}</TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell><strong>Type</strong></TableCell>
                                    <TableCell>{test.model_type || test.test_type || 'N/A'}</TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell><strong>Input</strong></TableCell>
                                    <TableCell>{test.input_type || 'N/A'}</TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell><strong>Timestamp</strong></TableCell>
                                    <TableCell>{new Date(test.created_at || test.timestamp).toLocaleString()}</TableCell>
                                  </TableRow>
                                  {test.processing_time && (
                                    <TableRow>
                                      <TableCell><strong>Processing Time</strong></TableCell>
                                      <TableCell>{test.processing_time}s</TableCell>
                                    </TableRow>
                                  )}
                                </TableBody>
                              </Table>
                            </Grid>
                            
                            {/* Test Results */}
                            <Grid item xs={12} md={6}>
                              <Typography variant="subtitle2" gutterBottom>
                                Results Summary
                              </Typography>
                              {test.results ? (
                                <Box>
                                  {test.results.detections && (
                                    <Typography variant="body2" sx={{ mb: 1 }}>
                                      <strong>Detections:</strong> {test.results.detections.length} objects found
                                    </Typography>
                                  )}
                                  {test.results.total_frames_analyzed && (
                                    <Typography variant="body2" sx={{ mb: 1 }}>
                                      <strong>Frames Analyzed:</strong> {test.results.total_frames_analyzed}
                                    </Typography>
                                  )}
                                  {test.results.video_duration && (
                                    <Typography variant="body2" sx={{ mb: 1 }}>
                                      <strong>Video Duration:</strong> {test.results.video_duration.toFixed(1)}s
                                    </Typography>
                                  )}
                                  {test.results.analysis && (
                                    <Typography variant="body2" sx={{ mb: 1 }}>
                                      <strong>Analysis:</strong> {test.results.analysis.substring(0, 100)}
                                      {test.results.analysis.length > 100 ? '...' : ''}
                                    </Typography>
                                  )}
                                </Box>
                              ) : (
                                <Typography variant="body2" color="text.secondary">
                                  No detailed results available
                                </Typography>
                              )}
                            </Grid>
                            
                            {/* Settings Used */}
                            {test.settings && (
                              <Grid item xs={12}>
                                <Typography variant="subtitle2" gutterBottom>
                                  Test Settings
                                </Typography>
                                <Box sx={{ 
                                  backgroundColor: 'grey.50', 
                                  p: 1, 
                                  borderRadius: 1,
                                  maxHeight: 200,
                                  overflow: 'auto'
                                }}>
                                  <pre style={{ fontSize: '0.75rem', margin: 0 }}>
                                    {JSON.stringify(test.settings, null, 2)}
                                  </pre>
                                </Box>
                              </Grid>
                            )}
                            
                            {/* Test Image/Results */}
                            {test.results?.image_url && (
                              <Grid item xs={12}>
                                <Typography variant="subtitle2" gutterBottom>
                                  Result Image
                                </Typography>
                                <Box sx={{ textAlign: 'center' }}>
                                  <img 
                                    src={test.results.image_url} 
                                    alt="Test Result"
                                    style={{ 
                                      maxWidth: '100%', 
                                      maxHeight: '300px',
                                      border: '1px solid #ddd',
                                      borderRadius: '4px'
                                    }}
                                    onError={(e) => {
                                      e.target.style.display = 'none';
                                    }}
                                  />
                                </Box>
                              </Grid>
                            )}
                          </Grid>
                        </AccordionDetails>
                      </Accordion>
                    ))}
                  </Box>
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