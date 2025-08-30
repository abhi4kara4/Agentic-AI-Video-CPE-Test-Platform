import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  IconButton,
  Paper,
  Grid,
  Tooltip
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Undo as UndoIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as ResetIcon,
  PanTool as PanIcon,
  Edit as LabelIcon,
  ContentCopy as CopyIcon,
  ContentPaste as PasteIcon
} from '@mui/icons-material';

import { OBJECT_DETECTION_CLASSES } from '../constants/datasetTypes.js';

const ObjectDetectionLabeler = ({ 
  image, 
  labels, 
  onLabelsChange, 
  copiedAnnotations,
  onCopyAnnotations,
  showCopyPaste = false,
  imageName = null
}) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null); // Cache the loaded image
  const zoomTimeoutRef = useRef(null); // Debounce zoom operations
  const [isDrawing, setIsDrawing] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState(null);
  const [startPos, setStartPos] = useState(null);
  const [currentBox, setCurrentBox] = useState(null);
  const [selectedClass, setSelectedClass] = useState('button');
  const [boundingBoxes, setBoundingBoxes] = useState(labels?.boundingBoxes || []);
  
  // Debug logging for received labels and update boundingBoxes when labels change
  useEffect(() => {
    console.log('ObjectDetectionLabeler received labels:', labels);
    if (labels?.boundingBoxes) {
      console.log('Bounding boxes:', labels.boundingBoxes);
      setBoundingBoxes(labels.boundingBoxes); // Update state when labels change
    } else {
      setBoundingBoxes([]); // Clear if no bounding boxes
    }
  }, [labels]);
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 }); // Track pan offset
  const [interactionMode, setInteractionMode] = useState('label'); // 'label' or 'pan'
  const [canvasSize, setCanvasSize] = useState({ width: 600, height: 400 });
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageParams, setImageParams] = useState({ offsetX: 0, offsetY: 0, drawWidth: 0, drawHeight: 0, baseOffsetX: 0, baseOffsetY: 0 });

  useEffect(() => {
    if (imageLoaded) {
      drawCanvas();
    }
  }, [boundingBoxes, zoom, imageLoaded, panOffset]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (event.target.tagName !== 'INPUT' && event.target.tagName !== 'TEXTAREA') {
        switch (event.key.toLowerCase()) {
          case 'p':
            toggleInteractionMode();
            event.preventDefault();
            break;
          case '=':
          case '+':
            handleZoomIn();
            event.preventDefault();
            break;
          case '-':
            handleZoomOut();
            event.preventDefault();
            break;
          case '0':
            handleZoomReset();
            event.preventDefault();
            break;
          case 'escape':
            if (isDrawing) {
              setIsDrawing(false);
              setCurrentBox(null);
              setStartPos(null);
            }
            if (isPanning) {
              setIsPanning(false);
              setPanStart(null);
            }
            event.preventDefault();
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isDrawing, isPanning]);

  useEffect(() => {
    onLabelsChange({
      ...labels,
      boundingBoxes: boundingBoxes
    });
  }, [boundingBoxes]);

  // Reset pan offset when zoom changes significantly or image changes
  useEffect(() => {
    if (zoom === 1) {
      setPanOffset({ x: 0, y: 0 });
    }
  }, [zoom]);

  useEffect(() => {
    if (image) {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        // Cache the loaded image
        imageRef.current = img;

        // Calculate aspect ratio and fit image to canvas with proper centering
        const aspectRatio = img.width / img.height;
        const canvasAspectRatio = canvas.width / canvas.height;
        
        let baseDrawWidth, baseDrawHeight;
        
        // Fit image to canvas maintaining aspect ratio
        if (aspectRatio > canvasAspectRatio) {
          // Image is wider - fit to width
          baseDrawWidth = canvas.width * 0.9; // Leave 10% margin
          baseDrawHeight = baseDrawWidth / aspectRatio;
        } else {
          // Image is taller - fit to height  
          baseDrawHeight = canvas.height * 0.9; // Leave 10% margin
          baseDrawWidth = baseDrawHeight * aspectRatio;
        }

        // Apply zoom
        const drawWidth = baseDrawWidth * zoom;
        const drawHeight = baseDrawHeight * zoom;

        // Center the image properly
        const baseOffsetX = (canvas.width - baseDrawWidth) / 2;
        const baseOffsetY = (canvas.height - baseDrawHeight) / 2;
        
        // Apply pan offset and zoom-adjusted centering
        const offsetX = baseOffsetX - (drawWidth - baseDrawWidth) / 2 + panOffset.x;
        const offsetY = baseOffsetY - (drawHeight - baseDrawHeight) / 2 + panOffset.y;

        setImageParams({ 
          offsetX, 
          offsetY, 
          drawWidth, 
          drawHeight, 
          baseOffsetX,
          baseOffsetY
        });
        setImageLoaded(true);
      };
      img.src = image;
    }
  }, [image, zoom, panOffset]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !imageLoaded || !imageRef.current) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw cached image to prevent flickering
    const { offsetX, offsetY, drawWidth, drawHeight } = imageParams;
    
    ctx.drawImage(imageRef.current, offsetX, offsetY, drawWidth, drawHeight);

    // Draw existing bounding boxes
    boundingBoxes.forEach((box, index) => {
      const cls = OBJECT_DETECTION_CLASSES[box.class];
      ctx.strokeStyle = cls?.color || '#FF0000';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        offsetX + (box.x * drawWidth),
        offsetY + (box.y * drawHeight),
        box.width * drawWidth,
        box.height * drawHeight
      );

      // Draw label background
      const labelText = cls?.name || box.class;
      ctx.font = '12px Arial';
      const textMetrics = ctx.measureText(labelText);
      const labelX = offsetX + (box.x * drawWidth);
      const labelY = offsetY + (box.y * drawHeight) - 5;
      
      ctx.fillStyle = cls?.color || '#FF0000';
      ctx.fillRect(labelX, labelY - 12, textMetrics.width + 4, 14);
      
      // Draw label text
      ctx.fillStyle = 'white';
      ctx.fillText(labelText, labelX + 2, labelY);
    });

    // Draw current drawing box
    if (currentBox) {
      const cls = OBJECT_DETECTION_CLASSES[selectedClass];
      ctx.strokeStyle = cls?.color || '#FF0000';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(
        currentBox.x,
        currentBox.y,
        currentBox.width,
        currentBox.height
      );
      ctx.setLineDash([]);
    }
  };

  const getCanvasCoordinates = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  };

  const handleMouseDown = (event) => {
    const pos = getCanvasCoordinates(event);
    
    if (interactionMode === 'pan') {
      setIsPanning(true);
      setPanStart(pos);
    } else {
      setIsDrawing(true);
      setStartPos(pos);
    }
  };

  const handleMouseMove = (event) => {
    const pos = getCanvasCoordinates(event);
    
    if (isPanning && panStart) {
      // Handle panning
      const deltaX = pos.x - panStart.x;
      const deltaY = pos.y - panStart.y;
      
      setPanOffset(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }));
      
      setPanStart(pos);
      return;
    }
    
    if (!isDrawing || !startPos) return;

    // Handle bounding box drawing
    const width = pos.x - startPos.x;
    const height = pos.y - startPos.y;

    const newBox = {
      x: width > 0 ? startPos.x : pos.x,
      y: height > 0 ? startPos.y : pos.y,
      width: Math.abs(width),
      height: Math.abs(height)
    };

    setCurrentBox(newBox);
    
    // Trigger immediate redraw for smooth drawing
    requestAnimationFrame(() => {
      drawCanvas();
    });
  };

  const handleMouseUp = (event) => {
    if (isPanning) {
      setIsPanning(false);
      setPanStart(null);
      return;
    }
    
    if (!isDrawing || !currentBox) return;

    // Convert canvas coordinates to normalized coordinates relative to the actual image
    const { offsetX, offsetY, drawWidth, drawHeight } = imageParams;
    
    // Calculate the bounding box relative to the image (not the canvas)
    const imageRelativeBox = {
      x: Math.max(0, (currentBox.x - offsetX) / drawWidth),
      y: Math.max(0, (currentBox.y - offsetY) / drawHeight),
      width: Math.min(1, currentBox.width / drawWidth),
      height: Math.min(1, currentBox.height / drawHeight)
    };

    // Ensure the box is within image bounds and has reasonable size
    if (imageRelativeBox.width > 0.02 && imageRelativeBox.height > 0.02 &&
        imageRelativeBox.x >= 0 && imageRelativeBox.y >= 0 &&
        imageRelativeBox.x + imageRelativeBox.width <= 1 &&
        imageRelativeBox.y + imageRelativeBox.height <= 1) {
      
      const normalizedBox = {
        id: Date.now(),
        class: selectedClass,
        ...imageRelativeBox
      };
      
      setBoundingBoxes(prev => [...prev, normalizedBox]);
    }

    setIsDrawing(false);
    setStartPos(null);
    setCurrentBox(null);
  };

  const handleDeleteBox = (boxId) => {
    setBoundingBoxes(prev => prev.filter(box => box.id !== boxId));
  };

  const handleClearAll = () => {
    setBoundingBoxes([]);
  };

  const handleZoomIn = () => {
    // Clear any pending zoom operations for immediate response
    if (zoomTimeoutRef.current) {
      clearTimeout(zoomTimeoutRef.current);
    }
    
    setZoom(prev => {
      const newZoom = Math.min(prev + 0.3, 4); // Larger steps, higher max zoom
      return newZoom;
    });
  };

  const handleZoomOut = () => {
    // Clear any pending zoom operations for immediate response
    if (zoomTimeoutRef.current) {
      clearTimeout(zoomTimeoutRef.current);
    }
    
    setZoom(prev => {
      const newZoom = Math.max(prev - 0.3, 0.5); // Larger steps for faster response
      return newZoom;
    });
  };

  const handleZoomReset = () => {
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const toggleInteractionMode = () => {
    setInteractionMode(prev => prev === 'label' ? 'pan' : 'label');
  };

  const handleCopyAnnotations = () => {
    if (boundingBoxes.length === 0) return;
    
    console.log('Copy - imageName prop:', imageName);
    console.log('Copy - image prop:', image);
    
    const finalImageName = imageName || image?.name || image?.filename || 'unknown';
    console.log('Copy - final imageName used:', finalImageName);
    
    const annotationsToCopy = {
      boundingBoxes: boundingBoxes.map(box => ({ ...box })), // Deep copy
      imageInfo: {
        width: canvasSize.width,
        height: canvasSize.height,
        imageName: finalImageName
      }
    };
    
    onCopyAnnotations(annotationsToCopy);
  };

  const handlePasteAnnotations = () => {
    if (!copiedAnnotations || !copiedAnnotations.boundingBoxes) return;
    
    // Generate new IDs for pasted boxes to avoid conflicts
    const pastedBoxes = copiedAnnotations.boundingBoxes.map(box => ({
      ...box,
      id: Date.now() + Math.random() // New unique ID
    }));
    
    setBoundingBoxes(pastedBoxes);
    
    // Update labels immediately
    const updatedLabels = {
      ...labels,
      boundingBoxes: pastedBoxes
    };
    onLabelsChange(updatedLabels);
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Object Detection Labeling
      </Typography>

      <Grid container spacing={2}>
        {/* Canvas */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            {/* Mode and Zoom Controls */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip 
                  label={interactionMode === 'label' ? 'Label Mode' : 'Pan Mode'} 
                  color={interactionMode === 'label' ? 'primary' : 'secondary'}
                  size="small"
                />
                {interactionMode === 'label' ? 'Draw bounding boxes' : 'Pan around image'}
              </Typography>
              <Box sx={{ display: 'flex', gap: 0.5 }}>
                <Tooltip title={`Switch to ${interactionMode === 'label' ? 'Pan' : 'Label'} Mode`}>
                  <IconButton 
                    size="small" 
                    onClick={toggleInteractionMode}
                    color={interactionMode === 'pan' ? 'primary' : 'default'}
                  >
                    {interactionMode === 'label' ? <PanIcon /> : <LabelIcon />}
                  </IconButton>
                </Tooltip>
                <Tooltip title="Zoom In">
                  <IconButton size="small" onClick={handleZoomIn}>
                    <ZoomInIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Zoom Out">
                  <IconButton size="small" onClick={handleZoomOut}>
                    <ZoomOutIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Reset Zoom & Pan">
                  <IconButton size="small" onClick={handleZoomReset}>
                    <ResetIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Clear All Boxes">
                  <IconButton size="small" color="error" onClick={handleClearAll}>
                    <UndoIcon />
                  </IconButton>
                </Tooltip>
                {showCopyPaste && (
                  <>
                    <Tooltip title={boundingBoxes.length === 0 ? "No annotations to copy" : "Copy Annotations"}>
                      <span>
                        <IconButton 
                          size="small" 
                          onClick={handleCopyAnnotations}
                          disabled={boundingBoxes.length === 0}
                          color="primary"
                        >
                          <CopyIcon />
                        </IconButton>
                      </span>
                    </Tooltip>
                    <Tooltip title={(!copiedAnnotations || !copiedAnnotations.boundingBoxes) ? "No annotations copied" : "Paste Annotations"}>
                      <span>
                        <IconButton 
                          size="small" 
                          onClick={handlePasteAnnotations}
                          disabled={!copiedAnnotations || !copiedAnnotations.boundingBoxes}
                          color="secondary"
                        >
                          <PasteIcon />
                        </IconButton>
                      </span>
                    </Tooltip>
                  </>
                )}
              </Box>
            </Box>
            
            {/* Zoom Level Indicator */}
            <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography variant="caption" color="text.secondary">
                Zoom: {(zoom * 100).toFixed(0)}%
              </Typography>
              {zoom > 1 && (
                <Typography variant="caption" color="text.secondary">
                  Pan: ({panOffset.x.toFixed(0)}, {panOffset.y.toFixed(0)})
                </Typography>
              )}
            </Box>
            
            <Box sx={{ 
              border: '1px solid', 
              borderColor: 'divider', 
              borderRadius: 1,
              overflow: 'hidden',
              position: 'relative'
            }}>
              <canvas
                ref={canvasRef}
                width={canvasSize.width}
                height={canvasSize.height}
                style={{ 
                  display: 'block', 
                  cursor: interactionMode === 'label' ? 'crosshair' : (isPanning ? 'grabbing' : 'grab'),
                  maxWidth: '100%',
                  userSelect: 'none'
                }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={() => {
                  if (isPanning) {
                    setIsPanning(false);
                    setPanStart(null);
                  }
                }}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Controls */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Class Selection
            </Typography>
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Object Class</InputLabel>
              <Select
                value={selectedClass}
                onChange={(e) => setSelectedClass(e.target.value)}
                size="small"
              >
                {Object.entries(OBJECT_DETECTION_CLASSES).map(([key, cls]) => (
                  <MenuItem key={key} value={key}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box
                        sx={{
                          width: 12,
                          height: 12,
                          backgroundColor: cls.color,
                          borderRadius: 1
                        }}
                      />
                      {cls.name}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Typography variant="subtitle2" gutterBottom>
              Current Annotations ({boundingBoxes.length})
            </Typography>
            
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              {boundingBoxes.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                  No annotations yet
                </Typography>
              ) : (
                boundingBoxes.map((box, index) => {
                  const cls = OBJECT_DETECTION_CLASSES[box.class];
                  return (
                    <Box
                      key={box.id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        p: 1,
                        mb: 1,
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        fontSize: '0.8rem'
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box
                          sx={{
                            width: 8,
                            height: 8,
                            backgroundColor: cls?.color,
                            borderRadius: 1
                          }}
                        />
                        <Typography variant="caption">
                          {cls?.name}
                        </Typography>
                      </Box>
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleteBox(box.id)}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  );
                })
              )}
            </Box>

            {/* Class Legend */}
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Available Classes:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                {Object.entries(OBJECT_DETECTION_CLASSES).slice(0, 8).map(([key, cls]) => (
                  <Chip
                    key={key}
                    label={cls.name}
                    size="small"
                    sx={{ 
                      backgroundColor: cls.color, 
                      color: 'white',
                      fontSize: '0.65rem',
                      height: '20px'
                    }}
                  />
                ))}
              </Box>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ObjectDetectionLabeler;