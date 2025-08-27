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
  ZoomOut as ZoomOutIcon
} from '@mui/icons-material';

import { OBJECT_DETECTION_CLASSES } from '../constants/datasetTypes.js';

const ObjectDetectionLabeler = ({ image, labels, onLabelsChange }) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null); // Cache the loaded image
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState(null);
  const [currentBox, setCurrentBox] = useState(null);
  const [selectedClass, setSelectedClass] = useState('button');
  const [boundingBoxes, setBoundingBoxes] = useState(labels?.boundingBoxes || []);
  const [zoom, setZoom] = useState(1);
  const [canvasSize, setCanvasSize] = useState({ width: 600, height: 400 });
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageParams, setImageParams] = useState({ offsetX: 0, offsetY: 0, drawWidth: 0, drawHeight: 0 });

  useEffect(() => {
    if (imageLoaded) {
      drawCanvas();
    }
  }, [boundingBoxes, zoom, imageLoaded]);

  useEffect(() => {
    onLabelsChange({
      ...labels,
      boundingBoxes: boundingBoxes
    });
  }, [boundingBoxes]);

  useEffect(() => {
    if (image) {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        // Cache the loaded image
        imageRef.current = img;

        // Calculate aspect ratio and fit image to canvas
        const aspectRatio = img.width / img.height;
        let drawWidth, drawHeight;
        
        if (aspectRatio > canvas.width / canvas.height) {
          drawWidth = canvas.width * zoom;
          drawHeight = drawWidth / aspectRatio;
        } else {
          drawHeight = canvas.height * zoom;
          drawWidth = drawHeight * aspectRatio;
        }

        const offsetX = (canvas.width - drawWidth) / 2;
        const offsetY = (canvas.height - drawHeight) / 2;

        setImageParams({ offsetX, offsetY, drawWidth, drawHeight });
        setImageLoaded(true);
      };
      img.src = image;
    }
  }, [image, zoom]);

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
    setIsDrawing(true);
    setStartPos(pos);
  };

  const handleMouseMove = (event) => {
    if (!isDrawing || !startPos) return;

    const pos = getCanvasCoordinates(event);
    const width = pos.x - startPos.x;
    const height = pos.y - startPos.y;

    const newBox = {
      x: width > 0 ? startPos.x : pos.x,
      y: height > 0 ? startPos.y : pos.y,
      width: Math.abs(width),
      height: Math.abs(height)
    };

    setCurrentBox(newBox);
    
    // Use requestAnimationFrame to prevent flickering
    requestAnimationFrame(() => {
      drawCanvas();
    });
  };

  const handleMouseUp = (event) => {
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
    setZoom(prev => Math.min(prev + 0.2, 3));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.2, 0.5));
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
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="subtitle2">
                Draw bounding boxes by clicking and dragging
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
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
                <Tooltip title="Clear All">
                  <IconButton size="small" color="error" onClick={handleClearAll}>
                    <UndoIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
            
            <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              <canvas
                ref={canvasRef}
                width={canvasSize.width}
                height={canvasSize.height}
                style={{ 
                  display: 'block', 
                  cursor: 'crosshair',
                  maxWidth: '100%'
                }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
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