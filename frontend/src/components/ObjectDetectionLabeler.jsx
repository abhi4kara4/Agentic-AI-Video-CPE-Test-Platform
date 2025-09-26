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
  Tooltip,
  Checkbox,
  Slider,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails
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
  ContentPaste as PasteIcon,
  SelectAll as SelectAllIcon,
  Deselect as DeselectIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';

import { OBJECT_DETECTION_CLASSES, AUGMENTATION_OPTIONS, DATASET_TYPES } from '../constants/datasetTypes.js';

const ObjectDetectionLabeler = ({ 
  image, 
  labels, 
  onLabelsChange, 
  copiedAnnotations,
  onCopyAnnotations,
  showCopyPaste = false,
  imageName = null,
  customClasses = null,
  copiedAugmentationOptions = null,
  onCopyAugmentationOptions = null
}) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null); // Cache the loaded image
  const zoomTimeoutRef = useRef(null); // Debounce zoom operations
  const [isDrawing, setIsDrawing] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState(null);
  const [startPos, setStartPos] = useState(null);
  const [currentBox, setCurrentBox] = useState(null);
  // Use custom classes if provided, otherwise fall back to defaults
  // Custom classes should contain ALL relevant classes for the dataset
  const availableClasses = customClasses || OBJECT_DETECTION_CLASSES;
  
  // Log when custom classes are provided for debugging
  if (customClasses) {
    console.log('ObjectDetectionLabeler - Using custom classes:', Object.keys(customClasses));
  }
  const [selectedClass, setSelectedClass] = useState(Object.keys(availableClasses)[0] || 'button');
  
  // Update selected class when available classes change
  useEffect(() => {
    const currentKeys = Object.keys(availableClasses);
    if (currentKeys.length > 0 && !currentKeys.includes(selectedClass)) {
      setSelectedClass(currentKeys[0]);
    }
  }, [availableClasses, selectedClass]);
  const [boundingBoxes, setBoundingBoxes] = useState(labels?.boundingBoxes || []);
  const [selectedAnnotations, setSelectedAnnotations] = useState(new Set());
  const [augmentationOptions, setAugmentationOptions] = useState(
    labels?.augmentationOptions || AUGMENTATION_OPTIONS[DATASET_TYPES.OBJECT_DETECTION] || {}
  );
  
  // New states for hover highlighting and dragging
  const [hoveredBoxId, setHoveredBoxId] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragBoxId, setDragBoxId] = useState(null);
  const [dragStart, setDragStart] = useState(null);
  const [dragBoxOriginal, setDragBoxOriginal] = useState(null);
  
  // Debug logging for received labels and update boundingBoxes when labels change
  useEffect(() => {
    console.log('ObjectDetectionLabeler received labels:', labels);
    if (labels?.boundingBoxes) {
      console.log('Bounding boxes:', labels.boundingBoxes);
      setBoundingBoxes(labels.boundingBoxes); // Update state when labels change
    } else {
      setBoundingBoxes([]); // Clear if no bounding boxes
    }
    // Update augmentation options from labels
    if (labels?.augmentationOptions) {
      setAugmentationOptions(labels.augmentationOptions);
    }
    // Clear selected annotations when labels change (e.g., switching images)
    setSelectedAnnotations(new Set());
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
  }, [boundingBoxes, zoom, imageLoaded, panOffset, hoveredBoxId, dragBoxId, dragStart, dragBoxOriginal]);

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
            if (isDragging) {
              setIsDragging(false);
              setDragBoxId(null);
              setDragStart(null);
              setDragBoxOriginal(null);
            }
            event.preventDefault();
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isDrawing, isPanning, isDragging]);

  useEffect(() => {
    onLabelsChange({
      ...labels,
      boundingBoxes: boundingBoxes,
      augmentationOptions: augmentationOptions
    });
  }, [boundingBoxes, augmentationOptions]);

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
      const cls = availableClasses[box.class];
      const isHovered = hoveredBoxId === box.id;
      const isDraggedBox = dragBoxId === box.id;
      
      // Enhanced styling for hovered/dragged boxes
      if (isHovered || isDraggedBox) {
        // Draw glow effect
        ctx.shadowColor = cls?.color || '#FF0000';
        ctx.shadowBlur = 15;
        ctx.lineWidth = 4;
      } else {
        ctx.shadowBlur = 0;
        ctx.lineWidth = 2;
      }
      
      ctx.strokeStyle = cls?.color || '#FF0000';
      
      // Calculate box position (use dragged position if being dragged)
      let boxX = box.x;
      let boxY = box.y;
      
      if (isDraggedBox && dragStart && dragBoxOriginal) {
        // Apply drag offset
        const dragOffsetX = (dragStart.currentX - dragStart.startX) / drawWidth;
        const dragOffsetY = (dragStart.currentY - dragStart.startY) / drawHeight;
        boxX = dragBoxOriginal.x + dragOffsetX;
        boxY = dragBoxOriginal.y + dragOffsetY;
      }
      
      ctx.strokeRect(
        offsetX + (boxX * drawWidth),
        offsetY + (boxY * drawHeight),
        box.width * drawWidth,
        box.height * drawHeight
      );

      // Draw label background
      const labelText = cls?.name || box.class;
      ctx.font = isHovered || isDraggedBox ? '14px Arial' : '12px Arial';
      const textMetrics = ctx.measureText(labelText);
      const labelX = offsetX + (boxX * drawWidth);
      const labelY = offsetY + (boxY * drawHeight) - 5;
      
      ctx.fillStyle = cls?.color || '#FF0000';
      ctx.fillRect(labelX, labelY - (isHovered || isDraggedBox ? 14 : 12), textMetrics.width + 4, isHovered || isDraggedBox ? 16 : 14);
      
      // Draw label text
      ctx.fillStyle = 'white';
      ctx.fillText(labelText, labelX + 2, labelY);
      
      // Reset shadow for next box
      ctx.shadowBlur = 0;
    });

    // Draw current drawing box
    if (currentBox) {
      const cls = availableClasses[selectedClass];
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

  const findBoxAtPosition = (canvasPos) => {
    const { offsetX, offsetY, drawWidth, drawHeight } = imageParams;
    
    // Convert canvas position to image coordinates
    const imageX = (canvasPos.x - offsetX) / drawWidth;
    const imageY = (canvasPos.y - offsetY) / drawHeight;
    
    // Check each bounding box (in reverse order to prioritize top boxes)
    for (let i = boundingBoxes.length - 1; i >= 0; i--) {
      const box = boundingBoxes[i];
      if (imageX >= box.x && imageX <= box.x + box.width &&
          imageY >= box.y && imageY <= box.y + box.height) {
        return box;
      }
    }
    return null;
  };

  const getCursorStyle = () => {
    if (interactionMode === 'pan') {
      return isPanning ? 'grabbing' : 'grab';
    }
    if (isDragging) {
      return 'grabbing';
    }
    // For label mode, we'll dynamically check on mouse move
    return 'crosshair';
  };

  const handleMouseDown = (event) => {
    const pos = getCanvasCoordinates(event);
    
    if (interactionMode === 'pan') {
      setIsPanning(true);
      setPanStart(pos);
    } else {
      // Check if clicking on an existing box for dragging
      const clickedBox = findBoxAtPosition(pos);
      
      if (clickedBox) {
        // Start dragging existing box
        setIsDragging(true);
        setDragBoxId(clickedBox.id);
        setDragBoxOriginal({ ...clickedBox });
        setDragStart({
          startX: pos.x,
          startY: pos.y,
          currentX: pos.x,
          currentY: pos.y
        });
      } else {
        // Start drawing new box
        setIsDrawing(true);
        setStartPos(pos);
      }
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
    
    if (isDragging && dragStart) {
      // Handle box dragging
      setDragStart(prev => ({
        ...prev,
        currentX: pos.x,
        currentY: pos.y
      }));
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
    
    if (isDragging) {
      // Complete drag operation
      if (dragBoxId && dragStart && dragBoxOriginal) {
        const { offsetX, offsetY, drawWidth, drawHeight } = imageParams;
        
        // Calculate new position
        const dragOffsetX = (dragStart.currentX - dragStart.startX) / drawWidth;
        const dragOffsetY = (dragStart.currentY - dragStart.startY) / drawHeight;
        const newX = Math.max(0, Math.min(1 - dragBoxOriginal.width, dragBoxOriginal.x + dragOffsetX));
        const newY = Math.max(0, Math.min(1 - dragBoxOriginal.height, dragBoxOriginal.y + dragOffsetY));
        
        // Update the box position
        setBoundingBoxes(prev => prev.map(box => 
          box.id === dragBoxId ? { ...box, x: newX, y: newY } : box
        ));
      }
      
      setIsDragging(false);
      setDragBoxId(null);
      setDragStart(null);
      setDragBoxOriginal(null);
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
    
    console.log('ObjectDetectionLabeler - Copy - imageName prop:', imageName);
    console.log('ObjectDetectionLabeler - Copy - image prop (url):', typeof image === 'string' ? image.substring(0, 100) + '...' : image);
    
    // Try to extract filename from URL if imageName is not provided
    let extractedName = imageName;
    if (!extractedName && typeof image === 'string') {
      // Try to extract filename from URL path
      const urlMatch = image.match(/\/([^\/]+\.jpg)$/);
      if (urlMatch) {
        extractedName = urlMatch[1];
      }
    }
    
    const finalImageName = extractedName || 'unknown';
    console.log('ObjectDetectionLabeler - Copy - final imageName used:', finalImageName);
    
    const annotationsToCopy = {
      boundingBoxes: boundingBoxes.map(box => ({ ...box })), // Deep copy
      imageInfo: {
        width: canvasSize.width,
        height: canvasSize.height,
        imageName: finalImageName
      }
    };
    
    console.log('ObjectDetectionLabeler - Copy - annotations to copy:', annotationsToCopy);
    onCopyAnnotations(annotationsToCopy);
  };

  const handlePasteAnnotations = () => {
    if (!copiedAnnotations || !copiedAnnotations.boundingBoxes) return;
    
    // Generate new IDs for pasted boxes to avoid conflicts
    const pastedBoxes = copiedAnnotations.boundingBoxes.map(box => ({
      ...box,
      id: Date.now() + Math.random() // New unique ID
    }));
    
    // Add to existing annotations instead of replacing them
    const updatedBoundingBoxes = [...boundingBoxes, ...pastedBoxes];
    setBoundingBoxes(updatedBoundingBoxes);
    
    // Update labels immediately
    const updatedLabels = {
      ...labels,
      boundingBoxes: updatedBoundingBoxes
    };
    onLabelsChange(updatedLabels);
  };

  const handleAnnotationSelect = (boxId, isSelected) => {
    setSelectedAnnotations(prev => {
      const newSelected = new Set(prev);
      if (isSelected) {
        newSelected.add(boxId);
      } else {
        newSelected.delete(boxId);
      }
      return newSelected;
    });
  };

  const handleSelectAll = () => {
    if (selectedAnnotations.size === boundingBoxes.length) {
      // If all are selected, deselect all
      setSelectedAnnotations(new Set());
    } else {
      // Select all
      setSelectedAnnotations(new Set(boundingBoxes.map(box => box.id)));
    }
  };

  const handleCopySelected = () => {
    if (selectedAnnotations.size === 0) return;
    
    // Filter to only selected annotations
    const selectedBoxes = boundingBoxes.filter(box => selectedAnnotations.has(box.id));
    
    console.log('ObjectDetectionLabeler - Copy Selected - imageName prop:', imageName);
    console.log('ObjectDetectionLabeler - Copy Selected - image prop (url):', typeof image === 'string' ? image.substring(0, 100) + '...' : image);
    
    // Try to extract filename from URL if imageName is not provided
    let extractedName = imageName;
    if (!extractedName && typeof image === 'string') {
      const urlMatch = image.match(/\/([^\/]+\.jpg)$/);
      if (urlMatch) {
        extractedName = urlMatch[1];
      }
    }
    
    const finalImageName = extractedName || 'unknown';
    console.log('ObjectDetectionLabeler - Copy Selected - final imageName used:', finalImageName);
    
    const annotationsToCopy = {
      boundingBoxes: selectedBoxes.map(box => ({ ...box })), // Deep copy
      imageInfo: {
        width: canvasSize.width,
        height: canvasSize.height,
        imageName: finalImageName
      }
    };
    
    console.log('ObjectDetectionLabeler - Copy Selected - annotations to copy:', annotationsToCopy);
    onCopyAnnotations(annotationsToCopy);
    
    // Clear selection after copying
    setSelectedAnnotations(new Set());
  };

  const handleAugmentationChange = (option, value) => {
    setAugmentationOptions(prev => ({
      ...prev,
      [option]: { ...prev[option], ...value }
    }));
  };

  const handleCopyAugmentationOptions = () => {
    if (onCopyAugmentationOptions) {
      onCopyAugmentationOptions({ ...augmentationOptions });
    }
  };

  const handlePasteAugmentationOptions = () => {
    if (copiedAugmentationOptions) {
      setAugmentationOptions({ ...copiedAugmentationOptions });
    }
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
                {interactionMode === 'label' ? 'Draw new boxes or drag existing ones' : 'Pan around image'}
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
                  cursor: getCursorStyle(),
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
                {Object.entries(availableClasses).map(([key, cls]) => (
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

            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle2">
                Current Annotations ({boundingBoxes.length})
              </Typography>
              {boundingBoxes.length > 0 && (
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                  <Tooltip title={selectedAnnotations.size === boundingBoxes.length ? "Deselect All" : "Select All"}>
                    <IconButton size="small" onClick={handleSelectAll}>
                      {selectedAnnotations.size === boundingBoxes.length ? <DeselectIcon fontSize="small" /> : <SelectAllIcon fontSize="small" />}
                    </IconButton>
                  </Tooltip>
                  <Tooltip title={selectedAnnotations.size === 0 ? "No annotations selected" : `Copy ${selectedAnnotations.size} selected annotation(s)`}>
                    <span>
                      <IconButton 
                        size="small" 
                        onClick={handleCopySelected}
                        disabled={selectedAnnotations.size === 0}
                        color="primary"
                      >
                        <CopyIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Box>
              )}
            </Box>
            
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              {boundingBoxes.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                  No annotations yet
                </Typography>
              ) : (
                boundingBoxes.map((box, index) => {
                  const cls = availableClasses[box.class];
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
                        borderColor: hoveredBoxId === box.id ? cls?.color : 'divider',
                        borderRadius: 1,
                        fontSize: '0.8rem',
                        backgroundColor: hoveredBoxId === box.id ? `${cls?.color}10` : 'transparent',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease-in-out'
                      }}
                      onMouseEnter={() => setHoveredBoxId(box.id)}
                      onMouseLeave={() => setHoveredBoxId(null)}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Checkbox
                          size="small"
                          checked={selectedAnnotations.has(box.id)}
                          onChange={(e) => handleAnnotationSelect(box.id, e.target.checked)}
                          sx={{ p: 0.5 }}
                        />
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
                {Object.entries(availableClasses).slice(0, 8).map(([key, cls]) => (
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

            {/* Augmentation Options */}
            <Accordion sx={{ mt: 2 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <SettingsIcon fontSize="small" />
                  <Typography variant="subtitle2">
                    Augmentation Options
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                {Object.entries(AUGMENTATION_OPTIONS[DATASET_TYPES.OBJECT_DETECTION] || {}).map(([option, defaultConfig]) => (
                  <Box key={option} sx={{ mb: 2 }}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={augmentationOptions[option]?.enabled ?? defaultConfig.enabled}
                          onChange={(e) => handleAugmentationChange(option, { enabled: e.target.checked })}
                          size="small"
                        />
                      }
                      label={option.charAt(0).toUpperCase() + option.slice(1)}
                    />
                    
                    {augmentationOptions[option]?.enabled && defaultConfig.range && (
                      <Box sx={{ mt: 1, px: 2 }}>
                        <Typography variant="caption" color="text.secondary">
                          Range: {defaultConfig.range[0]} to {defaultConfig.range[1]}
                        </Typography>
                        <Slider
                          size="small"
                          value={augmentationOptions[option]?.range || defaultConfig.range}
                          min={Math.min(...defaultConfig.range)}
                          max={Math.max(...defaultConfig.range)}
                          step={(Math.max(...defaultConfig.range) - Math.min(...defaultConfig.range)) / 20}
                          valueLabelDisplay="auto"
                          onChange={(_, value) => handleAugmentationChange(option, { range: value })}
                          sx={{ mt: 1 }}
                        />
                      </Box>
                    )}
                  </Box>
                ))}
                
                {/* Copy/Paste Augmentation Options */}
                <Box sx={{ display: 'flex', gap: 1, mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                  <Tooltip title="Copy Augmentation Options">
                    <IconButton 
                      size="small" 
                      onClick={handleCopyAugmentationOptions}
                      color="primary"
                    >
                      <CopyIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title={!copiedAugmentationOptions ? "No augmentation options copied" : "Paste Augmentation Options"}>
                    <span>
                      <IconButton 
                        size="small" 
                        onClick={handlePasteAugmentationOptions}
                        disabled={!copiedAugmentationOptions}
                        color="secondary"
                      >
                        <PasteIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Box>
              </AccordionDetails>
            </Accordion>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ObjectDetectionLabeler;