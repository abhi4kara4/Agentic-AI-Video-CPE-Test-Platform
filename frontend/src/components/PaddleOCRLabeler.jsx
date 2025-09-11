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
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
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
  TextFields as TextIcon
} from '@mui/icons-material';

import { PADDLEOCR_TEXT_TYPES } from '../constants/datasetTypes.js';

const PaddleOCRLabeler = ({ 
  image, 
  labels, 
  onLabelsChange, 
  copiedAnnotations,
  onCopyAnnotations,
  showCopyPaste = false,
  imageName = null,
  customTextTypes = null
}) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const zoomTimeoutRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState(null);
  const [startPos, setStartPos] = useState(null);
  const [currentBox, setCurrentBox] = useState(null);
  
  // Use custom text types if provided, otherwise fall back to defaults
  const availableTextTypes = customTextTypes || PADDLEOCR_TEXT_TYPES;
  const [selectedTextType, setSelectedTextType] = useState(Object.keys(availableTextTypes)[0] || 'button_text');
  const [textBoxes, setTextBoxes] = useState(labels?.textBoxes || []);
  
  // Text input dialog state
  const [textInputDialog, setTextInputDialog] = useState(null);
  const [currentText, setCurrentText] = useState('');
  
  useEffect(() => {
    console.log('PaddleOCRLabeler received labels:', labels);
    if (labels?.textBoxes) {
      console.log('Text boxes:', labels.textBoxes);
      setTextBoxes(labels.textBoxes);
      
      // Force redraw after setting text boxes
      setTimeout(() => {
        if (imageLoaded) {
          drawCanvas();
        }
      }, 100);
    } else {
      setTextBoxes([]);
    }
  }, [labels]);
  
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [interactionMode, setInteractionMode] = useState('label'); // 'label' or 'pan'
  const [canvasSize, setCanvasSize] = useState({ width: 600, height: 400 });
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageParams, setImageParams] = useState({ offsetX: 0, offsetY: 0, drawWidth: 0, drawHeight: 0, baseOffsetX: 0, baseOffsetY: 0 });

  useEffect(() => {
    if (imageLoaded) {
      drawCanvas();
    }
  }, [textBoxes, zoom, imageLoaded, panOffset, imageParams]);

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
            cancelDrawing();
            event.preventDefault();
            break;
          default:
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, []);

  const loadImage = () => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;

    const img = new Image();
    imageRef.current = img;
    
    img.onload = () => {
      const ctx = canvas.getContext('2d');
      
      // Calculate canvas size to maintain aspect ratio
      const maxWidth = Math.min(800, window.innerWidth * 0.6);
      const maxHeight = Math.min(600, window.innerHeight * 0.6);
      const aspectRatio = img.width / img.height;
      
      let canvasWidth, canvasHeight;
      if (aspectRatio > maxWidth / maxHeight) {
        canvasWidth = maxWidth;
        canvasHeight = maxWidth / aspectRatio;
      } else {
        canvasHeight = maxHeight;
        canvasWidth = maxHeight * aspectRatio;
      }
      
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
      setCanvasSize({ width: canvasWidth, height: canvasHeight });
      
      setImageLoaded(true);
      drawCanvas();
    };
    
    img.src = image;
  };

  useEffect(() => {
    loadImage();
  }, [image]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !imageLoaded) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate image position and size with zoom and pan
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    const imgAspectRatio = img.width / img.height;
    const canvasAspectRatio = canvasWidth / canvasHeight;
    
    let drawWidth, drawHeight, baseOffsetX, baseOffsetY;
    
    if (imgAspectRatio > canvasAspectRatio) {
      drawWidth = canvasWidth * zoom;
      drawHeight = (canvasWidth / imgAspectRatio) * zoom;
      baseOffsetX = 0;
      baseOffsetY = (canvasHeight - drawHeight / zoom) / 2;
    } else {
      drawHeight = canvasHeight * zoom;
      drawWidth = (canvasHeight * imgAspectRatio) * zoom;
      baseOffsetX = (canvasWidth - drawWidth / zoom) / 2;
      baseOffsetY = 0;
    }
    
    const offsetX = baseOffsetX + panOffset.x;
    const offsetY = baseOffsetY + panOffset.y;
    
    setImageParams({ offsetX, offsetY, drawWidth, drawHeight, baseOffsetX, baseOffsetY });
    
    // Draw image
    ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
    
    // Draw existing text boxes
    textBoxes.forEach((box, index) => {
      drawTextBox(ctx, box, index);
    });
    
    // Draw current box being drawn
    if (currentBox) {
      drawTextBox(ctx, { ...currentBox, type: selectedTextType }, -1, true);
    }
  };

  const drawTextBox = (ctx, box, index, isTemporary = false) => {
    const { x, y, width, height, text, type } = box;
    const textType = availableTextTypes[type] || availableTextTypes[Object.keys(availableTextTypes)[0]];
    const color = textType.color;
    
    // Check if imageParams is valid
    if (!imageParams.drawWidth || !imageParams.drawHeight) {
      return; // Skip drawing if image params not ready
    }
    
    // Convert normalized coordinates to canvas coordinates
    const canvasX = imageParams.offsetX + (x / 100) * imageParams.drawWidth;
    const canvasY = imageParams.offsetY + (y / 100) * imageParams.drawHeight;
    const canvasWidth = (width / 100) * imageParams.drawWidth;
    const canvasHeight = (height / 100) * imageParams.drawHeight;
    
    ctx.strokeStyle = color;
    ctx.fillStyle = isTemporary ? color + '20' : color + '10';
    ctx.lineWidth = 2;
    
    // Draw bounding box
    ctx.fillRect(canvasX, canvasY, canvasWidth, canvasHeight);
    ctx.strokeRect(canvasX, canvasY, canvasWidth, canvasHeight);
    
    // Draw text type label
    ctx.fillStyle = color;
    ctx.font = '12px Arial';
    const labelText = `${textType.name}${text ? `: ${text}` : ''}`;
    const textMetrics = ctx.measureText(labelText);
    const labelPadding = 4;
    
    // Background for label
    ctx.fillRect(
      canvasX - 1, 
      canvasY - 16,
      textMetrics.width + labelPadding * 2,
      16
    );
    
    // Label text
    ctx.fillStyle = 'white';
    ctx.fillText(labelText, canvasX + labelPadding, canvasY - 4);
    
    // Draw index number for non-temporary boxes
    if (!isTemporary) {
      ctx.fillStyle = color;
      ctx.font = 'bold 14px Arial';
      ctx.fillText((index + 1).toString(), canvasX + 2, canvasY + 16);
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

  const canvasToImageCoordinates = (canvasX, canvasY) => {
    // Convert canvas coordinates to normalized image coordinates (0-100)
    const relativeX = (canvasX - imageParams.offsetX) / imageParams.drawWidth;
    const relativeY = (canvasY - imageParams.offsetY) / imageParams.drawHeight;
    
    return {
      x: Math.max(0, Math.min(100, relativeX * 100)),
      y: Math.max(0, Math.min(100, relativeY * 100))
    };
  };

  const handleMouseDown = (event) => {
    if (interactionMode === 'pan') {
      setIsPanning(true);
      setPanStart(getCanvasCoordinates(event));
      return;
    }

    const coords = getCanvasCoordinates(event);
    const imageCoords = canvasToImageCoordinates(coords.x, coords.y);
    
    setIsDrawing(true);
    setStartPos(imageCoords);
    setCurrentBox({
      x: imageCoords.x,
      y: imageCoords.y,
      width: 0,
      height: 0,
      text: '',
      type: selectedTextType
    });
  };

  const handleMouseMove = (event) => {
    if (isPanning && panStart) {
      const coords = getCanvasCoordinates(event);
      const deltaX = coords.x - panStart.x;
      const deltaY = coords.y - panStart.y;
      
      setPanOffset(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }));
      setPanStart(coords);
      return;
    }

    if (isDrawing && startPos) {
      const coords = getCanvasCoordinates(event);
      const imageCoords = canvasToImageCoordinates(coords.x, coords.y);
      
      const width = Math.abs(imageCoords.x - startPos.x);
      const height = Math.abs(imageCoords.y - startPos.y);
      const x = Math.min(startPos.x, imageCoords.x);
      const y = Math.min(startPos.y, imageCoords.y);
      
      setCurrentBox({
        x,
        y,
        width,
        height,
        text: '',
        type: selectedTextType
      });
      
      drawCanvas();
    }
  };

  const handleMouseUp = () => {
    if (isPanning) {
      setIsPanning(false);
      setPanStart(null);
      return;
    }

    if (isDrawing && currentBox && currentBox.width > 1 && currentBox.height > 1) {
      // Open text input dialog
      setTextInputDialog(currentBox);
      setCurrentText('');
    }
    
    setIsDrawing(false);
    setStartPos(null);
    setCurrentBox(null);
  };

  const handleTextSubmit = () => {
    if (textInputDialog) {
      const newTextBox = {
        ...textInputDialog,
        text: currentText,
        id: Date.now()
      };
      
      const updatedTextBoxes = [...textBoxes, newTextBox];
      setTextBoxes(updatedTextBoxes);
      
      // Call parent callback
      onLabelsChange({
        textBoxes: updatedTextBoxes
      });
    }
    
    setTextInputDialog(null);
    setCurrentText('');
  };

  const handleTextCancel = () => {
    setTextInputDialog(null);
    setCurrentText('');
  };

  const removeTextBox = (index) => {
    const updatedTextBoxes = textBoxes.filter((_, i) => i !== index);
    setTextBoxes(updatedTextBoxes);
    
    onLabelsChange({
      textBoxes: updatedTextBoxes
    });
  };

  const clearAllTextBoxes = () => {
    setTextBoxes([]);
    onLabelsChange({
      textBoxes: []
    });
  };

  const toggleInteractionMode = () => {
    setInteractionMode(prev => prev === 'label' ? 'pan' : 'label');
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.2, 5));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev / 1.2, 0.1));
  };

  const handleZoomReset = () => {
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const cancelDrawing = () => {
    setIsDrawing(false);
    setCurrentBox(null);
    setStartPos(null);
    drawCanvas();
  };

  const handleCopyAnnotations = () => {
    if (onCopyAnnotations && textBoxes.length > 0) {
      onCopyAnnotations(textBoxes, imageName);
    }
  };

  const handlePasteAnnotations = () => {
    if (copiedAnnotations && copiedAnnotations.length > 0) {
      const updatedTextBoxes = [...textBoxes, ...copiedAnnotations.map(box => ({
        ...box,
        id: Date.now() + Math.random()
      }))];
      setTextBoxes(updatedTextBoxes);
      
      onLabelsChange({
        textBoxes: updatedTextBoxes
      });
    }
  };

  return (
    <Box>
      {/* Controls */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Text Type</InputLabel>
              <Select
                value={selectedTextType}
                onChange={(e) => setSelectedTextType(e.target.value)}
                label="Text Type"
              >
                {Object.entries(availableTextTypes).map(([key, textType]) => (
                  <MenuItem key={key} value={key}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box
                        sx={{
                          width: 16,
                          height: 16,
                          backgroundColor: textType.color,
                          borderRadius: '50%'
                        }}
                      />
                      {textType.name}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title={`Zoom: ${Math.round(zoom * 100)}%`}>
                <IconButton size="small" onClick={handleZoomIn}>
                  <ZoomInIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Zoom Out">
                <IconButton size="small" onClick={handleZoomOut}>
                  <ZoomOutIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Reset Zoom">
                <IconButton size="small" onClick={handleZoomReset}>
                  <ResetIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title={`Mode: ${interactionMode === 'label' ? 'Labeling' : 'Panning'} (P)`}>
                <IconButton 
                  size="small" 
                  onClick={toggleInteractionMode}
                  color={interactionMode === 'pan' ? 'primary' : 'default'}
                >
                  <PanIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Clear All">
                <IconButton size="small" onClick={clearAllTextBoxes} color="error">
                  <DeleteIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>
          
          {showCopyPaste && (
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Tooltip title="Copy Annotations">
                  <span>
                    <IconButton 
                      size="small" 
                      onClick={handleCopyAnnotations}
                      disabled={textBoxes.length === 0}
                    >
                      <CopyIcon />
                    </IconButton>
                  </span>
                </Tooltip>
                <Tooltip title="Paste Annotations">
                  <span>
                    <IconButton 
                      size="small" 
                      onClick={handlePasteAnnotations}
                      disabled={!copiedAnnotations || copiedAnnotations.length === 0}
                    >
                      <PasteIcon />
                    </IconButton>
                  </span>
                </Tooltip>
              </Box>
            </Grid>
          )}
        </Grid>
        
        {/* Text boxes summary */}
        {textBoxes.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Text Annotations ({textBoxes.length}):
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {textBoxes.map((box, index) => {
                const textType = availableTextTypes[box.type] || availableTextTypes[Object.keys(availableTextTypes)[0]];
                return (
                  <Chip
                    key={index}
                    label={`${index + 1}. ${textType.name}${box.text ? `: ${box.text.substring(0, 20)}${box.text.length > 20 ? '...' : ''}` : ''}`}
                    size="small"
                    sx={{ 
                      backgroundColor: textType.color + '20',
                      color: textType.color,
                      border: `1px solid ${textType.color}40`
                    }}
                    onDelete={() => removeTextBox(index)}
                    deleteIcon={<DeleteIcon />}
                  />
                );
              })}
            </Box>
          </Box>
        )}
      </Paper>

      {/* Canvas */}
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          style={{
            border: '2px solid #ccc',
            cursor: interactionMode === 'pan' ? 'grab' : 'crosshair',
            maxWidth: '100%',
            height: 'auto'
          }}
        />
      </Box>

      {/* Instructions */}
      <Paper sx={{ p: 2, backgroundColor: 'grey.50' }}>
        <Typography variant="subtitle2" gutterBottom>
          Instructions:
        </Typography>
        <Typography variant="body2" component="div">
          • Select a text type from the dropdown above
          <br />
          • Click and drag on the image to create bounding boxes around text
          <br />
          • Enter the actual text content when prompted
          <br />
          • Use P key to toggle between labeling and panning modes
          <br />
          • Use +/- keys to zoom in/out, 0 to reset zoom
          <br />
          • Press Escape to cancel current drawing
          {showCopyPaste && (
            <>
              <br />
              • Use copy/paste buttons to reuse annotations across images
            </>
          )}
        </Typography>
      </Paper>

      {/* Text Input Dialog */}
      <Dialog open={!!textInputDialog} onClose={handleTextCancel} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TextIcon />
            Enter Text Content
          </Box>
        </DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            fullWidth
            multiline
            rows={3}
            label="Text Content"
            value={currentText}
            onChange={(e) => setCurrentText(e.target.value)}
            placeholder="Enter the text that appears in this bounding box..."
            sx={{ mt: 1 }}
          />
          {textInputDialog && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Text Type: {availableTextTypes[textInputDialog.type]?.name || 'Unknown'}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleTextCancel}>Cancel</Button>
          <Button onClick={handleTextSubmit} variant="contained">
            Add Text Box
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PaddleOCRLabeler;