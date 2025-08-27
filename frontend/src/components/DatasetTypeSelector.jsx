import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Card,
  CardContent,
  Typography,
  Box,
  RadioGroup,
  FormControlLabel,
  Radio,
  Chip,
  Grid,
  Collapse,
  FormControl,
  FormLabel,
  FormGroup,
  Checkbox,
  Slider,
  Divider,
  Alert
} from '@mui/material';
import {
  SmartToy as VisionIcon,
  CropFree as DetectionIcon,
  Label as ClassificationIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  Tune as AugmentationIcon
} from '@mui/icons-material';

import { 
  DATASET_TYPES, 
  DATASET_TYPE_INFO, 
  OBJECT_DETECTION_CLASSES,
  CLASSIFICATION_CLASSES,
  AUGMENTATION_OPTIONS 
} from '../constants/datasetTypes.js';

const DatasetTypeSelector = ({ open, onClose, onSelect, currentConfig }) => {
  const [selectedType, setSelectedType] = useState(currentConfig?.datasetType || '');
  const [showAugmentations, setShowAugmentations] = useState(false);
  const [augmentationOptions, setAugmentationOptions] = useState(
    currentConfig?.augmentationOptions || {}
  );

  const handleTypeChange = (event) => {
    const type = event.target.value;
    setSelectedType(type);
    
    // Set default augmentation options for this type
    if (AUGMENTATION_OPTIONS[type]) {
      setAugmentationOptions(AUGMENTATION_OPTIONS[type]);
    }
  };

  const handleAugmentationChange = (option, value) => {
    setAugmentationOptions(prev => ({
      ...prev,
      [option]: { ...prev[option], ...value }
    }));
  };

  const handleConfirm = () => {
    if (!selectedType) return;
    
    onSelect({
      datasetType: selectedType,
      augmentationOptions: augmentationOptions
    });
    onClose();
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case DATASET_TYPES.VISION_LLM:
        return <VisionIcon sx={{ fontSize: 40, color: 'primary.main' }} />;
      case DATASET_TYPES.OBJECT_DETECTION:
        return <DetectionIcon sx={{ fontSize: 40, color: 'secondary.main' }} />;
      case DATASET_TYPES.IMAGE_CLASSIFICATION:
        return <ClassificationIcon sx={{ fontSize: 40, color: 'success.main' }} />;
      default:
        return null;
    }
  };

  const renderClassList = (type) => {
    if (type === DATASET_TYPES.OBJECT_DETECTION) {
      return (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Object Classes:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            {Object.entries(OBJECT_DETECTION_CLASSES).map(([key, cls]) => (
              <Chip
                key={key}
                label={cls.name}
                size="small"
                sx={{ 
                  backgroundColor: cls.color, 
                  color: 'white',
                  fontSize: '0.7rem'
                }}
              />
            ))}
          </Box>
        </Box>
      );
    }
    
    if (type === DATASET_TYPES.IMAGE_CLASSIFICATION) {
      return (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Classification Classes:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            {Object.entries(CLASSIFICATION_CLASSES).map(([key, cls]) => (
              <Chip
                key={key}
                label={cls.name}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem' }}
              />
            ))}
          </Box>
        </Box>
      );
    }
    
    return null;
  };

  const renderAugmentationOptions = () => {
    if (!selectedType || !augmentationOptions) return null;

    const options = AUGMENTATION_OPTIONS[selectedType];
    if (!options) return null;

    return (
      <Box sx={{ mt: 2 }}>
        <Button
          startIcon={showAugmentations ? <CollapseIcon /> : <ExpandIcon />}
          onClick={() => setShowAugmentations(!showAugmentations)}
          size="small"
          sx={{ mb: 1 }}
        >
          <AugmentationIcon sx={{ mr: 1 }} />
          Data Augmentation Options
        </Button>
        
        <Collapse in={showAugmentations}>
          <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
            <Grid container spacing={3}>
              {Object.entries(options).map(([option, config]) => (
                <Grid item xs={12} sm={6} key={option}>
                  <FormControl component="fieldset">
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={augmentationOptions[option]?.enabled ?? config.enabled}
                          onChange={(e) => handleAugmentationChange(option, { enabled: e.target.checked })}
                        />
                      }
                      label={option.charAt(0).toUpperCase() + option.slice(1)}
                    />
                    
                    {augmentationOptions[option]?.enabled && config.range && (
                      <Box sx={{ mt: 1, px: 2 }}>
                        <Typography variant="caption">Range:</Typography>
                        <Slider
                          size="small"
                          value={augmentationOptions[option]?.range || config.range}
                          min={Math.min(...config.range)}
                          max={Math.max(...config.range)}
                          valueLabelDisplay="auto"
                          onChange={(_, value) => handleAugmentationChange(option, { range: value })}
                        />
                      </Box>
                    )}
                  </FormControl>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Collapse>
      </Box>
    );
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          🎯 Select Dataset Type
        </Box>
      </DialogTitle>
      
      <DialogContent>
        <Alert severity="info" sx={{ mb: 3 }}>
          Choose the type of model you want to train. This will determine the labeling interface and export format.
        </Alert>

        <RadioGroup value={selectedType} onChange={handleTypeChange}>
          <Grid container spacing={2}>
            {Object.entries(DATASET_TYPE_INFO).map(([type, info]) => (
              <Grid item xs={12} key={type}>
                <Card
                  sx={{
                    cursor: 'pointer',
                    border: selectedType === type ? '2px solid' : '1px solid',
                    borderColor: selectedType === type ? 'primary.main' : 'divider',
                    '&:hover': {
                      borderColor: 'primary.light',
                      boxShadow: 2
                    }
                  }}
                  onClick={() => setSelectedType(type)}
                >
                  <CardContent>
                    <FormControlLabel
                      value={type}
                      control={<Radio />}
                      sx={{ mb: 1 }}
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, ml: 1 }}>
                          {getTypeIcon(type)}
                          <Box>
                            <Typography variant="h6" fontWeight="bold">
                              {info.icon} {info.name}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {info.description}
                            </Typography>
                          </Box>
                        </Box>
                      }
                    />
                    
                    <Box sx={{ ml: 4, mt: 1 }}>
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        {info.requiresBoundingBoxes && (
                          <Chip label="Bounding Boxes" size="small" color="primary" />
                        )}
                        {info.requiresTextLabels && (
                          <Chip label="Text Descriptions" size="small" color="secondary" />
                        )}
                        <Chip 
                          label={info.multiClass ? 'Multi-class' : 'Single-class'} 
                          size="small" 
                          variant="outlined" 
                        />
                      </Box>
                      
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                        Formats: {info.supportedFormats.join(', ')}
                      </Typography>
                      
                      {renderClassList(type)}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </RadioGroup>

        {selectedType && (
          <>
            <Divider sx={{ my: 3 }} />
            {renderAugmentationOptions()}
          </>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button 
          variant="contained" 
          onClick={handleConfirm}
          disabled={!selectedType}
        >
          Continue with {selectedType ? DATASET_TYPE_INFO[selectedType].name : 'Selection'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default DatasetTypeSelector;