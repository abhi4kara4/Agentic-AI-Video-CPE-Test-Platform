import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Paper,
  Grid,
  Card,
  CardContent,
  Chip,
  TextField,
  Alert,
  Divider
} from '@mui/material';

import { CLASSIFICATION_CLASSES } from '../constants/datasetTypes.js';

const ImageClassificationLabeler = ({ image, labels, onLabelsChange }) => {
  const [selectedClass, setSelectedClass] = useState(labels?.className || '');
  const [confidence, setConfidence] = useState(labels?.confidence || 100);
  const [notes, setNotes] = useState(labels?.notes || '');

  useEffect(() => {
    onLabelsChange({
      ...labels,
      className: selectedClass,
      confidence: confidence,
      notes: notes,
      timestamp: new Date().toISOString()
    });
  }, [selectedClass, confidence, notes]);

  const handleClassChange = (event) => {
    setSelectedClass(event.target.value);
  };

  const getClassColor = (className) => {
    if (className.includes('error') || className.includes('black') || className.includes('blue')) {
      return 'error';
    }
    if (className.includes('loading') || className.includes('buffer')) {
      return 'warning';
    }
    if (className.includes('normal') || className.includes('home')) {
      return 'success';
    }
    return 'primary';
  };

  const getClassCategory = (className) => {
    if (['normal_playback', 'home_screen', 'app_screen'].includes(className)) {
      return 'Normal States';
    }
    if (['black_screen', 'blue_screen', 'error_screen'].includes(className)) {
      return 'Error States';
    }
    if (['pixelation', 'blur_screen', 'artifacts'].includes(className)) {
      return 'Quality Issues';
    }
    if (['buffering', 'loading_screen'].includes(className)) {
      return 'Loading States';
    }
    if (['screensaver', 'standby', 'no_signal'].includes(className)) {
      return 'Special States';
    }
    return 'Other';
  };

  // Group classes by category
  const groupedClasses = Object.entries(CLASSIFICATION_CLASSES).reduce((acc, [key, cls]) => {
    const category = getClassCategory(key);
    if (!acc[category]) acc[category] = [];
    acc[category].push({ key, ...cls });
    return acc;
  }, {});

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Image Classification Labeling
      </Typography>

      <Grid container spacing={3}>
        {/* Image Preview */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Image Preview
            </Typography>
            {image && (
              <Box
                sx={{
                  width: '100%',
                  maxHeight: 400,
                  overflow: 'hidden',
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'divider',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center'
                }}
              >
                <img
                  src={image}
                  alt="Classification target"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain'
                  }}
                />
              </Box>
            )}
            
            {selectedClass && (
              <Box sx={{ mt: 2 }}>
                <Alert severity={getClassColor(selectedClass)} variant="outlined">
                  <Typography variant="subtitle2">
                    Selected: {CLASSIFICATION_CLASSES[selectedClass]?.name}
                  </Typography>
                  <Typography variant="body2">
                    {CLASSIFICATION_CLASSES[selectedClass]?.description}
                  </Typography>
                </Alert>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Classification Controls */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Select Classification
            </Typography>

            <FormControl component="fieldset" fullWidth>
              <RadioGroup
                value={selectedClass}
                onChange={handleClassChange}
              >
                {Object.entries(groupedClasses).map(([category, classes]) => (
                  <Box key={category} sx={{ mb: 2 }}>
                    <Typography 
                      variant="subtitle2" 
                      color="primary" 
                      sx={{ fontWeight: 'bold', mb: 1 }}
                    >
                      {category}
                    </Typography>
                    {classes.map((cls) => (
                      <Card
                        key={cls.key}
                        variant="outlined"
                        sx={{
                          mb: 1,
                          border: selectedClass === cls.key ? '2px solid' : '1px solid',
                          borderColor: selectedClass === cls.key ? 'primary.main' : 'divider',
                          cursor: 'pointer',
                          '&:hover': {
                            backgroundColor: 'action.hover'
                          }
                        }}
                        onClick={() => setSelectedClass(cls.key)}
                      >
                        <CardContent sx={{ py: 1, px: 2, '&:last-child': { pb: 1 } }}>
                          <FormControlLabel
                            value={cls.key}
                            control={<Radio size="small" />}
                            label={
                              <Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  <Typography variant="body2" fontWeight="medium">
                                    {cls.name}
                                  </Typography>
                                  <Chip
                                    label={category.split(' ')[0]}
                                    size="small"
                                    color={getClassColor(cls.key)}
                                    variant="outlined"
                                    sx={{ fontSize: '0.65rem' }}
                                  />
                                </Box>
                                <Typography variant="caption" color="text.secondary">
                                  {cls.description}
                                </Typography>
                              </Box>
                            }
                            sx={{ margin: 0, width: '100%' }}
                          />
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                ))}
              </RadioGroup>
            </FormControl>

            <Divider sx={{ my: 2 }} />

            {/* Additional Options */}
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Additional Information
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Notes (Optional)"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add any additional observations about this image..."
                variant="outlined"
                size="small"
                sx={{ mt: 1 }}
              />
            </Box>

            {/* Statistics */}
            {selectedClass && (
              <Box sx={{ mt: 2, p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  <strong>Category:</strong> {getClassCategory(selectedClass)}
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ImageClassificationLabeler;