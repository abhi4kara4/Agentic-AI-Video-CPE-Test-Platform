import React, { useState, useEffect } from 'react';
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
} from '@mui/material';
import {
  Science as TestingIcon,
  Speed as BenchmarkIcon,
  Compare as CompareIcon,
} from '@mui/icons-material';

const ModelTesting = ({ onNotification }) => {
  const [selectedModel, setSelectedModel] = useState('');
  const [testPrompt, setTestPrompt] = useState('');
  const [availableModels] = useState([
    'llava:7b',
    'llava:7b-q4',
    'moondream:latest',
    'tv-optimized-model'
  ]);

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Model Testing
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Test and compare vision models with live video streams
        </Typography>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Model Testing interface is coming soon! This will allow you to:
        <ul>
          <li>Test models with live video streams</li>
          <li>Compare multiple models side-by-side</li>
          <li>Benchmark model performance and accuracy</li>
          <li>Export test results and metrics</li>
        </ul>
      </Alert>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Single Model Testing
              </Typography>
              
              <FormControl fullWidth margin="normal">
                <InputLabel>Select Model</InputLabel>
                <Select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {availableModels.map((model) => (
                    <MenuItem key={model} value={model}>
                      {model}
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
                rows={3}
                placeholder="Describe what you see on this TV screen..."
              />

              <Button
                fullWidth
                variant="contained"
                startIcon={<TestingIcon />}
                sx={{ mt: 2 }}
                disabled
              >
                Test Current Frame
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Comparison
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<CompareIcon />}
                    disabled
                  >
                    Compare Models
                  </Button>
                </Grid>
                <Grid item xs={12}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<BenchmarkIcon />}
                    disabled
                  >
                    Benchmark Performance
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ModelTesting;