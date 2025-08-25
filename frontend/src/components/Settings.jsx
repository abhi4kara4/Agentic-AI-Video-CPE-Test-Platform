import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Save as SaveIcon,
} from '@mui/icons-material';

const Settings = ({ onNotification }) => {
  const [settings, setSettings] = useState({
    apiUrl: 'http://localhost:8000',
    videoDevice: '',
    macAddress: '',
    autoCapture: false,
    modelTimeout: 60,
  });

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Settings
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Configure platform settings and preferences
        </Typography>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Settings interface is coming soon! This will allow you to configure:
        <ul>
          <li>API endpoints and connection settings</li>
          <li>Default device configurations</li>
          <li>Model inference parameters</li>
          <li>Auto-capture and labeling preferences</li>
        </ul>
      </Alert>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Platform Configuration
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="API URL"
                value={settings.apiUrl}
                onChange={(e) => setSettings(prev => ({ ...prev, apiUrl: e.target.value }))}
                margin="normal"
                disabled
              />
              <TextField
                fullWidth
                label="Default Video Device"
                value={settings.videoDevice}
                onChange={(e) => setSettings(prev => ({ ...prev, videoDevice: e.target.value }))}
                margin="normal"
                disabled
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Default MAC Address"
                value={settings.macAddress}
                onChange={(e) => setSettings(prev => ({ ...prev, macAddress: e.target.value }))}
                margin="normal"
                disabled
              />
              <TextField
                fullWidth
                label="Model Timeout (seconds)"
                type="number"
                value={settings.modelTimeout}
                onChange={(e) => setSettings(prev => ({ ...prev, modelTimeout: parseInt(e.target.value) }))}
                margin="normal"
                disabled
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.autoCapture}
                    onChange={(e) => setSettings(prev => ({ ...prev, autoCapture: e.target.checked }))}
                    disabled
                  />
                }
                label="Enable auto-capture mode"
              />
            </Grid>
          </Grid>

          <Divider sx={{ my: 3 }} />

          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            disabled
          >
            Save Settings
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Settings;