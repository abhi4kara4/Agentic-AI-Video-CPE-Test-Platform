import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Button,
  Alert,
} from '@mui/material';
import {
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Memory as AIIcon,
  VideoLibrary as VideoIcon,
  Settings as DeviceIcon,
  Dataset as DatasetIcon,
} from '@mui/icons-material';
import { healthAPI, datasetAPI, trainingAPI } from '../services/api.jsx';

const Dashboard = ({ platformStatus, notifications }) => {
  const [stats, setStats] = useState({
    datasets: 0,
    trainedModels: 0,
    totalImages: 0,
    activeTraining: 0,
  });
  
  const [systemHealth, setSystemHealth] = useState({
    video_capture: 'unknown',
    device_controller: 'unknown',
    vision_agent: 'unknown',
  });

  useEffect(() => {
    loadDashboardData();
    const interval = setInterval(loadDashboardData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    try {
      // Load system health
      const healthResponse = await healthAPI.checkHealth();
      setSystemHealth(healthResponse.data.components);

      // Load datasets count
      const datasetsResponse = await datasetAPI.listDatasets();
      const datasets = Array.isArray(datasetsResponse.data?.datasets) 
        ? datasetsResponse.data.datasets 
        : [];
      const totalImages = datasets.reduce((sum, dataset) => sum + (dataset.image_count || 0), 0);

      // Load models count
      const modelsResponse = await trainingAPI.listModels();
      const models = Array.isArray(modelsResponse.data?.models) 
        ? modelsResponse.data.models 
        : [];

      // Load active training jobs
      const trainingResponse = await trainingAPI.listTrainingJobs();
      const jobs = Array.isArray(trainingResponse.data?.jobs) 
        ? trainingResponse.data.jobs 
        : [];
      const activeJobs = jobs.filter(job => 
        ['running', 'pending'].includes(job.status)
      ).length;

      setStats({
        datasets: datasets.length,
        trainedModels: Object.keys(models).length, // models is an object, not array
        totalImages,
        activeTraining: activeJobs,
      });
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
      case true:
        return <SuccessIcon color="success" />;
      case 'error':
      case false:
        return <ErrorIcon color="error" />;
      default:
        return <WarningIcon color="warning" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
      case true:
        return 'success';
      case 'error':
      case false:
        return 'error';
      default:
        return 'warning';
    }
  };

  const getNotificationIcon = (type) => {
    switch (type) {
      case 'success':
        return <SuccessIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          AI Video Test Platform Overview
        </Typography>
      </Box>

      {/* Platform Status Alert */}
      {platformStatus !== 'ready' && (
        <Alert 
          severity={platformStatus === 'error' ? 'error' : 'warning'} 
          sx={{ mb: 3 }}
        >
          {platformStatus === 'error' 
            ? 'Platform has errors. Some features may not work properly.'
            : 'Platform is starting up. Please wait...'
          }
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Stats Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <DatasetIcon color="primary" sx={{ mr: 1 }} />
                <Typography color="text.secondary" variant="h6">
                  Datasets
                </Typography>
              </Box>
              <Typography variant="h3" fontWeight="bold">
                {stats.datasets}
              </Typography>
              <Typography color="text.secondary" variant="body2">
                {stats.totalImages} total images
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <AIIcon color="secondary" sx={{ mr: 1 }} />
                <Typography color="text.secondary" variant="h6">
                  Models
                </Typography>
              </Box>
              <Typography variant="h3" fontWeight="bold">
                {stats.trainedModels}
              </Typography>
              <Typography color="text.secondary" variant="body2">
                Trained models
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <VideoIcon color="success" sx={{ mr: 1 }} />
                <Typography color="text.secondary" variant="h6">
                  Active Jobs
                </Typography>
              </Box>
              <Typography variant="h3" fontWeight="bold">
                {stats.activeTraining}
              </Typography>
              <Typography color="text.secondary" variant="body2">
                Training in progress
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <DeviceIcon color="info" sx={{ mr: 1 }} />
                <Typography color="text.secondary" variant="h6">
                  Status
                </Typography>
              </Box>
              <Chip
                label={platformStatus}
                color={getStatusColor(platformStatus === 'ready')}
                variant="filled"
                sx={{ fontWeight: 'bold' }}
              />
              <Typography color="text.secondary" variant="body2" sx={{ mt: 1 }}>
                Platform status
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* System Health */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon>
                    {getStatusIcon(systemHealth.video_capture)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Video Capture"
                    secondary={systemHealth.video_capture ? 'Running' : 'Stopped'}
                  />
                  <Chip
                    size="small"
                    label={systemHealth.video_capture ? 'OK' : 'ERROR'}
                    color={getStatusColor(systemHealth.video_capture)}
                    variant="outlined"
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    {getStatusIcon(systemHealth.device_controller)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Device Controller"
                    secondary={systemHealth.device_controller ? 'Connected' : 'Disconnected'}
                  />
                  <Chip
                    size="small"
                    label={systemHealth.device_controller ? 'OK' : 'ERROR'}
                    color={getStatusColor(systemHealth.device_controller)}
                    variant="outlined"
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    {getStatusIcon(systemHealth.vision_agent)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Vision Agent (AI)"
                    secondary={systemHealth.vision_agent ? 'Ready' : 'Not Available'}
                  />
                  <Chip
                    size="small"
                    label={systemHealth.vision_agent ? 'OK' : 'ERROR'}
                    color={getStatusColor(systemHealth.vision_agent)}
                    variant="outlined"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Notifications */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              {notifications.length === 0 ? (
                <Typography color="text.secondary" variant="body2">
                  No recent activity
                </Typography>
              ) : (
                <List dense>
                  {notifications.slice(0, 5).map((notification) => (
                    <ListItem key={notification.id}>
                      <ListItemIcon>
                        {getNotificationIcon(notification.type)}
                      </ListItemIcon>
                      <ListItemText
                        primary={notification.title}
                        secondary={
                          <React.Fragment>
                            <span style={{ display: 'block' }}>
                              {notification.message}
                            </span>
                            <span style={{ display: 'block', fontSize: '0.75rem', marginTop: '4px' }}>
                              {new Date(notification.timestamp).toLocaleString()}
                            </span>
                          </React.Fragment>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<DatasetIcon />}
                    href="/dataset"
                  >
                    Create Dataset
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<AIIcon />}
                    href="/training"
                  >
                    Start Training
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<VideoIcon />}
                    href="/testing"
                  >
                    Test Models
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<DeviceIcon />}
                    href="/settings"
                  >
                    Settings
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

export default Dashboard;