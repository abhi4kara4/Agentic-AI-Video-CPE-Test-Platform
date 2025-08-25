import React, { useState, useEffect } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate
} from 'react-router-dom';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  Alert,
  Snackbar,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
} from '@mui/material';
import {
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

// Import components
import Navigation from './components/Navigation.jsx';
import DatasetCreation from './components/DatasetCreation.jsx';
import ModelTraining from './components/ModelTraining.jsx';
import ModelTesting from './components/ModelTesting.jsx';
import Dashboard from './components/Dashboard.jsx';
import Settings from './components/Settings.jsx';

// Import services
import { healthAPI, wsManager } from './services/api.jsx';

// Create theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
    success: {
      main: '#2e7d32',
    },
    warning: {
      main: '#ed6c02',
    },
    error: {
      main: '#d32f2f',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontWeight: 600,
    },
    h2: {
      fontWeight: 600,
    },
    h3: {
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
          },
        },
      },
    },
  },
});

function App() {
  const [platformStatus, setPlatformStatus] = useState('checking');
  const [notifications, setNotifications] = useState([]);
  const [currentNotification, setCurrentNotification] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);

  // Initialize platform
  useEffect(() => {
    initializePlatform();
    
    // Set up WebSocket connection
    wsManager.connect();
    
    wsManager.on('connected', () => {
      setWsConnected(true);
    });
    
    wsManager.on('disconnected', () => {
      setWsConnected(false);
    });
    
    wsManager.on('notification', (notification) => {
      addNotification(notification);
    });
    
    wsManager.on('training_update', (update) => {
      addNotification({
        type: 'info',
        title: 'Training Update',
        message: `${update.job_name}: ${update.status}`,
        timestamp: new Date().toISOString(),
      });
    });
    
    wsManager.on('error', (error) => {
      addNotification({
        type: 'error',
        title: 'Connection Error',
        message: 'WebSocket connection lost. Some features may not work.',
        timestamp: new Date().toISOString(),
      });
    });
    
    return () => {
      wsManager.disconnect();
    };
  }, []);

  const initializePlatform = async () => {
    try {
      const response = await healthAPI.checkHealth();
      if (response.data.status === 'healthy') {
        setPlatformStatus('ready');
        addNotification({
          type: 'success',
          title: 'Platform Ready',
          message: 'AI Video Test Platform initialized successfully',
          timestamp: new Date().toISOString(),
        });
      } else {
        setPlatformStatus('error');
        addNotification({
          type: 'warning',
          title: 'Platform Issues',
          message: 'Some components are not ready',
          timestamp: new Date().toISOString(),
        });
      }
    } catch (error) {
      setPlatformStatus('error');
      addNotification({
        type: 'error',
        title: 'Connection Failed',
        message: 'Cannot connect to backend services',
        timestamp: new Date().toISOString(),
      });
    }
  };

  const addNotification = (notification) => {
    const newNotification = {
      ...notification,
      id: Date.now() + Math.random(),
      timestamp: notification.timestamp || new Date().toISOString(),
    };
    
    setNotifications(prev => [newNotification, ...prev.slice(0, 99)]); // Keep last 100
    setCurrentNotification(newNotification);
  };

  const handleCloseNotification = () => {
    setCurrentNotification(null);
  };

  const getStatusColor = () => {
    switch (platformStatus) {
      case 'ready': return 'success';
      case 'error': return 'error';
      case 'checking': return 'warning';
      default: return 'default';
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
          {/* Navigation Sidebar */}
          <Navigation 
            platformStatus={platformStatus}
            wsConnected={wsConnected}
          />
          
          {/* Main Content */}
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              bgcolor: 'background.default',
              overflow: 'hidden',
            }}
          >
            {/* Top App Bar */}
            <AppBar 
              position="static" 
              elevation={1}
              sx={{ 
                bgcolor: 'background.paper', 
                color: 'text.primary',
                borderBottom: '1px solid',
                borderBottomColor: 'divider',
              }}
            >
              <Toolbar>
                <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 600 }}>
                  AI Video Test Platform
                </Typography>
                
                {/* Status Indicator */}
                <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      bgcolor: wsConnected ? 'success.main' : 'error.main',
                      mr: 1,
                    }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {wsConnected ? 'Connected' : 'Disconnected'}
                  </Typography>
                </Box>
                
                {/* Notifications */}
                <IconButton color="inherit">
                  <Badge badgeContent={notifications.length > 0 ? notifications.length : null} color="error">
                    <NotificationsIcon />
                  </Badge>
                </IconButton>
                
                {/* Settings */}
                <IconButton color="inherit">
                  <SettingsIcon />
                </IconButton>
              </Toolbar>
            </AppBar>

            {/* Routes */}
            <Box sx={{ height: 'calc(100vh - 64px)', overflow: 'auto' }}>
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route 
                  path="/dashboard" 
                  element={
                    <Dashboard 
                      platformStatus={platformStatus}
                      notifications={notifications}
                    />
                  } 
                />
                <Route 
                  path="/dataset" 
                  element={
                    <DatasetCreation 
                      onNotification={addNotification}
                    />
                  } 
                />
                <Route 
                  path="/training" 
                  element={
                    <ModelTraining 
                      onNotification={addNotification}
                    />
                  } 
                />
                <Route 
                  path="/testing" 
                  element={
                    <ModelTesting 
                      onNotification={addNotification}
                    />
                  } 
                />
                <Route 
                  path="/settings" 
                  element={
                    <Settings 
                      onNotification={addNotification}
                    />
                  } 
                />
              </Routes>
            </Box>
          </Box>
        </Box>

        {/* Notification Snackbar */}
        <Snackbar
          open={!!currentNotification}
          autoHideDuration={6000}
          onClose={handleCloseNotification}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          {currentNotification && (
            <Alert
              onClose={handleCloseNotification}
              severity={currentNotification.type}
              variant="filled"
              sx={{ width: '100%' }}
            >
              <strong>{currentNotification.title}</strong>
              <br />
              {currentNotification.message}
            </Alert>
          )}
        </Snackbar>
      </Router>
    </ThemeProvider>
  );
}

export default App;