import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
  Typography,
  Chip,
  Avatar,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Dataset as DatasetIcon,
  ModelTraining as TrainingIcon,
  Science as TestingIcon,
  Settings as SettingsIcon,
  VideoLibrary as VideoIcon,
  Memory as AIIcon,
  Cable as ConnectionIcon,
} from '@mui/icons-material';

const DRAWER_WIDTH = 280;

const Navigation = ({ platformStatus, wsConnected }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const menuItems = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: <DashboardIcon />,
      path: '/dashboard',
      description: 'Overview & Status',
    },
    {
      id: 'dataset',
      label: 'Dataset Creation',
      icon: <DatasetIcon />,
      path: '/dataset',
      description: 'Annotate TV Screens',
    },
    {
      id: 'training',
      label: 'Model Training',
      icon: <TrainingIcon />,
      path: '/training',
      description: 'Train Vision Models',
    },
    {
      id: 'testing',
      label: 'Model Testing',
      icon: <TestingIcon />,
      path: '/testing',
      description: 'Test & Validate',
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: <SettingsIcon />,
      path: '/settings',
      description: 'Configuration',
    },
  ];

  const handleNavigation = (path) => {
    navigate(path);
  };

  const getStatusColor = () => {
    switch (platformStatus) {
      case 'ready': return 'success';
      case 'error': return 'error';
      case 'checking': return 'warning';
      default: return 'default';
    }
  };

  const getStatusText = () => {
    switch (platformStatus) {
      case 'ready': return 'Ready';
      case 'error': return 'Error';
      case 'checking': return 'Checking';
      default: return 'Unknown';
    }
  };

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: DRAWER_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: DRAWER_WIDTH,
          boxSizing: 'border-box',
          bgcolor: 'background.paper',
          borderRight: '1px solid',
          borderRightColor: 'divider',
        },
      }}
    >
      {/* Header */}
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Avatar
          sx={{
            width: 56,
            height: 56,
            bgcolor: 'primary.main',
            mx: 'auto',
            mb: 2,
          }}
        >
          <AIIcon fontSize="large" />
        </Avatar>
        <Typography variant="h6" fontWeight="bold" gutterBottom>
          AI Video Platform
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          TV/STB Vision Training
        </Typography>
        
        {/* Status Chips */}
        <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center', mt: 2 }}>
          <Chip
            size="small"
            icon={<VideoIcon />}
            label={getStatusText()}
            color={getStatusColor()}
            variant="outlined"
          />
          <Chip
            size="small"
            icon={<ConnectionIcon />}
            label={wsConnected ? 'Live' : 'Offline'}
            color={wsConnected ? 'success' : 'error'}
            variant="outlined"
          />
        </Box>
      </Box>

      <Divider />

      {/* Navigation Menu */}
      <Box sx={{ flexGrow: 1, py: 2 }}>
        <List>
          {menuItems.map((item) => {
            const isActive = location.pathname === item.path;
            
            return (
              <ListItem key={item.id} disablePadding>
                <ListItemButton
                  selected={isActive}
                  onClick={() => handleNavigation(item.path)}
                  sx={{
                    mx: 1,
                    borderRadius: 2,
                    '&.Mui-selected': {
                      bgcolor: 'primary.main',
                      color: 'primary.contrastText',
                      '&:hover': {
                        bgcolor: 'primary.dark',
                      },
                      '& .MuiListItemIcon-root': {
                        color: 'inherit',
                      },
                    },
                    '&:hover': {
                      bgcolor: isActive ? 'primary.dark' : 'action.hover',
                    },
                  }}
                >
                  <ListItemIcon
                    sx={{
                      color: isActive ? 'inherit' : 'text.secondary',
                      minWidth: 40,
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.label}
                    secondary={!isActive ? item.description : null}
                    primaryTypographyProps={{
                      fontWeight: isActive ? 600 : 400,
                    }}
                    secondaryTypographyProps={{
                      fontSize: '0.75rem',
                      color: isActive ? 'inherit' : 'text.secondary',
                      sx: { opacity: isActive ? 0.7 : 1 },
                    }}
                  />
                </ListItemButton>
              </ListItem>
            );
          })}
        </List>
      </Box>

      <Divider />

      {/* Footer Info */}
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" display="block">
          Version 1.0.0
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block">
          Powered by LLaVA & Ollama
        </Typography>
      </Box>
    </Drawer>
  );
};

export default Navigation;