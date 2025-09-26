import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Grid,
  Alert,
  CircularProgress,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Tooltip
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Palette as ColorIcon,
  Class as ClassIcon
} from '@mui/icons-material';
import { datasetAPI } from '../services/api.jsx';

const ClassManagement = ({ 
  open, 
  onClose, 
  datasetName, 
  onNotification,
  onClassesUpdated 
}) => {
  const [classes, setClasses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [editingClass, setEditingClass] = useState(null);
  const [newClassName, setNewClassName] = useState('');
  const [addingClass, setAddingClass] = useState(false);
  const [newClassToAdd, setNewClassToAdd] = useState('');
  const [newClassColor, setNewClassColor] = useState('#FF6B6B');
  const [bulkOperations, setBulkOperations] = useState([]);

  // Color palette for new classes
  const colorPalette = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD',
    '#FFD93D', '#6BCB77', '#FF6B9D', '#C44569', '#F8B500',
    '#786FA6', '#303952', '#574B90', '#F97F51', '#25CCF7',
    '#EE5A24', '#009432', '#8395A7', '#FD79A8', '#00B894'
  ];

  useEffect(() => {
    if (open && datasetName) {
      loadClasses();
    }
  }, [open, datasetName]);

  const loadClasses = async () => {
    setLoading(true);
    try {
      const response = await datasetAPI.getDatasetClasses(datasetName);
      setClasses(response.data.classes);
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Load Error',
        message: `Failed to load classes: ${error.response?.data?.detail || error.message}`
      });
    } finally {
      setLoading(false);
    }
  };

  const handleStartEdit = (className) => {
    setEditingClass(className);
    setNewClassName(className);
  };

  const handleCancelEdit = () => {
    setEditingClass(null);
    setNewClassName('');
  };

  const handleSaveRename = async () => {
    if (!newClassName.trim() || newClassName === editingClass) {
      handleCancelEdit();
      return;
    }

    try {
      const response = await datasetAPI.renameClassInDataset(
        datasetName, 
        editingClass, 
        newClassName.trim()
      );
      
      // Update local state
      setClasses(prev => prev.map(cls => cls === editingClass ? newClassName.trim() : cls));
      
      onNotification({
        type: 'success',
        title: 'Class Renamed',
        message: `Successfully renamed '${editingClass}' to '${newClassName.trim()}'. Updated ${response.data.updated_annotations} annotations in ${response.data.updated_files} files.`
      });
      
      setEditingClass(null);
      setNewClassName('');
      
      if (onClassesUpdated) {
        onClassesUpdated();
      }
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Rename Error',
        message: `Failed to rename class: ${error.response?.data?.detail || error.message}`
      });
    }
  };

  const handleAddClass = async () => {
    if (!newClassToAdd.trim()) return;

    try {
      const response = await datasetAPI.addClassToDataset(
        datasetName, 
        newClassToAdd.trim(), 
        newClassColor
      );
      
      // Update local state
      setClasses(prev => [...prev, newClassToAdd.trim()]);
      
      onNotification({
        type: 'success',
        title: 'Class Added',
        message: `Successfully added class '${newClassToAdd.trim()}'`
      });
      
      setAddingClass(false);
      setNewClassToAdd('');
      setNewClassColor('#FF6B6B');
      
      if (onClassesUpdated) {
        onClassesUpdated();
      }
    } catch (error) {
      onNotification({
        type: 'error',
        title: 'Add Error',
        message: `Failed to add class: ${error.response?.data?.detail || error.message}`
      });
    }
  };

  const getRandomColor = () => {
    return colorPalette[Math.floor(Math.random() * colorPalette.length)];
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="md" 
      fullWidth
      PaperProps={{
        sx: { minHeight: 400 }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ClassIcon />
          <Typography variant="h6">
            Manage Classes - {datasetName}
          </Typography>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <Box>
            <Alert severity="info" sx={{ mb: 2 }}>
              Rename existing classes to update all annotations across the dataset. 
              Add new classes to make them available for future annotations.
            </Alert>

            {/* Current Classes */}
            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="h6" gutterBottom>
                Current Classes ({classes.length})
              </Typography>
              
              {classes.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No classes found in this dataset.
                </Typography>
              ) : (
                <List>
                  {classes.map((className, index) => (
                    <ListItem key={index} divider>
                      {editingClass === className ? (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                          <TextField
                            value={newClassName}
                            onChange={(e) => setNewClassName(e.target.value)}
                            size="small"
                            fullWidth
                            autoFocus
                            onKeyPress={(e) => {
                              if (e.key === 'Enter') {
                                handleSaveRename();
                              } else if (e.key === 'Escape') {
                                handleCancelEdit();
                              }
                            }}
                          />
                          <Tooltip title="Save">
                            <IconButton 
                              onClick={handleSaveRename}
                              color="primary"
                              size="small"
                            >
                              <SaveIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Cancel">
                            <IconButton 
                              onClick={handleCancelEdit}
                              size="small"
                            >
                              <CancelIcon />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      ) : (
                        <>
                          <ListItemText
                            primary={
                              <Chip
                                label={className}
                                size="small"
                                sx={{ 
                                  backgroundColor: getRandomColor(),
                                  color: 'white'
                                }}
                              />
                            }
                          />
                          <ListItemSecondaryAction>
                            <Tooltip title="Rename class">
                              <IconButton 
                                onClick={() => handleStartEdit(className)}
                                size="small"
                              >
                                <EditIcon />
                              </IconButton>
                            </Tooltip>
                          </ListItemSecondaryAction>
                        </>
                      )}
                    </ListItem>
                  ))}
                </List>
              )}
            </Paper>

            {/* Add New Class */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Add New Class
              </Typography>
              
              {addingClass ? (
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                  <TextField
                    label="Class Name"
                    value={newClassToAdd}
                    onChange={(e) => setNewClassToAdd(e.target.value)}
                    size="small"
                    autoFocus
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        handleAddClass();
                      }
                    }}
                  />
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ColorIcon fontSize="small" />
                    <input
                      type="color"
                      value={newClassColor}
                      onChange={(e) => setNewClassColor(e.target.value)}
                      style={{
                        width: 40,
                        height: 32,
                        border: 'none',
                        borderRadius: 4,
                        cursor: 'pointer'
                      }}
                    />
                  </Box>
                  
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant="contained"
                      onClick={handleAddClass}
                      disabled={!newClassToAdd.trim()}
                      size="small"
                      startIcon={<SaveIcon />}
                    >
                      Add
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={() => {
                        setAddingClass(false);
                        setNewClassToAdd('');
                        setNewClassColor('#FF6B6B');
                      }}
                      size="small"
                    >
                      Cancel
                    </Button>
                  </Box>
                </Box>
              ) : (
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={() => setAddingClass(true)}
                >
                  Add New Class
                </Button>
              )}
            </Paper>
          </Box>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ClassManagement;