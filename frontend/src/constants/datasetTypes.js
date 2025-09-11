// Dataset type definitions
export const DATASET_TYPES = {
  VISION_LLM: 'vision_llm',
  OBJECT_DETECTION: 'object_detection',
  IMAGE_CLASSIFICATION: 'image_classification',
  PADDLEOCR: 'paddleocr'
};

// Dataset type metadata
export const DATASET_TYPE_INFO = {
  [DATASET_TYPES.VISION_LLM]: {
    name: 'Vision LLM',
    description: 'For training vision-language models (e.g., LLaVA)',
    icon: '🤖',
    supportedFormats: ['llava', 'json'],
    requiresBoundingBoxes: false,
    requiresTextLabels: true,
    multiClass: true
  },
  [DATASET_TYPES.OBJECT_DETECTION]: {
    name: 'Object Detection',
    description: 'For training YOLO models to detect UI elements',
    icon: '🎯',
    supportedFormats: ['yolo', 'coco'],
    requiresBoundingBoxes: true,
    requiresTextLabels: false,
    multiClass: true
  },
  [DATASET_TYPES.IMAGE_CLASSIFICATION]: {
    name: 'Image Classification',
    description: 'For classifying screen states and anomalies',
    icon: '🏷️',
    supportedFormats: ['folder_structure', 'csv'],
    requiresBoundingBoxes: false,
    requiresTextLabels: false,
    multiClass: false
  },
  [DATASET_TYPES.PADDLEOCR]: {
    name: 'PaddleOCR Fine-tuning',
    description: 'For fine-tuning PaddleOCR models for text detection and recognition',
    icon: '📝',
    supportedFormats: ['paddleocr'],
    requiresBoundingBoxes: true,
    requiresTextLabels: true,
    multiClass: false
  }
};

// Object detection classes for TV/STB
export const OBJECT_DETECTION_CLASSES = {
  // UI Elements
  button: { id: 0, name: 'Button', color: '#FF6B6B' },
  tile: { id: 1, name: 'Tile/Card', color: '#4ECDC4' },
  menu: { id: 2, name: 'Menu', color: '#45B7D1' },
  dialog: { id: 3, name: 'Dialog/Modal', color: '#96CEB4' },
  keyboard: { id: 4, name: 'Virtual Keyboard', color: '#DDA0DD' },
  
  // Focus indicators
  focused_button: { id: 5, name: 'Focused Button', color: '#FFD93D' },
  focused_tile: { id: 6, name: 'Focused Tile', color: '#6BCB77' },
  
  // Media elements
  video_player: { id: 7, name: 'Video Player', color: '#FF6B9D' },
  progress_bar: { id: 8, name: 'Progress Bar', color: '#C44569' },
  buffering_spinner: { id: 9, name: 'Buffering Animation', color: '#F8B500' },
  
  // Text elements
  closed_caption: { id: 10, name: 'Closed Captions', color: '#786FA6' },
  subtitle: { id: 11, name: 'Subtitles', color: '#303952' },
  
  // Navigation
  rail: { id: 12, name: 'Navigation Rail', color: '#574B90' },
  tab_bar: { id: 13, name: 'Tab Bar', color: '#F97F51' },
  
  // Status indicators
  loading_indicator: { id: 14, name: 'Loading Indicator', color: '#25CCF7' },
  error_message: { id: 15, name: 'Error Message', color: '#EE5A24' },
  notification: { id: 16, name: 'Notification', color: '#009432' }
};

// Image classification classes for TV/STB
export const CLASSIFICATION_CLASSES = {
  // Normal states
  normal_playback: { name: 'Normal Playback', description: 'Video playing normally' },
  home_screen: { name: 'Home Screen', description: 'Main home/launcher screen' },
  app_screen: { name: 'App Screen', description: 'Inside an application' },
  
  // Error states
  black_screen: { name: 'Black Screen', description: 'Complete black screen' },
  blue_screen: { name: 'Blue Screen', description: 'Blue or solid color screen' },
  error_screen: { name: 'Error Screen', description: 'Error message displayed' },
  
  // Quality issues
  pixelation: { name: 'Pixelation', description: 'Video quality degraded with blocks' },
  blur_screen: { name: 'Blur/Fuzzy', description: 'Blurred or out of focus content' },
  artifacts: { name: 'Artifacts', description: 'Visual artifacts or corruption' },
  
  // Loading states
  buffering: { name: 'Buffering', description: 'Video buffering/loading' },
  loading_screen: { name: 'Loading Screen', description: 'App or content loading' },
  
  // Special states
  screensaver: { name: 'Screensaver', description: 'Screensaver active' },
  standby: { name: 'Standby', description: 'Device in standby mode' },
  no_signal: { name: 'No Signal', description: 'No input signal' }
};

// PaddleOCR text types for TV/STB interfaces
export const PADDLEOCR_TEXT_TYPES = {
  // UI Text Elements
  button_text: { id: 0, name: 'Button Text', color: '#FF6B6B' },
  menu_item: { id: 1, name: 'Menu Item', color: '#4ECDC4' },
  title_text: { id: 2, name: 'Title/Header', color: '#45B7D1' },
  body_text: { id: 3, name: 'Body Text', color: '#96CEB4' },
  
  // Media Text
  channel_name: { id: 4, name: 'Channel Name', color: '#DDA0DD' },
  program_title: { id: 5, name: 'Program Title', color: '#FFD93D' },
  time_display: { id: 6, name: 'Time Display', color: '#6BCB77' },
  channel_number: { id: 7, name: 'Channel Number', color: '#FF6B9D' },
  
  // Subtitles and Captions
  subtitle_text: { id: 8, name: 'Subtitle Text', color: '#C44569' },
  caption_text: { id: 9, name: 'Caption Text', color: '#F8B500' },
  
  // Navigation and Controls
  navigation_text: { id: 10, name: 'Navigation Text', color: '#786FA6' },
  status_text: { id: 11, name: 'Status Text', color: '#303952' },
  
  // Error and Information
  error_text: { id: 12, name: 'Error Text', color: '#574B90' },
  info_text: { id: 13, name: 'Information Text', color: '#F97F51' },
  notification_text: { id: 14, name: 'Notification Text', color: '#25CCF7' },
  
  // Generic
  other_text: { id: 15, name: 'Other Text', color: '#EE5A24' }
};

// Data augmentation options
export const AUGMENTATION_OPTIONS = {
  [DATASET_TYPES.VISION_LLM]: {
    brightness: { enabled: true, range: [-30, 30] },
    contrast: { enabled: true, range: [0.7, 1.3] },
    rotation: { enabled: false }, // Usually not needed for TV screens
    flip: { enabled: false }, // TV screens shouldn't be flipped
    noise: { enabled: true, amount: 'low' },
    blur: { enabled: true, amount: 'low' }
  },
  [DATASET_TYPES.OBJECT_DETECTION]: {
    brightness: { enabled: true, range: [-20, 20] },
    contrast: { enabled: true, range: [0.8, 1.2] },
    rotation: { enabled: false },
    flip: { enabled: false },
    scale: { enabled: true, range: [0.9, 1.1] }, // Slight scaling for robustness
    noise: { enabled: false }, // Can affect small objects
    mosaic: { enabled: true } // YOLO-specific augmentation
  },
  [DATASET_TYPES.IMAGE_CLASSIFICATION]: {
    brightness: { enabled: true, range: [-40, 40] },
    contrast: { enabled: true, range: [0.6, 1.4] },
    rotation: { enabled: false },
    flip: { enabled: false },
    noise: { enabled: true, amount: 'medium' },
    blur: { enabled: true, amount: 'medium' },
    compression: { enabled: true, quality: [70, 95] } // For artifact detection
  },
  [DATASET_TYPES.PADDLEOCR]: {
    brightness: { enabled: true, range: [-15, 15] },
    contrast: { enabled: true, range: [0.9, 1.1] },
    rotation: { enabled: true, range: [-2, 2] }, // Small rotation for text
    flip: { enabled: false }, // Don't flip text
    noise: { enabled: false }, // Can affect text recognition
    blur: { enabled: false }, // Can affect text clarity
    perspective: { enabled: true, amount: 'low' }, // OCR-specific augmentation
    elastic_transform: { enabled: true, amount: 'low' } // OCR-specific augmentation
  }
};

// Export format specifications
export const EXPORT_FORMATS = {
  yolo: {
    name: 'YOLO Format',
    description: 'Darknet YOLO format with .txt annotations',
    fileStructure: {
      images: 'images/',
      labels: 'labels/',
      config: ['classes.txt', 'train.txt', 'val.txt']
    }
  },
  coco: {
    name: 'COCO Format',
    description: 'COCO JSON format for object detection',
    fileStructure: {
      annotations: 'annotations.json',
      images: 'images/'
    }
  },
  llava: {
    name: 'LLaVA Format',
    description: 'JSON format for vision-language models',
    fileStructure: {
      data: 'dataset.json',
      images: 'images/'
    }
  },
  folder_structure: {
    name: 'Folder Structure',
    description: 'Images organized in class folders',
    fileStructure: {
      train: 'train/{class_name}/',
      val: 'val/{class_name}/',
      test: 'test/{class_name}/'
    }
  },
  csv: {
    name: 'CSV Format',
    description: 'CSV file with image paths and labels',
    fileStructure: {
      data: 'labels.csv',
      images: 'images/'
    }
  },
  paddleocr: {
    name: 'PaddleOCR Format',
    description: 'PaddleOCR training format with text detection and recognition',
    fileStructure: {
      images: 'images/',
      labels: 'labels/',
      train_list: 'train_list.txt',
      val_list: 'val_list.txt',
      rec_gt: 'rec_gt_train.txt',
      det_gt: 'det_gt_train.txt'
    }
  }
};