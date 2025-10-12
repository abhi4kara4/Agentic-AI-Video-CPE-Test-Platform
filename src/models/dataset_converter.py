"""
Convert existing object detection datasets to YOLO format for training
"""
import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random


def polygon_to_bbox(points: List[Dict]) -> Tuple[float, float, float, float]:
    """
    Convert polygon points to minimum bounding rectangle
    
    Args:
        points: List of points with x, y coordinates (normalized 0-1)
    
    Returns:
        Tuple of (x, y, width, height) for bounding box (normalized 0-1)
    """
    if not points or len(points) < 3:
        return 0, 0, 0, 0
    
    x_coords = [p['x'] for p in points]
    y_coords = [p['y'] for p in points]
    
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    # Ensure coordinates are within bounds
    min_x = max(0, min(1, min_x))
    max_x = max(0, min(1, max_x))
    min_y = max(0, min(1, min_y))
    max_y = max(0, min(1, max_y))
    
    width = max_x - min_x
    height = max_y - min_y
    
    return min_x, min_y, width, height


def convert_annotations_to_yolo(annotations: List[Dict], image_width: int, image_height: int, class_mapping: Dict[str, int]) -> List[str]:
    """
    Convert bounding box and polygon annotations to YOLO format
    
    Args:
        annotations: List of annotations with either:
                    - Rectangle: x, y, width, height (normalized 0-1)
                    - Polygon: points array with x, y coordinates (normalized 0-1)
        image_width: Original image width
        image_height: Original image height
        class_mapping: Mapping from class names to class indices
    
    Returns:
        List of YOLO format strings
    """
    yolo_lines = []
    
    for ann in annotations:
        class_name = ann.get('class', 'button')
        if class_name not in class_mapping:
            continue
            
        class_id = class_mapping[class_name]
        
        # Handle both polygon and rectangle annotations
        if 'points' in ann and ann['points']:
            # Polygon annotation - convert to bounding box
            x, y, w, h = polygon_to_bbox(ann['points'])
            print(f"Converted polygon with {len(ann['points'])} points to bbox: ({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")
        else:
            # Rectangle annotation
            x = ann.get('x', 0)  # top-left x (normalized)
            y = ann.get('y', 0)  # top-left y (normalized)
            w = ann.get('width', 0)  # width (normalized)
            h = ann.get('height', 0)  # height (normalized)
        
        # Skip invalid bounding boxes
        if w <= 0 or h <= 0:
            print(f"Warning: Skipping invalid bounding box with dimensions {w}x{h}")
            continue
        
        # Convert to center coordinates for YOLO format
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Ensure coordinates are within bounds
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}"
        yolo_lines.append(yolo_line)
    
    return yolo_lines


def create_yolo_dataset(dataset_name: str, train_split: float = 0.8, class_mapping: Dict[str, int] = None) -> str:
    """
    Convert an existing object detection dataset to YOLO format
    
    Args:
        dataset_name: Name of the dataset in datasets/ directory
        train_split: Fraction of data to use for training (rest for validation)
        class_mapping: Optional mapping of class names to indices
    
    Returns:
        Path to the created YOLO dataset directory
    """
    datasets_dir = Path("datasets")
    source_dataset = datasets_dir / dataset_name
    yolo_dataset = datasets_dir / f"{dataset_name}_yolo"
    
    if not source_dataset.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset}")
    
    print(f"Converting dataset {dataset_name} to YOLO format...")
    
    # Create YOLO directory structure
    yolo_dataset.mkdir(exist_ok=True)
    (yolo_dataset / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dataset / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_dataset / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dataset / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Find all images with annotations
    images_dir = source_dataset / "images"
    annotations_dir = source_dataset / "annotations"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Collect all annotated images
    annotated_images = []
    all_classes = set()
    
    for image_file in images_dir.glob("*.jpg"):
        annotation_file = annotations_dir / f"{image_file.stem}.json"
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
                
                # Extract bounding boxes
                bounding_boxes = annotation_data.get('bounding_boxes', [])
                if bounding_boxes:
                    annotated_images.append((image_file, annotation_data))
                    # Collect all class names
                    for box in bounding_boxes:
                        all_classes.add(box.get('class', 'button'))
                        
            except Exception as e:
                print(f"Warning: Failed to read annotation {annotation_file}: {e}")
                continue
    
    if not annotated_images:
        raise ValueError(f"No annotated images found in dataset {dataset_name}")
    
    print(f"Found {len(annotated_images)} annotated images")
    print(f"Classes found: {sorted(all_classes)}")
    
    # Create class mapping if not provided
    if class_mapping is None:
        class_mapping = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    
    # Split into train/val
    random.shuffle(annotated_images)
    split_index = int(len(annotated_images) * train_split)
    train_images = annotated_images[:split_index]
    val_images = annotated_images[split_index:]
    
    print(f"Train set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    # Process training images
    for image_file, annotation_data in train_images:
        # Copy image
        dest_image = yolo_dataset / "images" / "train" / image_file.name
        shutil.copy2(image_file, dest_image)
        
        # Convert annotations
        bounding_boxes = annotation_data.get('bounding_boxes', [])
        yolo_lines = convert_annotations_to_yolo(bounding_boxes, 1, 1, class_mapping)  # Already normalized
        
        # Save YOLO annotation
        label_file = yolo_dataset / "labels" / "train" / f"{image_file.stem}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    # Process validation images
    for image_file, annotation_data in val_images:
        # Copy image
        dest_image = yolo_dataset / "images" / "val" / image_file.name
        shutil.copy2(image_file, dest_image)
        
        # Convert annotations
        bounding_boxes = annotation_data.get('bounding_boxes', [])
        yolo_lines = convert_annotations_to_yolo(bounding_boxes, 1, 1, class_mapping)  # Already normalized
        
        # Save YOLO annotation
        label_file = yolo_dataset / "labels" / "val" / f"{image_file.stem}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    # Create data.yaml
    class_names = [name for name, idx in sorted(class_mapping.items(), key=lambda x: x[1])]
    yaml_content = {
        'path': str(yolo_dataset.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = yolo_dataset / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"✅ YOLO dataset created successfully at: {yolo_dataset}")
    print(f"📄 Dataset configuration: {yaml_path}")
    print(f"🏷️  Classes ({len(class_names)}): {class_names}")
    
    return str(yolo_dataset)


def prepare_dataset_for_training(dataset_name: str) -> str:
    """
    Prepare an existing dataset for YOLO training
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Path to the prepared YOLO dataset
    """
    try:
        yolo_dataset_path = create_yolo_dataset(dataset_name)
        return yolo_dataset_path
    except Exception as e:
        raise Exception(f"Failed to prepare dataset {dataset_name} for training: {str(e)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python dataset_converter.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    try:
        yolo_path = prepare_dataset_for_training(dataset_name)
        print(f"Dataset prepared successfully: {yolo_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)