"""
Real Image Augmentation Implementation
Adapted from ML_Training_Platform_Reference for dataset generation
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import List, Dict, Tuple, Any
from pathlib import Path
import json


class ImageAugmentator:
    """Handle real image augmentations for dataset generation"""
    
    def __init__(self, augmentation_config: Dict[str, Any] = None):
        """
        Initialize augmentator with configuration
        
        Args:
            augmentation_config: Configuration dict with augmentation parameters
        """
        self.config = augmentation_config or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default augmentation configuration"""
        return {
            "brightness": {"enabled": True, "range": [-20, 20]},
            "contrast": {"enabled": True, "range": [0.8, 1.2]},
            "rotation": {"enabled": True, "angles": [0, 90, 180, 270]},
            "flip": {"enabled": True, "horizontal": True, "vertical": False},
            "scale": {"enabled": True, "range": [0.9, 1.1]},
            "noise": {"enabled": True, "amount": "low"},
            "blur": {"enabled": True, "amount": "low"}
        }
    
    def apply_brightness(self, image: Image.Image, factor: float = None) -> Image.Image:
        """Apply brightness adjustment"""
        if factor is None:
            brightness_range = self.config.get("brightness", {}).get("range", [-20, 20])
            factor = 1.0 + random.uniform(brightness_range[0], brightness_range[1]) / 100.0
        
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def apply_contrast(self, image: Image.Image, factor: float = None) -> Image.Image:
        """Apply contrast adjustment"""
        if factor is None:
            contrast_range = self.config.get("contrast", {}).get("range", [0.8, 1.2])
            factor = random.uniform(contrast_range[0], contrast_range[1])
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def apply_rotation(self, image: Image.Image, angle: int = None) -> Tuple[Image.Image, Dict]:
        """Apply rotation and return transformation info for bounding boxes"""
        if angle is None:
            angles = self.config.get("rotation", {}).get("angles", [0, 90, 180, 270])
            angle = random.choice(angles)
        
        if angle == 0:
            return image, {"type": "rotation", "angle": 0}
        
        # Rotate image
        rotated = image.rotate(angle, expand=True, fillcolor='black')
        
        return rotated, {"type": "rotation", "angle": angle}
    
    def apply_flip(self, image: Image.Image, horizontal: bool = None, vertical: bool = None) -> Tuple[Image.Image, Dict]:
        """Apply flipping and return transformation info"""
        flip_config = self.config.get("flip", {})
        
        if horizontal is None:
            horizontal = flip_config.get("horizontal", True) and random.choice([True, False])
        if vertical is None:
            vertical = flip_config.get("vertical", False) and random.choice([True, False])
        
        result_image = image
        transforms = {"type": "flip", "horizontal": horizontal, "vertical": vertical}
        
        if horizontal:
            result_image = result_image.transpose(Image.FLIP_LEFT_RIGHT)
        if vertical:
            result_image = result_image.transpose(Image.FLIP_TOP_BOTTOM)
        
        return result_image, transforms
    
    def apply_scaling(self, image: Image.Image, scale_factor: float = None) -> Tuple[Image.Image, Dict]:
        """Apply scaling transformation"""
        if scale_factor is None:
            scale_range = self.config.get("scale", {}).get("range", [0.9, 1.1])
            scale_factor = random.uniform(scale_range[0], scale_range[1])
        
        if abs(scale_factor - 1.0) < 0.01:  # No significant scaling
            return image, {"type": "scale", "factor": 1.0}
        
        # Get original size
        orig_width, orig_height = image.size
        
        # Calculate new size
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        
        # Resize image
        scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # If scaling up, crop to original size from center
        if scale_factor > 1.0:
            left = (new_width - orig_width) // 2
            top = (new_height - orig_height) // 2
            scaled_image = scaled_image.crop((left, top, left + orig_width, top + orig_height))
        
        # If scaling down, pad to original size
        elif scale_factor < 1.0:
            new_image = Image.new('RGB', (orig_width, orig_height), 'black')
            paste_x = (orig_width - new_width) // 2
            paste_y = (orig_height - new_height) // 2
            new_image.paste(scaled_image, (paste_x, paste_y))
            scaled_image = new_image
        
        return scaled_image, {"type": "scale", "factor": scale_factor}
    
    def apply_noise(self, image: Image.Image, amount: str = None) -> Image.Image:
        """Apply noise to image"""
        if amount is None:
            amount = self.config.get("noise", {}).get("amount", "low")
        
        # Convert to numpy
        img_array = np.array(image)
        
        # Determine noise level
        noise_levels = {"low": 10, "medium": 20, "high": 30}
        noise_std = noise_levels.get(amount, 10)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, img_array.shape).astype(np.int16)
        noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    
    def apply_blur(self, image: Image.Image, amount: str = None) -> Image.Image:
        """Apply blur to image"""
        if amount is None:
            amount = self.config.get("blur", {}).get("amount", "low")
        
        # Determine blur radius
        blur_levels = {"low": 0.5, "medium": 1.0, "high": 1.5}
        radius = blur_levels.get(amount, 0.5)
        
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def transform_bounding_boxes(self, bounding_boxes: List[Dict], transformation: Dict, 
                                image_width: int, image_height: int) -> List[Dict]:
        """
        Transform bounding box coordinates based on applied transformations
        
        Args:
            bounding_boxes: List of bounding box dicts with x, y, width, height (normalized 0-1)
            transformation: Dict describing the transformation applied
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            List of transformed bounding boxes
        """
        if not bounding_boxes:
            return []
        
        transformed_boxes = []
        
        for box in bounding_boxes:
            new_box = dict(box)  # Copy original
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            
            if transformation["type"] == "flip":
                if transformation["horizontal"]:
                    # Flip horizontally: x' = 1 - (x + width)
                    new_box['x'] = 1.0 - (x + w)
                if transformation["vertical"]:
                    # Flip vertically: y' = 1 - (y + height)
                    new_box['y'] = 1.0 - (y + h)
            
            elif transformation["type"] == "rotation":
                angle = transformation["angle"]
                if angle == 90:
                    # 90° clockwise: (x,y) -> (1-y-h, x)
                    new_box['x'] = 1.0 - y - h
                    new_box['y'] = x
                    new_box['width'] = h
                    new_box['height'] = w
                elif angle == 180:
                    # 180°: (x,y) -> (1-x-w, 1-y-h)
                    new_box['x'] = 1.0 - x - w
                    new_box['y'] = 1.0 - y - h
                elif angle == 270:
                    # 270° clockwise: (x,y) -> (y, 1-x-w)
                    new_box['x'] = y
                    new_box['y'] = 1.0 - x - w
                    new_box['width'] = h
                    new_box['height'] = w
            
            elif transformation["type"] == "scale":
                scale_factor = transformation["factor"]
                if scale_factor != 1.0:
                    # For scaling, coordinates remain the same in normalized space
                    # (assuming center crop/pad maintains relative positions)
                    pass  # No coordinate transformation needed for center scaling
            
            # Ensure coordinates stay within bounds
            new_box['x'] = max(0, min(1, new_box['x']))
            new_box['y'] = max(0, min(1, new_box['y']))
            new_box['width'] = max(0, min(1 - new_box['x'], new_box['width']))
            new_box['height'] = max(0, min(1 - new_box['y'], new_box['height']))
            
            # Only keep boxes that are still valid size
            if new_box['width'] > 0.01 and new_box['height'] > 0.01:
                transformed_boxes.append(new_box)
        
        return transformed_boxes
    
    def augment_image_with_boxes(self, image_path: str, bounding_boxes: List[Dict], 
                                num_augmentations: int = 3) -> List[Dict]:
        """
        Apply augmentations to an image and transform its bounding boxes
        
        Args:
            image_path: Path to the original image
            bounding_boxes: List of bounding box annotations
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of dicts with augmented image data and transformed bounding boxes
        """
        original_image = Image.open(image_path)
        image_width, image_height = original_image.size
        
        augmented_data = []
        
        for i in range(num_augmentations):
            # Start with original image
            aug_image = original_image.copy()
            transformations = []
            
            # Apply random augmentations
            if self.config.get("brightness", {}).get("enabled", True):
                aug_image = self.apply_brightness(aug_image)
            
            if self.config.get("contrast", {}).get("enabled", True):
                aug_image = self.apply_contrast(aug_image)
            
            if self.config.get("rotation", {}).get("enabled", True) and random.random() < 0.3:
                aug_image, rotation_transform = self.apply_rotation(aug_image)
                transformations.append(rotation_transform)
            
            if self.config.get("flip", {}).get("enabled", True) and random.random() < 0.5:
                aug_image, flip_transform = self.apply_flip(aug_image)
                transformations.append(flip_transform)
            
            if self.config.get("scale", {}).get("enabled", True) and random.random() < 0.4:
                aug_image, scale_transform = self.apply_scaling(aug_image)
                transformations.append(scale_transform)
            
            if self.config.get("noise", {}).get("enabled", True) and random.random() < 0.3:
                aug_image = self.apply_noise(aug_image)
            
            if self.config.get("blur", {}).get("enabled", True) and random.random() < 0.2:
                aug_image = self.apply_blur(aug_image)
            
            # Transform bounding boxes for geometric transformations
            transformed_boxes = list(bounding_boxes)  # Start with original
            for transform in transformations:
                transformed_boxes = self.transform_bounding_boxes(
                    transformed_boxes, transform, image_width, image_height
                )
            
            augmented_data.append({
                "image": aug_image,
                "bounding_boxes": transformed_boxes,
                "transformations": transformations
            })
        
        return augmented_data


def create_augmented_dataset(image_path: str, annotation_path: str, output_dir: str, 
                           augmentation_factor: int = 3, augmentation_config: Dict = None) -> List[str]:
    """
    Create augmented versions of an image with proper bounding box transformations
    
    Args:
        image_path: Path to original image
        annotation_path: Path to annotation JSON file
        output_dir: Directory to save augmented images
        augmentation_factor: Number of augmented copies to create
        augmentation_config: Augmentation configuration
        
    Returns:
        List of paths to created augmented images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original annotation
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)
    
    bounding_boxes = annotation_data.get('bounding_boxes', [])
    
    # Initialize augmentator
    augmentator = ImageAugmentator(augmentation_config)
    
    # Generate augmented versions
    augmented_versions = augmentator.augment_image_with_boxes(
        image_path, bounding_boxes, augmentation_factor
    )
    
    created_files = []
    original_filename = Path(image_path).stem
    
    for i, aug_data in enumerate(augmented_versions):
        # Save augmented image
        aug_image_path = output_dir / f"{original_filename}_aug_{i}.jpg"
        aug_data["image"].save(aug_image_path, "JPEG", quality=95)
        
        # Save augmented annotation
        aug_annotation_path = output_dir / f"{original_filename}_aug_{i}.json"
        aug_annotation = dict(annotation_data)
        aug_annotation["bounding_boxes"] = aug_data["bounding_boxes"]
        aug_annotation["augmentation_info"] = {
            "source_image": str(image_path),
            "transformations": aug_data["transformations"],
            "augmentation_index": i
        }
        
        with open(aug_annotation_path, 'w') as f:
            json.dump(aug_annotation, f, indent=2)
        
        created_files.extend([str(aug_image_path), str(aug_annotation_path)])
    
    return created_files


def augment_for_yolo_format(image_path: str, yolo_label_path: str, output_images_dir: str, 
                           output_labels_dir: str, base_filename: str, 
                           augmentation_factor: int = 3, augmentation_config: Dict = None) -> List[str]:
    """
    Create augmented images in YOLO format with transformed label coordinates
    
    Args:
        image_path: Path to original image
        yolo_label_path: Path to YOLO format label file (.txt)
        output_images_dir: Directory for augmented images
        output_labels_dir: Directory for augmented labels
        base_filename: Base filename for output files
        augmentation_factor: Number of augmented copies
        augmentation_config: Augmentation configuration
        
    Returns:
        List of created file paths
    """
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO labels
    yolo_boxes = []
    if Path(yolo_label_path).exists():
        with open(yolo_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, center_x, center_y, width, height = map(float, parts[:5])
                    # Convert from YOLO center format to corner format for transformation
                    x = center_x - width / 2
                    y = center_y - height / 2
                    yolo_boxes.append({
                        "class_id": int(class_id),
                        "x": x, "y": y, "width": width, "height": height
                    })
    
    # Initialize augmentator
    augmentator = ImageAugmentator(augmentation_config)
    
    # Generate augmented versions
    augmented_versions = augmentator.augment_image_with_boxes(
        image_path, yolo_boxes, augmentation_factor
    )
    
    created_files = []
    
    for i, aug_data in enumerate(augmented_versions):
        # Save augmented image
        aug_image_path = output_images_dir / f"{base_filename}_aug_{i}.jpg"
        aug_data["image"].save(aug_image_path, "JPEG", quality=95)
        
        # Save augmented YOLO labels
        aug_label_path = output_labels_dir / f"{base_filename}_aug_{i}.txt"
        with open(aug_label_path, 'w') as f:
            for box in aug_data["bounding_boxes"]:
                # Convert back to YOLO center format
                center_x = box['x'] + box['width'] / 2
                center_y = box['y'] + box['height'] / 2
                f.write(f"{box['class_id']} {center_x:.6f} {center_y:.6f} {box['width']:.6f} {box['height']:.6f}\n")
        
        created_files.extend([str(aug_image_path), str(aug_label_path)])
    
    return created_files


# Preset configurations for different dataset types
AUGMENTATION_PRESETS = {
    "object_detection": {
        "brightness": {"enabled": True, "range": [-15, 15]},
        "contrast": {"enabled": True, "range": [0.85, 1.15]},
        "rotation": {"enabled": True, "angles": [0, 90, 180, 270]},
        "flip": {"enabled": True, "horizontal": True, "vertical": False},
        "scale": {"enabled": True, "range": [0.95, 1.05]},
        "noise": {"enabled": True, "amount": "low"},
        "blur": {"enabled": False}  # Avoid blur for object detection
    },
    
    "image_classification": {
        "brightness": {"enabled": True, "range": [-25, 25]},
        "contrast": {"enabled": True, "range": [0.75, 1.25]},
        "rotation": {"enabled": True, "angles": [0, 90, 180, 270]},
        "flip": {"enabled": True, "horizontal": True, "vertical": True},
        "scale": {"enabled": True, "range": [0.9, 1.1]},
        "noise": {"enabled": True, "amount": "medium"},
        "blur": {"enabled": True, "amount": "low"}
    },
    
    "vision_llm": {
        "brightness": {"enabled": True, "range": [-20, 20]},
        "contrast": {"enabled": True, "range": [0.8, 1.2]},
        "rotation": {"enabled": False},  # Keep text readable
        "flip": {"enabled": False},      # Keep text orientation
        "scale": {"enabled": True, "range": [0.95, 1.05]},
        "noise": {"enabled": True, "amount": "low"},
        "blur": {"enabled": True, "amount": "low"}
    }
}