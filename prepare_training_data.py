#!/usr/bin/env python3
"""
Prepare training dataset for fine-tuning vision models on TV/STB screens
"""
import json
import os
import sys
from datetime import datetime
import shutil
from typing import Dict, List, Any
import hashlib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import log
from PIL import Image
import pandas as pd

# TV/STB specific screen states
SCREEN_STATES = {
    "home": "Main home screen with app rail",
    "app_rail": "App selection rail/ribbon",
    "app_loading": "App is loading/launching", 
    "app_content": "App content displayed (video, menu, etc)",
    "login": "Login/authentication screen",
    "profile_selection": "User profile selection",
    "settings": "Settings menu",
    "guide": "TV guide/EPG",
    "error": "Error message displayed",
    "no_signal": "No signal screen",
    "black_screen": "Black/blank screen",
    "buffering": "Video buffering indicator",
    "video_playing": "Video content playing",
    "advertisement": "Ad playing",
    "search": "Search interface",
    "details": "Content details/info page"
}

# Common apps to recognize
COMMON_APPS = [
    "Netflix", "YouTube", "Prime Video", "Disney+", "Hulu",
    "HBO Max", "Apple TV+", "Peacock", "Paramount+", "ESPN",
    "Sling TV", "Spotify", "Pandora", "Settings", "Store"
]

class TrainingDataPreparer:
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.annotations_file = os.path.join(data_dir, "annotations.json")
        self.dataset_file = os.path.join(data_dir, "dataset.json")
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Load existing annotations if available
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> Dict:
        """Load existing annotations"""
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r') as f:
                return json.load(f)
        return {"images": {}, "stats": {}}
    
    def _save_annotations(self):
        """Save annotations"""
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
    
    def _get_image_hash(self, image_path: str) -> str:
        """Get hash of image file"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def add_labeled_image(self, image_path: str, labels: Dict[str, Any]):
        """Add a labeled image to the dataset"""
        if not os.path.exists(image_path):
            log.error(f"Image not found: {image_path}")
            return False
        
        # Copy image to training directory
        image_hash = self._get_image_hash(image_path)
        ext = os.path.splitext(image_path)[1]
        new_filename = f"{image_hash}{ext}"
        new_path = os.path.join(self.images_dir, new_filename)
        
        if not os.path.exists(new_path):
            shutil.copy2(image_path, new_path)
        
        # Add annotation
        self.annotations["images"][new_filename] = {
            "original_path": image_path,
            "hash": image_hash,
            "labels": labels,
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_annotations()
        log.info(f"✓ Added image: {new_filename}")
        return True
    
    def interactive_labeling(self, image_path: str):
        """Interactive labeling interface"""
        log.info(f"\n{'='*60}")
        log.info(f"Labeling: {os.path.basename(image_path)}")
        log.info(f"{'='*60}")
        
        # Display image info
        try:
            img = Image.open(image_path)
            log.info(f"Image size: {img.size}")
            img.show()  # Opens in default viewer
        except Exception as e:
            log.error(f"Cannot open image: {e}")
            return
        
        labels = {}
        
        # Screen type
        log.info("\nScreen types:")
        for i, (state, desc) in enumerate(SCREEN_STATES.items(), 1):
            print(f"{i}. {state} - {desc}")
        
        choice = input("\nSelect screen type (number): ").strip()
        try:
            screen_type = list(SCREEN_STATES.keys())[int(choice)-1]
            labels["screen_type"] = screen_type
        except:
            labels["screen_type"] = "unknown"
        
        # App name
        log.info("\nCommon apps:")
        for i, app in enumerate(COMMON_APPS, 1):
            print(f"{i}. {app}", end="  ")
            if i % 5 == 0:
                print()
        
        app_input = input("\n\nApp name (number or custom name, blank if none): ").strip()
        if app_input:
            try:
                labels["app_name"] = COMMON_APPS[int(app_input)-1]
            except:
                labels["app_name"] = app_input
        else:
            labels["app_name"] = None
        
        # UI elements
        labels["ui_elements"] = []
        log.info("\nVisible UI elements (comma-separated):")
        print("Options: menu, button, video_player, rail, list, keyboard, dialog, spinner")
        elements = input("Elements: ").strip()
        if elements:
            labels["ui_elements"] = [e.strip() for e in elements.split(",")]
        
        # Text content
        text = input("\nMain visible text (blank to skip): ").strip()
        if text:
            labels["visible_text"] = text
        
        # Anomalies
        labels["anomalies"] = []
        log.info("\nAny anomalies? (y/n for each)")
        for anomaly in ["black_screen", "frozen", "buffering", "error", "artifacts"]:
            if input(f"  {anomaly}? (y/n): ").lower() == 'y':
                labels["anomalies"].append(anomaly)
        
        # Navigation state
        labels["navigation"] = {}
        log.info("\nNavigation hints:")
        labels["navigation"]["focused_element"] = input("Focused element description: ").strip()
        labels["navigation"]["can_navigate"] = {
            "up": input("Can navigate up? (y/n): ").lower() == 'y',
            "down": input("Can navigate down? (y/n): ").lower() == 'y',
            "left": input("Can navigate left? (y/n): ").lower() == 'y',
            "right": input("Can navigate right? (y/n): ").lower() == 'y',
        }
        
        # Additional notes
        notes = input("\nAdditional notes (optional): ").strip()
        if notes:
            labels["notes"] = notes
        
        # Save
        self.add_labeled_image(image_path, labels)
        log.info("✓ Image labeled and saved!")
    
    def generate_training_dataset(self, output_format: str = "llava"):
        """Generate training dataset in specific format"""
        if not self.annotations["images"]:
            log.error("No labeled images found!")
            return
        
        dataset = []
        
        for image_file, data in self.annotations["images"].items():
            labels = data["labels"]
            
            if output_format == "llava":
                # LLaVA fine-tuning format
                entry = {
                    "id": data["hash"][:8],
                    "image": image_file,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nWhat do you see on this TV screen?"
                        },
                        {
                            "from": "gpt",
                            "value": self._generate_description(labels)
                        },
                        {
                            "from": "human", 
                            "value": "What is the current screen state?"
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps({
                                "screen_type": labels.get("screen_type", "unknown"),
                                "app_name": labels.get("app_name"),
                                "ui_elements": labels.get("ui_elements", []),
                                "anomalies": labels.get("anomalies", [])
                            })
                        }
                    ]
                }
            
            elif output_format == "simple":
                # Simple prompt-completion format
                entry = {
                    "image": image_file,
                    "prompt": "Analyze this TV screen image",
                    "completion": json.dumps(labels)
                }
            
            dataset.append(entry)
        
        # Save dataset
        with open(self.dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Generate statistics
        self._generate_stats()
        
        log.info(f"\n✓ Generated {output_format} dataset with {len(dataset)} examples")
        log.info(f"  Saved to: {self.dataset_file}")
    
    def _generate_description(self, labels: Dict) -> str:
        """Generate natural language description from labels"""
        desc_parts = []
        
        # Screen type
        screen = labels.get("screen_type", "unknown")
        if screen in SCREEN_STATES:
            desc_parts.append(f"This is a {SCREEN_STATES[screen].lower()}")
        
        # App
        if labels.get("app_name"):
            desc_parts.append(f"The {labels['app_name']} app is visible")
        
        # UI elements
        if labels.get("ui_elements"):
            elements = ", ".join(labels["ui_elements"])
            desc_parts.append(f"I can see these UI elements: {elements}")
        
        # Visible text
        if labels.get("visible_text"):
            desc_parts.append(f"The screen shows: '{labels['visible_text']}'")
        
        # Anomalies
        if labels.get("anomalies"):
            desc_parts.append(f"Issues detected: {', '.join(labels['anomalies'])}")
        
        # Navigation
        if labels.get("navigation", {}).get("focused_element"):
            desc_parts.append(f"The focus is on: {labels['navigation']['focused_element']}")
        
        return ". ".join(desc_parts) + "."
    
    def _generate_stats(self):
        """Generate dataset statistics"""
        stats = {
            "total_images": len(self.annotations["images"]),
            "screen_types": {},
            "apps": {},
            "anomalies": {},
            "ui_elements": {}
        }
        
        for data in self.annotations["images"].values():
            labels = data["labels"]
            
            # Count screen types
            screen = labels.get("screen_type", "unknown")
            stats["screen_types"][screen] = stats["screen_types"].get(screen, 0) + 1
            
            # Count apps
            app = labels.get("app_name", "none")
            if app:
                stats["apps"][app] = stats["apps"].get(app, 0) + 1
            
            # Count anomalies
            for anomaly in labels.get("anomalies", []):
                stats["anomalies"][anomaly] = stats["anomalies"].get(anomaly, 0) + 1
            
            # Count UI elements
            for element in labels.get("ui_elements", []):
                stats["ui_elements"][element] = stats["ui_elements"].get(element, 0) + 1
        
        self.annotations["stats"] = stats
        self._save_annotations()
        
        # Display stats
        log.info("\n📊 Dataset Statistics:")
        log.info(f"Total images: {stats['total_images']}")
        log.info(f"Screen types: {stats['screen_types']}")
        log.info(f"Apps: {stats['apps']}")
        log.info(f"UI elements: {stats['ui_elements']}")
        log.info(f"Anomalies: {stats['anomalies']}")

def main():
    """Main function"""
    preparer = TrainingDataPreparer()
    
    while True:
        print("\n" + "="*60)
        print("TV/STB Screen Dataset Preparation")
        print("="*60)
        print("1. Label images interactively")
        print("2. Batch import with CSV")
        print("3. Generate training dataset")
        print("4. View statistics")
        print("5. Export for fine-tuning")
        print("6. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            # Interactive labeling
            image_path = input("Enter image path (or 'done' to finish): ").strip()
            while image_path.lower() != 'done':
                if os.path.exists(image_path):
                    preparer.interactive_labeling(image_path)
                else:
                    log.error(f"Image not found: {image_path}")
                
                image_path = input("\nNext image path (or 'done' to finish): ").strip()
        
        elif choice == "3":
            # Generate dataset
            format_choice = input("Format (1=llava, 2=simple): ").strip()
            format_type = "llava" if format_choice == "1" else "simple"
            preparer.generate_training_dataset(format_type)
        
        elif choice == "4":
            # View stats
            preparer._generate_stats()
        
        elif choice == "5":
            # Export instructions
            log.info("\n📦 Fine-tuning Instructions:")
            log.info("1. Dataset is in: training_data/dataset.json")
            log.info("2. Images are in: training_data/images/")
            log.info("3. Use LLaVA fine-tuning scripts:")
            log.info("   https://github.com/haotian-liu/LLaVA#fine-tuning")
            log.info("4. Or use Axolotl for easier fine-tuning:")
            log.info("   https://github.com/OpenAccess-AI-Collective/axolotl")
        
        elif choice == "6":
            break

if __name__ == "__main__":
    main()