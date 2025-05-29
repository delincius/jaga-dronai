import cv2
import numpy as np
import os
import time
from pathlib import Path
import threading
from datetime import datetime
import json
import argparse
import signal
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RedObjectDetector:
    def __init__(self, folder_path, file_pattern="*.jpg", check_interval=1.0):
        """
        Initialize the red object detector
        
        Args:
            folder_path: Path to monitor for images
            file_pattern: File pattern to match (e.g., "*.jpg", "image_*.png")
            check_interval: How often to check for new images (seconds)
        """
        self.folder_path = Path(folder_path)
        self.file_pattern = file_pattern
        self.check_interval = check_interval
        self.processed_files = set()
        self.running = False
        
        # Red color ranges in HSV
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Minimum area for object detection
        self.min_area = 100
        
        # Detection results file for GUI communication
        self.detection_results_file = "detection_results.json"
        
    def analyze_red_content(self, image_path):
        """
        Analyze red content in image and return detection confidence
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (has_red_objects, confidence_percentage, object_count)
        """
        try:
            # Read the image
            image = cv2.imread(str(image_path))
            if image is None:
                return False, 0.0, 0
            
            # Convert BGR to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create masks for red color
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = mask1 + mask2
            
            # Remove noise
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Calculate red pixel percentage
            total_pixels = image.shape[0] * image.shape[1]
            red_pixels = cv2.countNonZero(mask)
            red_percentage = (red_pixels / total_pixels) * 100
            
            # Find contours for object counting
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count significant red objects
            object_count = 0
            for contour in contours:
                if cv2.contourArea(contour) > self.min_area:
                    object_count += 1
            
            # Determine if red objects are present
            has_objects = object_count > 0
            
            # Calculate confidence based on red content and object presence
            if has_objects:
                # Confidence is higher with more red content and more objects
                confidence = min(95.0, red_percentage * 10 + object_count * 5)
            else:
                # Low confidence if no distinct objects, even if some red pixels exist
                confidence = min(30.0, red_percentage * 2)
            
            return has_objects, round(confidence, 1), object_count
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False, 0.0, 0
    
    def write_detection_result(self, image_path, has_red, confidence, count, drone_number=None):
        """Write detection result to JSON file for GUI communication"""
        try:
            result = {
                "timestamp": datetime.now().isoformat(),
                "image_path": str(image_path),
                "filename": image_path.name,
                "has_red_objects": has_red,
                "confidence": confidence,
                "object_count": count,
                "drone_number": drone_number,
                "status": "🔴 RED DETECTED" if has_red else "⚪ NO RED OBJECTS"
            }
            
            # Read existing results
            existing_results = []
            if os.path.exists(self.detection_results_file):
                try:
                    with open(self.detection_results_file, 'r') as f:
                        existing_results = json.load(f)
                except:
                    existing_results = []
            
            # Add new result
            existing_results.append(result)
            
            # Keep only last 100 results to prevent file from growing too large
            if len(existing_results) > 100:
                existing_results = existing_results[-100:]
            
            # Write back to file
            with open(self.detection_results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
                
        except Exception as e:
            print(f"Error writing detection result: {e}")
    
    def process_single_image(self, image_path, drone_number=None):
        """Process a single image and report results"""
        has_red, confidence, count = self.analyze_red_content(image_path)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        filename = image_path.name
        
        if has_red:
            status = "🔴 RED DETECTED"
            details = f"({count} object{'s' if count != 1 else ''})"
        else:
            status = "⚪ NO RED OBJECTS"
            details = ""
        
        print(f"[{timestamp}] {filename:<25} | {status:<20} | {confidence:>5.1f}% confidence {details}")
        
        # Write result to file for GUI communication
        self.write_detection_result(image_path, has_red, confidence, count, drone_number)
        
        return has_red, confidence, count
    
    def scan_folder_once(self):
        """Scan the folder once for images matching the pattern"""
        if not self.folder_path.exists():
            print(f"❌ Error: Folder {self.folder_path} does not exist!")
            return
        
        image_files = list(self.folder_path.glob(self.file_pattern))
        
        if not image_files:
            print(f"❌ No files matching pattern '{self.file_pattern}' found in {self.folder_path}")
            return
        
        print(f"\n🔍 Scanning {len(image_files)} image(s) in {self.folder_path}")
        print(f"📂 Pattern: {self.file_pattern}")
        print("-" * 80)
        print(f"{'TIME':<10} {'FILENAME':<25} | {'STATUS':<20} | {'CONFIDENCE'}")
        print("-" * 80)
        
        total_with_red = 0
        total_objects = 0
        
        for image_path in sorted(image_files):
            has_red, confidence, count = self.process_single_image(image_path)
            if has_red:
                total_with_red += 1
                total_objects += count
        
        print("-" * 80)
        print(f"📊 SUMMARY:")
        print(f"   • Images with red objects: {total_with_red}/{len(image_files)}")
        print(f"   • Total red objects found: {total_objects}")
        if len(image_files) > 0:
            print(f"   • Detection rate: {(total_with_red/len(image_files)*100):.1f}%")

class ImageFileHandler(FileSystemEventHandler):
    """Handle new image files being created"""
    
    def __init__(self, detector, file_pattern="*.jpg"):
        self.detector = detector
        self.file_pattern = file_pattern
        
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Check if it matches our pattern
        if file_path.match(self.file_pattern):
            # Wait a bit to ensure file is fully written
            time.sleep(0.5)
            
            # Extract drone number from path if possible
            drone_number = None
            if "drone_" in str(file_path):
                try:
                    # Extract drone number from path like "drone_1/image.jpg"
                    parts = str(file_path).split("drone_")
                    if len(parts) > 1:
                        drone_number = int(parts[1].split("/")[0])
                except:
                    pass
            
            print(f"📸 New image detected: {file_path.name}")
            self.detector.process_single_image(file_path, drone_number)

def monitor_multiple_drone_folders(base_folder, file_pattern="*.jpg"):
    """Monitor multiple drone folders for new images"""
    print(f"🔍 Starting real-time monitoring for multiple drone folders in: {base_folder}")
    print(f"📂 Pattern: {file_pattern}")
    print("💡 Press Ctrl+C to stop")
    print("-" * 80)
    print(f"{'TIME':<10} {'FILENAME':<25} | {'STATUS':<20} | {'CONFIDENCE'}")
    print("-" * 80)
    
    # Create detector instance
    detector = RedObjectDetector(base_folder, file_pattern)
    
    # Set up file system watcher
    event_handler = ImageFileHandler(detector, file_pattern)
    observer = Observer()
    observer.schedule(event_handler, str(base_folder), recursive=True)
    
    # Clear previous detection results
    if os.path.exists(detector.detection_results_file):
        os.remove(detector.detection_results_file)
    
    observer.start()
    detector.running = True
    
    try:
        while detector.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")
    finally:
        observer.stop()
        observer.join()
        detector.running = False

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle termination signals gracefully"""
    global running
    print(f"\n[Signal {sig}] Shutting down detection service...")
    running = False

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='Red object detection service')
    parser.add_argument('--folder', '-f', type=str, required=True,
                       help='Base folder to monitor for images')
    parser.add_argument('--pattern', '-p', type=str, default='*.jpg',
                       help='File pattern to match (default: *.jpg)')
    parser.add_argument('--mode', '-m', type=str, default='monitor',
                       choices=['scan', 'monitor'],
                       help='Mode: scan (once) or monitor (continuous)')
    
    args = parser.parse_args()
    
    print(f"🔍 Red Object Detection Service")
    print(f"📂 Monitoring folder: {args.folder}")
    print(f"📄 File pattern: {args.pattern}")
    print(f"🔄 Mode: {args.mode}")
    
    try:
        if args.mode == 'scan':
            # One-time scan
            detector = RedObjectDetector(args.folder, args.pattern)
            detector.scan_folder_once()
        else:
            # Continuous monitoring
            monitor_multiple_drone_folders(args.folder, args.pattern)
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    
    print("🔍 Detection service stopped")