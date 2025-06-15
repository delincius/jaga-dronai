import os
from ultralytics import YOLO

# Paths
#images_dir = 'images/valUAV/JPEGImages'
#output_dir = 'images/resultsUAV'
images_dir = 'images/test'
output_dir = 'images/test-results'
#model_path = '/root/Thermal-Image-Object-Detection/runs/detect/train5/weights/best.pt'
model_path = 'YOLOv8 model/train2/weights/best.pt'
# data_path = 'images/labels/data.yaml'
# annotations_path = 'images/valUAV/Annotations/val.json'
# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load model
model = YOLO(model_path)

# labels/annotations
# metrics = model.val(data=data_path)

# List all images
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Loop through each image
for image_file in image_files:
    image_path = os.path.join(images_dir, image_file)

    # Run detection
    results = model(image_path)

    # Save results
    for result in results:
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_result.jpg")
        result.save(filename=output_path)

        # Optional: Show result window (uncomment if you want live visualization)
        result.show()

print("All detections finished. Results are saved in 'images/results/'")
