from pathlib import Path
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

def setup():
    # Load a model
    # Train the model
    train_results = model.train(
        data="coco8.yaml",  # path to dataset YAML
        epochs=100,         # number of training epochs
        imgsz=640,          # training image size
        device="cpu",       # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

def read_files_with_pathlib(path):
    counter = 0
    all_labels = []  # List to store labels from all images

    for file_path in Path(path).rglob('*'):
        if file_path.is_file() and file_path.suffix == '.jpg' and file_path.suffix != '-result.jpg':
            print(f"Processing file: {file_path}")
            image_labels = run_yolo(file_path)
            # Append labels with image identifier to all_labels
            if image_labels:
                image_identifier = file_path.name
                for label_info in image_labels:
                    all_labels.append(f"{image_identifier}: {label_info}")
            counter += 1
        if counter > 50:
            break

    # Save all labels to a single text file
    labels_file = Path(path) / "all_labels.txt"
    labels_text = '\n'.join(all_labels)
    with open(labels_file, 'w') as f:
        f.write(labels_text)

    # Optionally, print the collected labels
    print("All Detections:")
    print(labels_text)

def run_yolo(image_path):
    results = model(image_path)

    # Save the result image with detections
    save_path = image_path.parent / f"{image_path.stem}-result.jpg"
    # results[0].save(save_path)

    # Extract labels from results
    detections = results[0].boxes  # Get Boxes object
    labels = []
    for box in detections:
        cls_id = int(box.cls[0])        # Class ID
        label = model.names[cls_id]     # Class label
        confidence = float(box.conf[0]) # Confidence score
        if (label == "person"):
            labels.append(f"Label: {label}, Confidence: {confidence:.2f}")

    return labels  # Return the labels for this image

# Uncomment the following line if you want to train the model
# setup()

# Replace 'lfw-deepfunneled/' with your actual path
read_files_with_pathlib('lfw-deepfunneled/')
