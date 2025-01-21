from pathlib import Path
from ultralytics import YOLO
from PIL import Image
# Load the YOLO model
model = YOLO("yolo11l.pt")
from PIL import Image

def pixelate_image(image_path, output_path, pixel_size):
    # Open the image
    image = Image.open(image_path)

    # Get the original dimensions
    width, height = image.size

    # Resize the image to a smaller size (smaller pixel size), then back to original size
    image = image.resize(
        (width // pixel_size, height // pixel_size),
        resample=Image.NEAREST
    )

    # Resize it back to the original dimensions
    image = image.resize((width, height), Image.NEAREST)
    return image
'''
    # Save the pixelated image
    image.save(output_path)
    print(f"Pixelated image saved as {output_path}")
'''
# Example usage
def pixelate_multiple_sizes(image_path, output_path, pixel_sizes):
    dict = {}

    for size in pixel_sizes:
        # Path to save the pixelated image
        output_image = image_path.stem + str(size) + image_path.suffix
        dict[size] = pixelate_image(image_path, output_path+output_image, size)
    dict[0] = Image.open(image_path)
    return dict

pixel_sizes = [1, 3, 5, 7, 10]  # Size of the pixels (adjust as needed)

#pixelate_multiple_sizes("/home/sbiswas/lfw-deepfunneled/Josh_Evans/Josh_Evans_0001.jpg", "/tmp/pixelation", pixel_sizes)
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
        if file_path.is_file() and file_path.suffix == '.jpg' and (not file_path.name.endswith('-result.jpg')):
            print(f"Processing file: {file_path}")
            image_dict = pixelate_multiple_sizes(file_path, "", pixel_sizes)
            for pixelation, image in image_dict.items():
                image_labels = run_yolo_image(image)
                # Append labels with image identifier to all_labels
                image_identifier = file_path.name
                if image_labels:
                    for label_info in image_labels:
                        all_labels.append(image_identifier + ", " + str(pixelation) + ", " + label_info)
                else:
                    all_labels.append(image_identifier + ", " + str(pixelation) + ", no object detected")
            counter += 1
        if counter > 2000:
            break

    # Save all labels to a single csv file
    labels_file = Path(path) / "all_labels_2.csv"
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

def run_yolo_image(image):
    results = model(image)

    # Save the result image with detections
    # save_path = image_path.parent / f"{image_path.stem}-result.jpg"
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
# datasets\CelebAMask-HQ\CelebAMask-HQ\CelebA-HQ-img
read_files_with_pathlib('/home/sbiswas/datasets/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img')