from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo11n.pt")


def setup():
    # Load a model

    # Train the model
    train_results = model.train(
        data="coco8.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

def read_files_with_pathlib(path):
    counter = 0
    for file_path in Path(path).rglob('*'):  # Use rglob to traverse subdirectories
        if file_path.is_file():
            print(file_path)
            print(str(file_path.parent) + "/" + str(file_path.stem) + "-result.jpg")
            run_yolo(file_path)
            counter += 1
        if counter > 5:
            break

        # invoke pixelation code for each of these files

def run_yolo(name):
    results = model(name)
    results[0].save(filename= str(name.parent) + "/" + str(name.stem) + "-result.jpg")

setup()
read_files_with_pathlib('lfw-deepfunneled/')
# replace the given file path with the actual path