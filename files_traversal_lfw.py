from pathlib import Path

def read_files_with_pathlib(path):
    counter = 0
    for file_path in Path(path).rglob('*'):  # Use rglob to traverse subdirectories
        if file_path.is_file():
            print(file_path.name)
            counter += 1
        if counter > 2000:
            break

        # invoke pixelation code for each of these files

read_files_with_pathlib('C:\\Users\\14085\\Documents\\G6_Science_Fair_School\\lfw-deepfunneled\\lfw-deepfunneled')
# replace the given file path with the actual path