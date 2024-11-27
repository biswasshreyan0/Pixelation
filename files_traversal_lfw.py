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

# replace the given file path with the actual path
read_files_with_pathlib('sample_path')
