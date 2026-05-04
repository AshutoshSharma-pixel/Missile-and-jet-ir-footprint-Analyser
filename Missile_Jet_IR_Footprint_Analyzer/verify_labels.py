import glob
from pathlib import Path
from tqdm import tqdm

def check_labels(label_dir, num_classes=5):
    print(f"Checking labels in {label_dir}...")
    files = list(Path(label_dir).rglob("*.txt"))
    invalid_files = []
    
    for file_path in tqdm(files):
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= num_classes:
                        print(f"Invalid class {class_id} in {file_path} at line {i+1}")
                        invalid_files.append(file_path)
                        break # Stop checking this file
                except ValueError:
                    print(f"Malformed line in {file_path} at line {i+1}: {line.strip()}")
                    invalid_files.append(file_path)
                    break

    if invalid_files:
        print(f"Found {len(invalid_files)} files with invalid labels.")
    else:
        print("All labels appear valid.")

if __name__ == "__main__":
    check_labels("merged_yolo_dataset/labels", num_classes=5)
