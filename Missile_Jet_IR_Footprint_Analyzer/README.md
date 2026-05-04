# Missile & Jet IR Footprint Analyzer

A defense-grade computer vision system designed to detect and analyze heat/IR signatures of missiles, jets, aircraft, and thermal objects using YOLOv8.

## Project Goal
Create a robust object detection and tracking system that:
- Detects missiles, jets, aircraft, helicopters, and thermal heat signatures.
- Works on thermal/IR-style datasets.
- Merges multiple heterogeneous datasets into a unified YOLO format.
- Trains efficiently on Mac M-series chips (MPS backend).
- Performs real-time detection and tracking on video feeds.

## Features
- **Data Merging**: Automatically ingests and normalizes multiple datasets (IR System, Military Aircraft, Normal Vehicles).
- **YOLOv8 Training**: Fine-tuned on thermal data with specific augmentations.
- **Real-time Detection**: Inference on webcam, video files, or images with thermal colormap overlays.
- **Object Tracking**: Integrated tracking (Norfair) with trajectory visualization.
- **Video Recording**: Save annotated detection streams.
- **Missile Plume Highlighting**: Special visualization for high-intensity heat sources.

## Class Definitions
The system normalizes all inputs into 5 core classes:
0. `missile`
1. `jet`
2. `aircraft`
3. `helicopter`
4. `heat_signature`

## Setup

### Prerequisites
- Python 3.9+
- macOS (for MPS support) or CUDA-enabled GPU

### Installation
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Dataset
Merge the raw datasets into a YOLO-ready format. This script will also generate the `data.yaml` configuration file.
```bash
python src/merge_and_convert.py
```
*Note: Ensure your raw datasets are located in the `datasets/` folder or the external `Dataset/` directory as configured.*

### 2. Train Model
Train the YOLOv8n model. The script automatically detects MPS (Mac) or CUDA.
```bash
python src/train_yolo.py --epochs 50 --batch 8
```
- **Output**: Best model saved to `models/best.pt`.
- **Logs**: Training metrics saved to `outputs/logs/`.

### 3. Run Detection
Run inference on a video file or webcam.
```bash
# Webcam
python src/detect.py --source 0 --heatmap

# Video File
python src/detect.py --source path/to/video.mp4 --heatmap
```
- `--heatmap`: Applies a thermal colormap to the input video.

### 4. Object Tracking
Track objects and visualize their trajectories.
```bash
python src/track.py --source 0 --record
```
- `--record`: Saves the output to `outputs/recorded_videos/`.

### 5. Record Video
Record a detection session without tracking overlays.
```bash
python src/record_video.py --source 0
```

## Project Structure
```
Missile_Jet_IR_Footprint_Analyzer/
├── datasets/               # Raw datasets
├── merged_yolo_dataset/    # Processed YOLO dataset (images/labels)
├── models/                 # Trained models (best.pt)
├── outputs/                # Logs, runs, and recorded videos
├── src/
│   ├── merge_and_convert.py
│   ├── train_yolo.py
│   ├── detect.py
│   ├── track.py
│   ├── record_video.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Troubleshooting
- **ModuleNotFoundError**: Ensure you are running commands from the root directory and the virtual environment is active.
- **MPS Warning**: "Pin memory" warnings on Mac are normal and can be ignored.
