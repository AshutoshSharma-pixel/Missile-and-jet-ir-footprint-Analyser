"""Shared utility helpers for the Missile & Jet IR Footprint Analyzer."""
from __future__ import annotations

import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from lxml import etree

# Unified class mapping expected by the project.
CLASS_NAME_TO_ID = {
    "missile": 0,
    "jet": 1,
    "aircraft": 2,
    "helicopter": 3,
    "heat_signature": 4,
}

# Known dataset folders provided by the user outside the repo root.
DEFAULT_EXTERNAL_DATASETS = [
    "Ir System dataset",
    "Miliitary aircraft detection dataset",
    "Normal Vehicles Dataset",
]

# Dataset-specific aliases that collapse into the five core classes.
CLASS_ALIASES = {
    "fighter": "jet",
    "bomber": "aircraft",
    "plane": "aircraft",
    "airliner": "aircraft",
    "uav": "aircraft",
    "drone": "aircraft",
    "helicopter": "helicopter",
    "chopper": "helicopter",
    "rotor": "helicopter",
    "rocket": "missile",
    "projectile": "missile",
    "heat": "heat_signature",
    "thermal": "heat_signature",
    "vehicle": "heat_signature",
}


def normalize_class_name(raw_name: str) -> str:
    name = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
    if name in CLASS_NAME_TO_ID:
        return name
    return CLASS_ALIASES.get(name, "heat_signature")


def xml_to_yolo(xml_path: Path, class_map: Dict[str, int]) -> List[Tuple[int, float, float, float, float]]:
    """Convert a Pascal VOC XML file into YOLO format rows."""
    xml_tree = etree.parse(str(xml_path))
    root = xml_tree.getroot()

    size = root.find("size")
    width = float(size.findtext("width", default="1"))
    height = float(size.findtext("height", default="1"))

    rows: List[Tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", default="heat_signature")
        class_name = normalize_class_name(raw_name)
        class_id = class_map.get(class_name, class_map["heat_signature"])

        bbox = obj.find("bndbox")
        xmin, ymin = float(bbox.findtext("xmin", default="0")), float(bbox.findtext("ymin", default="0"))
        xmax, ymax = float(bbox.findtext("xmax", default="0")), float(bbox.findtext("ymax", default="0"))

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_w = (xmax - xmin) / width
        box_h = (ymax - ymin) / height
        rows.append((class_id, x_center, y_center, box_w, box_h))
    return rows


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_paths(base_dir: Path | None = None) -> Dict[str, Path]:
    base = Path(base_dir or Path(__file__).resolve().parents[1])
    datasets_dir = base / "datasets"
    external_dataset_dir = base.parent / "Dataset"

    def has_expected_folders(root: Path) -> bool:
        return all((root / name).exists() for name in DEFAULT_EXTERNAL_DATASETS)

    if has_expected_folders(external_dataset_dir):
        datasets_dir = external_dataset_dir
    elif not any(datasets_dir.iterdir()) and external_dataset_dir.exists():
        datasets_dir = external_dataset_dir

    return {
        "root": base,
        "datasets": datasets_dir,
        "merged": base / "merged_yolo_dataset",
        "models": base / "models",
        "outputs": base / "outputs",
        "logs": base / "outputs" / "logs",
        "runs": base / "outputs" / "runs",
        "videos": base / "outputs" / "recorded_videos",
    }


def get_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def timestamped_filename(prefix: str, suffix: str = ".mp4", folder: Path | None = None) -> Path:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = ensure_dir(folder or Path.cwd())
    return folder / f"{prefix}_{now}{suffix}"


def apply_thermal_map(frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
    return cv2.addWeighted(heatmap, alpha, frame, 1 - alpha, 0)


def highlight_missile_plume(frame: np.ndarray, bbox: Sequence[int]) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    plume_region = frame[y1:y2, x1:x2]
    if plume_region.size == 0:
        return
    overlay = plume_region.copy()
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + 80, 0, 255)
    cv2.addWeighted(overlay, 0.5, plume_region, 0.5, 0, dst=plume_region)


class FPSCounter:
    def __init__(self, averaging_window: int = 30) -> None:
        self.timestamps: Deque[float] = deque(maxlen=averaging_window)

    def update(self) -> float:
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0
        fps = len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])
        return fps


def draw_fps(frame: np.ndarray, fps: float, origin: Tuple[int, int] = (20, 30)) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def compute_centroid(bbox: Sequence[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)


def smooth_points(points: Iterable[Tuple[int, int]], window: int = 5) -> List[Tuple[int, int]]:
    pts = list(points)
    if len(pts) <= 2:
        return pts
    smoothed: List[Tuple[int, int]] = []
    for i in range(len(pts)):
        start = max(0, i - window + 1)
        chunk = pts[start : i + 1]
        xs = sum(p[0] for p in chunk) / len(chunk)
        ys = sum(p[1] for p in chunk) / len(chunk)
        smoothed.append((int(xs), int(ys)))
    return smoothed


def rolling_trajectory(maxlen: int = 32) -> Deque[Tuple[int, int]]:
    return deque(maxlen=maxlen)
