"""Real-time detection with optional thermal visualization."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Deque, Dict

import cv2
import numpy as np
from ultralytics import YOLO

from utils import (
    CLASS_NAME_TO_ID,
    FPSCounter,
    apply_thermal_map,
    compute_centroid,
    draw_fps,
    highlight_missile_plume,
    project_paths,
    rolling_trajectory,
)

ID_TO_CLASS = {idx: name for name, idx in CLASS_NAME_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="0 for webcam or path to media")
    parser.add_argument("--weights", type=Path, default=Path("../models/best.pt"))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--heatmap", action="store_true", help="Apply thermal colormap overlay")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, mps, cuda)")
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def main() -> None:
    args = parse_args()
    weights = args.weights if args.weights.exists() else project_paths()["models"] / "best.pt"
    model = YOLO(str(weights))
    if args.device:
        model.to(args.device)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit("Unable to open source")

    fps_counter = FPSCounter()
    trajectories: Dict[int, Deque] = defaultdict(lambda: rolling_trajectory(48))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (args.width, args.height))
        vis_frame = apply_thermal_map(frame) if args.heatmap else frame.copy()

        results = model.predict(vis_frame, verbose=False)
        current = results[0]
        boxes = current.boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf)
                if conf < args.confidence:
                    continue
                cls_id = int(box.cls)
                label = ID_TO_CLASS.get(cls_id, "obj")
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                color = (0, 255, 255) if label == "missile" else (0, 200, 255)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, f"{label}:{conf:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if label == "missile":
                    highlight_missile_plume(vis_frame, (x1, y1, x2, y2))

                centroid = compute_centroid((x1, y1, x2, y2))
                trajectories[cls_id].append(centroid)
                pts = list(trajectories[cls_id])
                for i in range(1, len(pts)):
                    cv2.line(vis_frame, pts[i - 1], pts[i], (255, 255, 0), 2)

        fps = fps_counter.update()
        draw_fps(vis_frame, fps)
        cv2.imshow("IR Footprint Detection", vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
