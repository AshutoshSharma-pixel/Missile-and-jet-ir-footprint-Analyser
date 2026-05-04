"""Object tracking with Norfair + YOLO detections."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Deque, Dict, List

import cv2
import numpy as np
from norfair import Detection, Tracker
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
    smooth_points,
    timestamped_filename,
)

ID_TO_CLASS = {idx: name for name, idx in CLASS_NAME_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--weights", type=Path, default=Path("../models/best.pt"))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--record", action="store_true", help="Save annotated video")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, mps, cuda)")
    return parser.parse_args()


def distance(detection: Detection, tracked_object: Detection) -> float:
    return np.linalg.norm(detection.points - tracked_object.estimate)


def main() -> None:
    args = parse_args()
    weights = args.weights if args.weights.exists() else project_paths()["models"] / "best.pt"
    model = YOLO(str(weights))
    if args.device:
        model.to(args.device)

    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Unable to open video source")

    tracker = Tracker(
        distance_function=distance,
        distance_threshold=30,
        initialization_delay=3,
    )

    fps_counter = FPSCounter()
    track_histories: Dict[int, Deque] = defaultdict(lambda: rolling_trajectory(96))

    writer = None
    if args.record:
        out_path = timestamped_filename("tracked", folder=project_paths()["videos"])
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (args.width, args.height))
        if not writer.isOpened():
            print("Failed to open video writer with avc1, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (args.width, args.height))
        
        if writer.isOpened():
            print(f"Recording to {out_path}")
        else:
            print("Error: Could not open video writer.")
            writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (args.width, args.height))
        canvas = apply_thermal_map(frame) if args.heatmap else frame.copy()

        results = model.predict(canvas, verbose=False)
        detections: List[Detection] = []
        for box in results[0].boxes:
            conf = float(box.conf)
            if conf < args.confidence:
                continue
            cls_id = int(box.cls)
            label = ID_TO_CLASS.get(cls_id, "obj")
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            centroid = np.array([[ (x1 + x2) / 2.0, (y1 + y2) / 2.0 ]])
            detections.append(Detection(points=centroid, scores=np.array([conf]), data={"bbox": (x1, y1, x2, y2), "cls": label}))

        tracked_objects = tracker.update(detections=detections)
        for tracked in tracked_objects:
            det_data = tracked.last_detection.data if tracked.last_detection else {}
            bbox = det_data.get("bbox", (0, 0, 0, 0))
            x1, y1, x2, y2 = map(int, bbox)
            cls_name = det_data.get("cls", "obj")
            tid = tracked.id
            color = (0, 255, 0) if cls_name != "missile" else (0, 128, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{cls_name} | ID {tid}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if cls_name == "missile":
                highlight_missile_plume(canvas, (x1, y1, x2, y2))

            centroid = tracked.estimate[0]
            track_histories[tid].append((int(centroid[0]), int(centroid[1])))
            smoothed = smooth_points(track_histories[tid], window=5)
            for i in range(1, len(smoothed)):
                thickness = max(1, 3 - (len(smoothed) - i) // 10)
                cv2.line(canvas, smoothed[i - 1], smoothed[i], (255, 255, 0), thickness)

        fps = fps_counter.update()
        draw_fps(canvas, fps)
        cv2.imshow("Tracked IR Targets", canvas)
        if writer is not None:
            writer.write(canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
