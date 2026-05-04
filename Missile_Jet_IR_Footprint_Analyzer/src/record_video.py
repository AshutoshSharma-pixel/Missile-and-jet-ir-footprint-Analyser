"""Detect and save annotated IR video feeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import (
    FPSCounter,
    apply_thermal_map,
    compute_centroid,
    draw_fps,
    highlight_missile_plume,
    project_paths,
    rolling_trajectory,
    timestamped_filename,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--weights", type=Path, default=Path("../models/best.pt"))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, mps, cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights if args.weights.exists() else project_paths()["models"] / "best.pt"))
    if args.device:
        model.to(args.device)

    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Unable to open source")

    videos_dir = project_paths()["videos"]
    out_path = timestamped_filename("recording", folder=videos_dir)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (args.width, args.height))
    if not writer.isOpened():
        print("Failed to open video writer with avc1, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (args.width, args.height))
    
    if not writer.isOpened():
        raise SystemExit("Error: Could not open video writer.")
        
    print(f"Recording annotated stream to {out_path}")

    fps_counter = FPSCounter()
    trails = rolling_trajectory(64)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (args.width, args.height))
        canvas = apply_thermal_map(frame) if args.heatmap else frame.copy()

        results = model.predict(canvas, verbose=False)
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf)
                if conf < args.confidence:
                    continue
                cls_id = int(box.cls)
                label = model.model.names.get(cls_id, "obj")
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                color = (0, 255, 0)
                if label == "missile":
                    color = (0, 165, 255)
                    highlight_missile_plume(canvas, (x1, y1, x2, y2))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"{label}:{conf:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                trails.append(compute_centroid((x1, y1, x2, y2)))
        pts = list(trails)
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], (255, 255, 0), 2)

        fps = fps_counter.update()
        draw_fps(canvas, fps)
        writer.write(canvas)
        cv2.imshow("Recording", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Recording stopped.")


if __name__ == "__main__":
    main()
