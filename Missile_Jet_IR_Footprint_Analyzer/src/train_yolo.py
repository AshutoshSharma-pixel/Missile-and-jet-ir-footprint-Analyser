"""Train a YOLOv8n detector on the merged IR dataset."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from utils import ensure_dir, get_device, project_paths

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--run-name", type=str, default="ir_yolov8n")
    parser.add_argument("--data", type=Path, default=None, help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Base checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, mps, cuda)")
    args = parser.parse_args()

    paths = project_paths()
    data_yaml = args.data if args.data is not None else paths["merged"] / "data.yaml"
    device = args.device if args.device else get_device()

    model = YOLO(args.weights)
    train_results = model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=device,
        cache=False,
        project=str(paths["runs"]),
        name=args.run_name,
        exist_ok=True,
        seed=args.seed,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        patience=15,
        augment=True,
        amp=False,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.5,
        translate=0.1,
        scale=0.4,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.3,
    )

    trainer = model.trainer
    best_model_path = Path(trainer.best)
    dest_best = paths["models"] / "best.pt"
    ensure_dir(dest_best.parent)
    shutil.copy2(best_model_path, dest_best)

    log_payload = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "run_dir": trainer.save_dir,
        "metrics": train_results,
    }

    log_file = ensure_dir(paths["logs"]) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2)

    print(f"Training complete. Best weights copied to {dest_best}")


if __name__ == "__main__":
    main()
