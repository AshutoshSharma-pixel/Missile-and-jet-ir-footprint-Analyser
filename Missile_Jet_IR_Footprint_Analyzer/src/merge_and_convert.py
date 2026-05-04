"""Merge heterogeneous thermal datasets into a single YOLO dataset."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import (
    CLASS_NAME_TO_ID,
    ensure_dir,
    normalize_class_name,
    project_paths,
    xml_to_yolo,
)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_BOXES_PER_IMAGE = 400


def load_mapping_config(config_path: Path | None) -> Dict[str, Dict[str, str]]:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, Dict[str, str]] = {}
    for dataset_name, dataset_cfg in data.items():
        sub_map = {}
        for key, value in dataset_cfg.get("map", {}).items():
            sub_map[str(key)] = normalize_class_name(value)
        mapping[dataset_name] = sub_map
    return mapping


def parse_yolo_label_file(label_path: Path, dataset_map: Dict[str, str]) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return rows
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_token = parts[0]
            class_name = dataset_map.get(class_token)
            if class_name is None and class_token.isdigit():
                alias = dataset_map.get(str(class_token))
                if alias is not None:
                    class_name = alias
            if class_name is None:
                # If token is already a known class name.
                class_name = normalize_class_name(class_token)
            class_id = CLASS_NAME_TO_ID.get(class_name, CLASS_NAME_TO_ID["heat_signature"])
            coords = tuple(float(x) for x in parts[1:])
            rows.append((class_id, *coords))
    return rows


def collect_voc_annotations(xml_files: Iterable[Path], dataset_map: Dict[str, str]) -> Dict[Path, List[Tuple[int, float, float, float, float]]]:
    annotations = {}
    for xml_path in xml_files:
        rows = xml_to_yolo(xml_path, {k: CLASS_NAME_TO_ID.get(normalize_class_name(v), CLASS_NAME_TO_ID["heat_signature"]) for k, v in CLASS_NAME_TO_ID.items()})
        annotations[xml_path.with_suffix("")] = rows
    return annotations


def discover_image_files(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: float, height: float) -> Tuple[float, float, float, float]:
    width = max(width, 1e-6)
    height = max(height, 1e-6)
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_w = abs(x2 - x1) / width
    box_h = abs(y2 - y1) / height
    clamp = lambda v: max(0.0, min(1.0, v))
    return clamp(x_center), clamp(y_center), clamp(box_w), clamp(box_h)


def truncate_boxes(rows: List[Tuple[int, float, float, float, float]]) -> List[Tuple[int, float, float, float, float]]:
    if len(rows) <= MAX_BOXES_PER_IMAGE:
        return rows
    scored = [
        (idx, row, row[3] * row[4])
        for idx, row in enumerate(rows)
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    keep = sorted(scored[:MAX_BOXES_PER_IMAGE], key=lambda x: x[0])
    return [row for _, row, _ in keep]


def infer_ir1_class(stem: str) -> str:
    if stem.startswith("HELICOPTER"):
        return "helicopter"
    if stem.startswith("DRONE"):
        return "aircraft"
    if stem.startswith("AIRPLANE"):
        return "aircraft"
    return "heat_signature"


def read_image_shape(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def load_ir_system_dataset(dataset_dir: Path) -> List[Tuple[Path, List[Tuple[int, float, float, float, float]]]]:
    pairs: List[Tuple[Path, List[Tuple[int, float, float, float, float]]]] = []
    subsets = [
        dataset_dir / "dataset_IR_1" / "dataset_IR_1",
        dataset_dir / "dataset_IR_2" / "dataset_IR_2",
    ]
    for subset in subsets:
        if not subset.exists():
            continue
        for image_path in sorted(subset.glob("*.png")):
            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                continue
            try:
                width, height = read_image_shape(image_path)
            except ValueError:
                continue
            rows: List[Tuple[int, float, float, float, float]] = []
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = [float(x) for x in line.replace("\t", " ").split() if x.strip()]
                    if len(tokens) < 4:
                        continue
                    x1, y1, x2, y2 = tokens[:4]
                    x_center, y_center, box_w, box_h = xyxy_to_yolo(x1, y1, x2, y2, width, height)
                    if "dataset_IR_1" in str(image_path):
                        class_name = infer_ir1_class(image_path.stem.upper())
                    else:
                        class_name = "heat_signature"
                    class_id = CLASS_NAME_TO_ID.get(class_name, CLASS_NAME_TO_ID["heat_signature"])
                    rows.append((class_id, x_center, y_center, box_w, box_h))
            if rows:
                pairs.append((image_path, truncate_boxes(rows)))
    return pairs


def load_military_aircraft_dataset(dataset_dir: Path, dataset_map: Dict[str, str]) -> List[Tuple[Path, List[Tuple[int, float, float, float, float]]]]:
    csv_path = dataset_dir / "labels_with_split.csv"
    image_root = dataset_dir / "dataset"
    if not csv_path.exists() or not image_root.exists():
        return []
    grouped: Dict[Path, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row["filename"]
            width = float(row["width"])
            height = float(row["height"])
            class_token = row["class"]
            class_name = dataset_map.get(class_token, normalize_class_name(class_token))
            class_id = CLASS_NAME_TO_ID.get(class_name, CLASS_NAME_TO_ID["heat_signature"])
            x1 = float(row["xmin"])
            y1 = float(row["ymin"])
            x2 = float(row["xmax"])
            y2 = float(row["ymax"])
            x_center, y_center, box_w, box_h = xyxy_to_yolo(x1, y1, x2, y2, width, height)
            image_path = image_root / f"{file_name}.jpg"
            grouped[image_path].append((class_id, x_center, y_center, box_w, box_h))
    return [
        (image_path, truncate_boxes(rows))
        for image_path, rows in grouped.items()
        if rows and image_path.exists()
    ]


def parse_coco_subset(json_path: Path, image_dir: Path, dataset_map: Dict[str, str]) -> List[Tuple[Path, List[Tuple[int, float, float, float, float]]]]:
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: cat for cat in data.get("categories", [])}
    annotations: Dict[int, List[Dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        annotations[ann["image_id"]].append(ann)

    pairs: List[Tuple[Path, List[Tuple[int, float, float, float, float]]]] = []
    for image_id, meta in images.items():
        width = float(meta["width"])
        height = float(meta["height"])
        file_name = meta["file_name"]
        candidate_paths = [
            image_dir / file_name,
            image_dir.parent / file_name,
            image_dir.parent / Path(file_name).name,
        ]
        image_path = next((p for p in candidate_paths if p.exists()), None)
        if image_path is None:
            continue
        rows: List[Tuple[int, float, float, float, float]] = []
        for ann in annotations.get(image_id, []):
            cat = categories.get(ann["category_id"], {})
            cat_name = cat.get("name", "heat_signature")
            class_name = dataset_map.get(cat_name, dataset_map.get(str(ann["category_id"]), normalize_class_name(cat_name)))
            class_id = CLASS_NAME_TO_ID.get(class_name, CLASS_NAME_TO_ID["heat_signature"])
            x, y, w, h = ann["bbox"]
            x_center, y_center, box_w, box_h = xyxy_to_yolo(x, y, x + w, y + h, width, height)
            rows.append((class_id, x_center, y_center, box_w, box_h))
        if rows:
            pairs.append((image_path, truncate_boxes(rows)))
    return pairs


def load_normal_vehicle_dataset(dataset_dir: Path, dataset_map: Dict[str, str]) -> List[Tuple[Path, List[Tuple[int, float, float, float, float]]]]:
    subsets = [
        "images_rgb_train",
        "images_rgb_val",
        "images_thermal_train",
        "images_thermal_val",
        "video_rgb_test",
        "video_thermal_test",
    ]
    pairs: List[Tuple[Path, List[Tuple[int, float, float, float, float]]]] = []
    for subset in subsets:
        subset_dir = dataset_dir / subset
        json_path = subset_dir / "coco.json"
        image_dir = subset_dir / "data"
        if not json_path.exists() or not image_dir.exists():
            continue
        pairs.extend(parse_coco_subset(json_path, image_dir, dataset_map))
    return pairs


def gather_generic_pairs(dataset_dir: Path, dataset_map: Dict[str, str]) -> List[Tuple[Path, List[Tuple[int, float, float, float, float]]]]:
    pairs: List[Tuple[Path, List[Tuple[int, float, float, float, float]]]] = []
    image_files = discover_image_files(dataset_dir)
    xml_files = list(dataset_dir.rglob("*.xml"))
    xml_index = {xml_path.with_suffix("").name: xml_path for xml_path in xml_files}

    for image_path in image_files:
        label_candidates = [
            image_path.with_suffix(".txt"),
            image_path.parent / "labels" / f"{image_path.stem}.txt",
        ]
        rows: List[Tuple[int, float, float, float, float]] = []
        for candidate in label_candidates:
            if candidate.exists():
                rows = parse_yolo_label_file(candidate, dataset_map)
                if rows:
                    break
        if not rows:
            xml_path = xml_index.get(image_path.stem)
            if xml_path:
                rows = xml_to_yolo(xml_path, CLASS_NAME_TO_ID)
        if rows:
            pairs.append((image_path, truncate_boxes(rows)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge thermal datasets into YOLO format.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[
            "Ir System dataset",
            "Miliitary aircraft detection dataset",
            "Normal Vehicles Dataset",
        ],
        help="Dataset sub-folders under the detected datasets directory",
    )
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--map-config", type=Path, default=None, help="JSON file describing dataset-specific class remaps")
    args = parser.parse_args()

    paths = project_paths()
    mapping_cfg = load_mapping_config(args.map_config)

    image_label_pairs: List[Tuple[Path, List[Tuple[int, float, float, float, float]]]] = []

    for dataset_name in args.datasets:
        dataset_dir = paths["datasets"] / dataset_name
        if not dataset_dir.exists():
            print(f"[WARN] Dataset {dataset_dir} missing. Skipping.")
            continue

        dataset_map = mapping_cfg.get(dataset_name, {})
        dataset_key = dataset_name.lower()
        print(f"Scanning {dataset_name} ...")
        if "ir system" in dataset_key:
            pairs = load_ir_system_dataset(dataset_dir)
        elif "miliitary aircraft" in dataset_key:
            pairs = load_military_aircraft_dataset(dataset_dir, dataset_map)
        elif "normal vehicles" in dataset_key:
            pairs = load_normal_vehicle_dataset(dataset_dir, dataset_map)
        else:
            pairs = gather_generic_pairs(dataset_dir, dataset_map)
        print(f"  -> collected {len(pairs)} labeled samples")
        image_label_pairs.extend(pairs)

    if not image_label_pairs:
        raise SystemExit("No annotations were discovered. Please verify dataset paths and label formats.")

    train_pairs, val_pairs = train_test_split(
        image_label_pairs,
        train_size=args.train_split,
        random_state=args.seed,
        shuffle=True,
    )

    merged_root = paths["merged"]
    for split_name, pairs in (("train", train_pairs), ("valid", val_pairs)):
        img_dest = ensure_dir(merged_root / "images" / split_name)
        lbl_dest = ensure_dir(merged_root / "labels" / split_name)
        for src_img, rows in tqdm(pairs, desc=f"Writing {split_name}"):
            prefix = src_img.stem
            dest_img = img_dest / f"{prefix}{src_img.suffix.lower()}"
            shutil.copy2(src_img, dest_img)

            label_path = lbl_dest / f"{prefix}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                for row in rows:
                    class_id, *coords = row
                    coord_str = " ".join(f"{c:.6f}" for c in coords)
                    f.write(f"{class_id} {coord_str}\n")

    print(f"Merged dataset ready under {merged_root}")

    # Generate data.yaml
    data_yaml_content = f"""path: {merged_root.resolve()}
train: images/train
val: images/valid
nc: {len(CLASS_NAME_TO_ID)}
names:
"""
    # Sort by ID to ensure correct mapping
    sorted_classes = sorted(CLASS_NAME_TO_ID.items(), key=lambda x: x[1])
    for name, idx in sorted_classes:
        data_yaml_content += f"  {idx}: {name}\n"

    yaml_path = merged_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    print(f"Generated {yaml_path}")


if __name__ == "__main__":
    main()
