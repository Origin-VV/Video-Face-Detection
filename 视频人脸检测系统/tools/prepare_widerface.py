import argparse
import shutil
from pathlib import Path

import cv2

# 解析命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WIDER FACE annotations to YOLO detection format."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("datasets/WIDER_FACE"),
        help="Raw WIDER FACE root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/WIDER_FACE_YOLO"),
        help="Converted YOLO dataset output directory.",
    )
    return parser.parse_args()

# 执行转换主逻辑
def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()

    split_root = raw_root / "wider_face_split"
    train_images_root = raw_root / "WIDER_train" / "images"
    val_images_root = raw_root / "WIDER_val" / "images"

    required_paths = [
        split_root / "wider_face_train_bbx_gt.txt",
        split_root / "wider_face_val_bbx_gt.txt",
        train_images_root,
        val_images_root,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(missing_paths)
        raise FileNotFoundError(f"Missing WIDER FACE files:\n{missing_text}")

    convert_split(
        images_root=train_images_root,
        annotation_path=split_root / "wider_face_train_bbx_gt.txt",
        output_root=output_root,
        split_name="train",
    )
    convert_split(
        images_root=val_images_root,
        annotation_path=split_root / "wider_face_val_bbx_gt.txt",
        output_root=output_root,
        split_name="val",
    )

    print(f"Done. YOLO dataset created at: {output_root}")

# 转换数据集分栏
def convert_split(
    images_root: Path,
    annotation_path: Path,
    output_root: Path,
    split_name: str,
) -> None:
    output_images_root = output_root / "images" / split_name
    output_labels_root = output_root / "labels" / split_name
    output_images_root.mkdir(parents=True, exist_ok=True)
    output_labels_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_boxes = 0

    for image_rel_path, boxes in read_wider_annotations(annotation_path):
        source_image_path = images_root / image_rel_path
        if not source_image_path.exists():
            raise FileNotFoundError(f"Image not found: {source_image_path}")

        image = cv2.imread(str(source_image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {source_image_path}")

        image_height, image_width = image.shape[:2]
        yolo_lines = []
        for box in boxes:
            yolo_line = wider_box_to_yolo_line(box, image_width, image_height)
            if yolo_line:
                yolo_lines.append(yolo_line)

        target_image_path = output_images_root / image_rel_path
        target_label_path = (output_labels_root / image_rel_path).with_suffix(".txt")
        target_image_path.parent.mkdir(parents=True, exist_ok=True)
        target_label_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_image_path, target_image_path)
        target_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        total_images += 1
        total_boxes += len(yolo_lines)
        if total_images % 500 == 0:
            print(
                f"[{split_name}] converted {total_images} images, "
                f"{total_boxes} valid boxes"
            )

    print(f"[{split_name}] finished: {total_images} images, {total_boxes} valid boxes")

# 读取原始标注信息
def read_wider_annotations(annotation_path: Path):
    lines = annotation_path.read_text(encoding="utf-8").splitlines()
    index = 0
    total_lines = len(lines)

    while index < total_lines:
        image_rel_path = lines[index].strip()
        index += 1
        if not image_rel_path:
            continue

        face_count = int(lines[index].strip())
        index += 1

        boxes = []
        if face_count == 0:
            if index < total_lines and looks_like_box_line(lines[index]):
                index += 1
            yield image_rel_path, boxes
            continue

        for _ in range(face_count):
            parts = [int(value) for value in lines[index].split()]
            index += 1

            if len(parts) < 4:
                continue

            x, y, width, height = parts[:4]
            invalid = parts[7] if len(parts) > 7 else 0
            if width <= 0 or height <= 0 or invalid == 1:
                continue

            boxes.append((x, y, width, height))

        yield image_rel_path, boxes

# 识别标注行格式
def looks_like_box_line(line: str) -> bool:
    parts = line.split()
    if len(parts) < 4:
        return False
    try:
        [int(value) for value in parts[:4]]
    except ValueError:
        return False
    return True

# 坐标转为YOLO格式
def wider_box_to_yolo_line(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> str | None:
    x, y, width, height = box

    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(image_width), float(x + width))
    y2 = min(float(image_height), float(y + height))

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    center_x = (x1 + x2) / 2.0 / image_width
    center_y = (y1 + y2) / 2.0 / image_height
    norm_width = clipped_width / image_width
    norm_height = clipped_height / image_height

    return f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"

if __name__ == "__main__":
    main()