# Validate the dataset structure, image integrity and expected counts
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Required Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow;
      5. no split or class is empty.

    The report must not contain timestamps because it is a DVC output.
    """

    report = {
        "status": "PASS",
        "classes": sorted(EXPECTED_CLASSES),
        "splits": {},
        "total_images": 0
    }

    for split_name, expected_count in EXPECTED_SPLITS.items():

        # Check 1: Required split exists
        split_path = raw_root / split_name

        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Missing required split: {split_path}"
            )

        # Check 2: Exactly the expected classes exist
        actual_classes = set()

        for directory in split_path.iterdir():
            if directory.is_dir():
                actual_classes.add(directory.name)

        if actual_classes != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Incorrect classes in {split_name}: "
                f"expected {sorted(EXPECTED_CLASSES)}, "
                f"found {sorted(actual_classes)}"
            )

        # Initialize counters for the current split
        class_counts = {}
        split_total = 0

        for class_name in sorted(EXPECTED_CLASSES):

            class_path = split_path / class_name

            # Identify supported image files
            image_files = []

            for file in class_path.iterdir():
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append(file)

            # Check that the class is not empty
            if len(image_files) == 0:
                raise RuntimeError(
                    f"Empty class: {class_path}"
                )

            # Verify that every image can be fully decoded
            for image_path in sorted(image_files):
                with Image.open(image_path) as image:
                    image.load()

            # Count images
            class_counts[class_name] = len(image_files)
            split_total += len(image_files)

        # Verify the expected image count
        if split_total != expected_count:
            raise RuntimeError(
                f"Incorrect image count in {split_name}: "
                f"expected {expected_count}, "
                f"found {split_total}"
            )

        # Store the split results
        report["splits"][split_name] = {
            "images": split_total,
            "class_counts": class_counts
        }

        # Update the total dataset count
        report["total_images"] += split_total

    return report

def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
