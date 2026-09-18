"""Validate locally supplied Re-ID datasets without downloading or changing them."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Pattern

import cv2

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
MARKET_ID_PATTERN = re.compile(r"^(?P<identity>-?\d+)_")
MSMT_ID_PATTERN = re.compile(r"^(?P<identity>\d+)_")


def find_images(directory: Path) -> list[Path]:
    """Find supported image files recursively, returning an empty list if absent."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def inspect_images(paths: Iterable[Path], pattern: Pattern[str] | None = None) -> dict[str, Any]:
    """Return actual image count, corruption, dimensions, and reliable filename IDs."""
    files = list(paths)
    corrupt: list[str] = []
    dimensions: set[tuple[int, int]] = set()
    identities: set[str] = set()
    for path in files:
        if pattern:
            match = pattern.match(path.name)
            if match and match.group("identity") != "-1":
                identities.add(match.group("identity"))
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            corrupt.append(str(path))
        else:
            height, width = image.shape[:2]
            dimensions.add((width, height))
    return {"image_count": len(files), "corrupt_images": len(corrupt), "corrupt_image_files": corrupt,
            "image_dimensions": [{"width": w, "height": h} for w, h in sorted(dimensions)],
            "identities": len(identities) if pattern else None}


def _split(path: Path, pattern: Pattern[str]) -> dict[str, Any]:
    return inspect_images(find_images(path), pattern) if path.is_dir() else {"image_count": None}


def validate_market1501(dataset_path: Path) -> dict[str, Any]:
    """Validate common Market-1501 directories and standard filename IDs."""
    if not dataset_path.is_dir():
        return {"status": "NOT_FOUND", "warnings": ["Dataset directory is missing."]}
    names = ("bounding_box_train", "query", "bounding_box_test", "gt_bbox")
    all_stats = inspect_images(find_images(dataset_path), MARKET_ID_PATTERN)
    return {"status": "FOUND", "recognized_directories": [n for n in names if (dataset_path / n).is_dir()],
            "missing_directories": [n for n in names if not (dataset_path / n).is_dir()],
            "train_images": _split(dataset_path / "bounding_box_train", MARKET_ID_PATTERN)["image_count"],
            "query_images": _split(dataset_path / "query", MARKET_ID_PATTERN)["image_count"],
            "gallery_images": _split(dataset_path / "bounding_box_test", MARKET_ID_PATTERN)["image_count"],
            "total_images": all_stats["image_count"], "identities": all_stats["identities"],
            "corrupt_images": all_stats["corrupt_images"], "corrupt_image_files": all_stats["corrupt_image_files"],
            "image_dimensions": all_stats["image_dimensions"], "warnings": []}


def _list_images(root: Path, name: str) -> list[Path] | None:
    listing = root / name
    if not listing.is_file():
        return None
    try:
        lines = listing.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    paths = []
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if parts:
            candidate = root / parts[0]
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(candidate)
    return paths


def validate_msmt17(dataset_path: Path) -> dict[str, Any]:
    """Validate MSMT17 directories and split-list entries that exist on disk."""
    if not dataset_path.is_dir():
        return {"status": "NOT_FOUND", "warnings": ["Dataset directory is missing."]}
    lists = {n: _list_images(dataset_path, n) for n in ("list_train.txt", "list_val.txt", "list_query.txt", "list_gallery.txt")}
    all_stats = inspect_images(find_images(dataset_path), MSMT_ID_PATTERN)
    missing = [n for n in ("train", "test") if not (dataset_path / n).is_dir()]
    train = lists["list_train.txt"] or find_images(dataset_path / "train")
    gallery = lists["list_gallery.txt"] or find_images(dataset_path / "test")
    query = lists["list_query.txt"]
    return {"status": "FOUND", "recognized_directories": [n for n in ("train", "test") if (dataset_path / n).is_dir()],
            "available_list_files": [n for n, paths in lists.items() if paths is not None], "missing_directories": missing,
            "train_images": len(train), "query_images": len(query) if query is not None else None,
            "gallery_images": len(gallery), "total_images": all_stats["image_count"], "identities": all_stats["identities"],
            "corrupt_images": all_stats["corrupt_images"], "corrupt_image_files": all_stats["corrupt_image_files"],
            "image_dimensions": all_stats["image_dimensions"],
            "warnings": [f"Missing expected directories: {', '.join(missing)}."] if missing else []}


def validate_datasets(reid_root: Path) -> dict[str, Any]:
    """Create a JSON-serializable report for available optional datasets."""
    reid_root = Path(reid_root)
    return {"validation_timestamp": datetime.now(timezone.utc).isoformat(), "dataset_root": str(reid_root),
            "supported_image_extensions": sorted(IMAGE_EXTENSIONS),
            "datasets": {"market1501": validate_market1501(reid_root / "market1501"),
                         "msmt17": validate_msmt17(reid_root / "msmt17")}}
