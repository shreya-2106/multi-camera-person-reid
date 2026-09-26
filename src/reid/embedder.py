"""Phase 7: OSNet inference for Phase 5 person crops.

This module produces appearance vectors only. It supports either the
default pretrained OSNet weights or an optional locally trained OSNet
checkpoint. It never compares vectors, assigns global identities, or
performs cross-camera matching.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from src.reid.types import (
    EmbeddingExtractionResult,
    EmbeddingMetadata,
    write_embeddings_csv,
)

logger = logging.getLogger(__name__)

STATUS_OK, STATUS_NOT_FOUND, STATUS_EMPTY, STATUS_ERROR = (
    "OK",
    "NOT_FOUND",
    "EMPTY",
    "ERROR",
)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
TRACK_PATTERN = re.compile(r"^track_(?P<track_id>\d+)$", re.IGNORECASE)
FRAME_PATTERN = re.compile(r"^frame_(?P<frame_number>\d+)$", re.IGNORECASE)

DEFAULT_REID_CONFIG = {
    "enabled": True,
    "model": "osnet",
    "device": "auto",
    "image_width": 128,
    "image_height": 256,
    "batch_size": 16,
    "normalize_embeddings": True,
    "output_dir": "outputs/embeddings",
    "checkpoint": None,
}


class ReIDError(ValueError):
    """Raised for invalid Re-ID settings or unavailable dependencies."""


@dataclass(frozen=True)
class CropRecord:
    """A valid Phase 5 crop and its camera/track/frame identifiers."""

    path: Path
    camera_id: str
    track_id: int
    frame_number: int
    crop_path: str


def validate_reid_config(config: Optional[dict]) -> dict:
    """Merge and validate the explicit Re-ID configuration."""
    value = {**DEFAULT_REID_CONFIG, **(config or {})}

    if not isinstance(value["enabled"], bool):
        raise ReIDError("reid.enabled must be a boolean")

    if str(value["model"]).lower() not in {"osnet", "osnet_x1_0"}:
        raise ReIDError("reid.model must be 'osnet' or 'osnet_x1_0'")

    if str(value["device"]).lower() not in {"auto", "cpu", "cuda"}:
        raise ReIDError("reid.device must be 'auto', 'cpu', or 'cuda'")

    for key in ("image_width", "image_height", "batch_size"):
        if (
            not isinstance(value[key], int)
            or isinstance(value[key], bool)
            or value[key] < 1
        ):
            raise ReIDError(f"reid.{key} must be a positive integer")

    if not isinstance(value["normalize_embeddings"], bool):
        raise ReIDError("reid.normalize_embeddings must be a boolean")

    if not isinstance(value["output_dir"], str) or not value["output_dir"].strip():
        raise ReIDError("reid.output_dir must be a non-empty string")

    if value["checkpoint"] is not None:
        if not isinstance(value["checkpoint"], str) or not value["checkpoint"].strip():
            raise ReIDError(
                "reid.checkpoint must be null or a non-empty string"
            )

    return value


def resolve_device(device_setting: str) -> str:
    """Resolve a requested device, falling back cleanly when CUDA is absent."""
    from src.env_check import detect_device

    setting = (device_setting or "auto").lower()

    if setting == "cpu":
        return "cpu"

    info = detect_device(prefer_cuda=True)

    if setting == "cuda" and not info.cuda_available:
        logger.warning(
            "reid.device='cuda' requested but CUDA is unavailable; "
            "falling back to CPU"
        )
        return "cpu"

    return info.device


def _posix_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def parse_crop_metadata(
    metadata_path: Path,
) -> dict[str, tuple[str, int, int]]:
    """Read usable Phase 5 CSV rows keyed by crop_path."""
    metadata_path = Path(metadata_path)

    if not metadata_path.is_file():
        return {}

    records: dict[str, tuple[str, int, int]] = {}

    try:
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                try:
                    crop_path = _posix_path(row["crop_path"])
                    records[crop_path] = (
                        row["camera_id"],
                        int(row["track_id"]),
                        int(row["frame_number"]),
                    )
                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "Skipping malformed crop metadata row in %s",
                        metadata_path,
                    )

    except OSError as exc:
        logger.warning(
            "Could not read crop metadata %s: %s",
            metadata_path,
            exc,
        )

    return records


def _path_identifiers(
    path: Path,
    input_root: Path,
) -> Optional[tuple[str, int, int, str]]:
    """Infer camera, track, and frame values from the Phase 5 layout."""
    track_match = TRACK_PATTERN.match(path.parent.name)
    frame_match = FRAME_PATTERN.match(path.stem)

    if not track_match or not frame_match:
        return None

    relative = path.relative_to(input_root)

    if TRACK_PATTERN.match(input_root.name):
        return None

    if TRACK_PATTERN.match(relative.parts[0]) if relative.parts else False:
        camera_id = input_root.name
        crop_path = _posix_path(Path(camera_id) / relative)

    elif len(relative.parts) >= 3:
        camera_id = relative.parts[0]
        crop_path = _posix_path(relative)

    else:
        return None

    return (
        camera_id,
        int(track_match.group("track_id")),
        int(frame_match.group("frame_number")),
        crop_path,
    )


def discover_crop_images(input_directory: Path) -> list[CropRecord]:
    """Discover valid Phase 5 crop images in deterministic order."""
    root = Path(input_directory)

    if not root.is_dir():
        return []

    records: list[CropRecord] = []
    metadata = parse_crop_metadata(root / "crops.csv")

    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ):
        values = _path_identifiers(path, root)

        if values is None:
            continue

        camera_id, track_id, frame_number, crop_path = values

        csv_values = metadata.get(crop_path)

        if csv_values is not None:
            camera_id, track_id, frame_number = csv_values

        records.append(
            CropRecord(
                path,
                camera_id,
                track_id,
                frame_number,
                crop_path,
            )
        )

    return records


class ReIDEmbedder:
    """Load OSNet and turn Phase 5 crops into normalized appearance vectors."""

    def __init__(self, reid_config: Optional[dict] = None):
        self.config = validate_reid_config(reid_config)
        self.device = resolve_device(self.config["device"])
        self._model: Any = None

    @classmethod
    def from_config(cls, cfg: Any) -> "ReIDEmbedder":
        return cls(dict(cfg.get("reid", {})) if hasattr(cfg, "get") else {})

    def _create_backend_model(self) -> Any:
        """Build OSNet using either the trained checkpoint or pretrained weights."""
        try:
            from torchreid.reid.models import build_model
        except ImportError as exc:
            raise ReIDError(
                "The torchreid OSNet runtime or one of its dependencies "
                "is missing. Install requirements.txt."
            ) from exc

        checkpoint_setting = self.config.get("checkpoint")

        # No checkpoint configured:
        # preserve the original Phase 6 pretrained OSNet behavior.
        if checkpoint_setting is None:
            try:
                return build_model(
                    name="osnet_x1_0",
                    num_classes=1000,
                    pretrained=True,
                )
            except Exception as exc:
                raise ReIDError(
                    "Could not load pretrained OSNet weights. They may need "
                    "an initial download; check network access or the "
                    f"torchreid cache. Details: {exc}"
                ) from exc

        checkpoint_path = Path(checkpoint_setting)

        if not checkpoint_path.is_absolute():
            from src.config import PROJECT_ROOT

            checkpoint_path = PROJECT_ROOT / checkpoint_path

        if not checkpoint_path.is_file():
            raise ReIDError(
                f"Re-ID checkpoint not found: {checkpoint_path}"
            )

        try:
            import torch

            model = build_model(
                name="osnet_x1_0",
                num_classes=751,
                pretrained=False,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            if not isinstance(checkpoint, dict):
                raise ReIDError(
                    "Re-ID checkpoint must contain a dictionary."
                )

            state_dict = checkpoint.get("model_state_dict")

            if state_dict is None:
                raise ReIDError(
                    "Re-ID checkpoint does not contain "
                    "'model_state_dict'."
                )

            model.load_state_dict(state_dict, strict=True)

            logger.info(
                "Loaded trained OSNet checkpoint: %s",
                checkpoint_path,
            )

            if "epoch" in checkpoint:
                logger.info(
                    "Checkpoint epoch: %s",
                    checkpoint["epoch"],
                )

            if "accuracy" in checkpoint:
                logger.info(
                    "Checkpoint training accuracy: %.2f%%",
                    float(checkpoint["accuracy"]),
                )

            return model

        except ReIDError:
            raise

        except Exception as exc:
            raise ReIDError(
                f"Could not load trained OSNet checkpoint "
                f"{checkpoint_path}. Details: {exc}"
            ) from exc

    def load_model(self) -> None:
        """Load once, put OSNet in evaluation mode, and place it on the selected device."""
        if self._model is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise ReIDError(
                "The 'torch' package is required for OSNet inference. "
                "Install requirements.txt."
            ) from exc

        logger.info("Loading OSNet Re-ID model...")
        logger.info("Device: %s", self.device)

        self._model = (
            self._create_backend_model()
            .to(torch.device(self.device))
            .eval()
        )

    def _preprocess(self, image: np.ndarray) -> Any:
        """Convert OpenCV BGR pixels to OSNet's RGB normalized tensor."""
        try:
            import cv2
            import torch
        except ImportError as exc:
            raise ReIDError(
                "OpenCV and torch are required for crop preprocessing."
            ) from exc

        if image is None or image.ndim not in (2, 3):
            raise ReIDError("Crop image is invalid")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (
                self.config["image_width"],
                self.config["image_height"],
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )

        mean = torch.tensor(
            (0.485, 0.456, 0.406),
            dtype=tensor.dtype,
        ).view(3, 1, 1)

        std = torch.tensor(
            (0.229, 0.224, 0.225),
            dtype=tensor.dtype,
        ).view(3, 1, 1)

        return (tensor - mean) / std

    def _embed_batch(self, batch: Any) -> np.ndarray:
        self.load_model()

        import torch

        with torch.inference_mode():
            embeddings = self._model(batch.to(self.device))

            if isinstance(embeddings, (tuple, list)):
                embeddings = embeddings[0]

            if (
                not isinstance(embeddings, torch.Tensor)
                or embeddings.ndim != 2
                or embeddings.shape[0] != batch.shape[0]
            ):
                raise ReIDError(
                    "OSNet returned embeddings with an invalid shape"
                )

            if self.config["normalize_embeddings"]:
                embeddings = torch.nn.functional.normalize(
                    embeddings,
                    p=2,
                    dim=1,
                )

            return (
                embeddings
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

    def extract_directory(
        self,
        input_directory: Path,
        output_dir: Optional[Path] = None,
        max_images: Optional[int] = None,
    ) -> EmbeddingExtractionResult:
        """Embed crops and write NPY vectors plus CSV metadata."""
        root = Path(input_directory)

        camera_id = (
            root.name
            if root.is_dir() and not TRACK_PATTERN.match(root.name)
            else None
        )

        result = EmbeddingExtractionResult(
            str(root),
            camera_id,
            STATUS_OK,
            device=self.device,
        )

        if not root.is_dir():
            result.status = STATUS_NOT_FOUND
            result.error = "Input crop directory not found"
            return result

        records = discover_crop_images(root)

        if max_images is not None:
            records = records[:max(0, max_images)]

        if not records:
            result.status = STATUS_EMPTY
            result.error = "No valid Phase 5 crop images found"
            return result

        destination = Path(
            output_dir or self.config["output_dir"]
        )

        result.output_directory = str(destination)

        grouped: dict[
            tuple[str, int],
            list[tuple[CropRecord, np.ndarray]],
        ] = {}

        pending_records: list[CropRecord] = []
        pending_tensors: list[Any] = []

        def flush() -> None:
            if not pending_tensors:
                return

            import torch

            vectors = self._embed_batch(
                torch.stack(pending_tensors)
            )

            result.embedding_dimension = int(vectors.shape[1])

            for record, vector in zip(
                pending_records,
                vectors,
            ):
                grouped.setdefault(
                    (record.camera_id, record.track_id),
                    [],
                ).append((record, vector))

                result.embeddings_generated += 1

            pending_records.clear()
            pending_tensors.clear()

        try:
            import cv2

            for record in records:
                image = cv2.imread(
                    str(record.path),
                    cv2.IMREAD_UNCHANGED,
                )

                if image is None:
                    result.images_skipped += 1
                    logger.warning(
                        "Skipping unreadable crop: %s",
                        record.path,
                    )
                    continue

                pending_records.append(record)
                pending_tensors.append(
                    self._preprocess(image)
                )

                result.images_processed += 1

                if (
                    len(pending_tensors)
                    >= self.config["batch_size"]
                ):
                    flush()

            flush()

        except Exception as exc:
            logger.exception(
                "Re-ID embedding extraction failed"
            )
            result.status = STATUS_ERROR
            result.error = str(exc)
            return result

        per_camera: dict[
            str,
            list[EmbeddingMetadata],
        ] = {}

        for (camera, track_id), values in grouped.items():
            track_output = (
                destination
                / camera
                / f"track_{track_id:03d}"
            )

            track_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                track_output / "embeddings.npy",
                np.stack(
                    [vector for _, vector in values]
                ),
            )

            for index, (record, _) in enumerate(values):
                per_camera.setdefault(
                    camera,
                    [],
                ).append(
                    EmbeddingMetadata(
                        camera,
                        track_id,
                        record.frame_number,
                        record.crop_path,
                        index,
                    )
                )

        for camera, metadata in per_camera.items():
            metadata.sort(
                key=lambda item: (
                    item.track_id,
                    item.frame_number,
                )
            )

            result.metadata_csv_paths.append(
                str(
                    write_embeddings_csv(
                        metadata,
                        destination
                        / camera
                        / "embeddings.csv",
                    )
                )
            )

        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7: extract OSNet embeddings from "
            "Phase 5 crops."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Crop camera directory or crops root",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    from src.config import PROJECT_ROOT, load_config
    from src.logging_setup import setup_logging

    args = parse_args(argv)

    setup_logging(level="INFO")

    cfg = load_config(
        args.config,
        apply_env=False,
    )

    reid_config = dict(cfg.get("reid", {}))

    if args.device is not None:
        reid_config["device"] = args.device

    if args.batch_size is not None:
        reid_config["batch_size"] = args.batch_size

    embedder = ReIDEmbedder(reid_config)

    output = Path(
        args.output_dir
        or embedder.config["output_dir"]
    )

    if not output.is_absolute():
        output = PROJECT_ROOT / output

    result = embedder.extract_directory(
        Path(args.input),
        output,
        args.max_images,
    )

    print(
        f"Re-ID embedding extraction status: {result.status}\n"
        f"Camera ID: {result.camera_id}\n"
        f"Images processed: {result.images_processed}\n"
        f"Embeddings generated: {result.embeddings_generated}\n"
        f"Images skipped: {result.images_skipped}\n"
        f"Embedding dimension: {result.embedding_dimension}\n"
        f"Device: {result.device}\n"
        f"Output directory: {result.output_directory}"
    )

    if result.error:
        print(f"Note: {result.error}")

    return 0 if result.status == STATUS_OK else 1


if __name__ == "__main__":
    sys.exit(main())