"""
YOLO-based person detection.

Provides a `Detector` class that wraps the Ultralytics YOLO Python package
to detect people (COCO "person" class only) in individual frames or entire
videos. This module does NOT perform tracking or re-identification -- every
frame's detections are independent.

Model weights are loaded lazily (on first use, or via explicit
`load_model()`), so simply importing/constructing a `Detector` never
triggers a network call or GPU allocation.

Run standalone:
    python -m src.detection.detector --input data/videos/cam01.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from src.detection.types import Detection, DetectionVideoResult, write_detections_csv

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "auto"
DEFAULT_CLASSES = ("person",)

STATUS_OK = "OK"
STATUS_NOT_FOUND = "NOT_FOUND"
STATUS_UNREADABLE = "UNREADABLE"
STATUS_ERROR = "ERROR"


class DetectorError(Exception):
    """Raised when the detection model fails to load or run."""


def resolve_device(device_setting: str) -> str:
    """
    Resolve a "device" config value ("auto" | "cpu" | "cuda") to a concrete
    device string, falling back to CPU if CUDA was requested/preferred but
    is not actually available.
    """
    from src.env_check import detect_device

    setting = (device_setting or "auto").lower()
    if setting == "cpu":
        return "cpu"

    info = detect_device(prefer_cuda=True)
    if setting == "cuda":
        if info.cuda_available:
            return "cuda"
        logger.warning("device='cuda' requested but CUDA is not available; falling back to CPU")
        return "cpu"

    # "auto"
    return info.device


def resolve_model_path(model_name_or_path: str) -> str:
    """
    Resolve a configured model value to a concrete path/name to hand to
    Ultralytics.

    - A bare model name with no path separator (e.g. "yolov8n.pt") is left
      untouched, so Ultralytics' own cache/download resolution applies.
    - A relative path containing a separator (e.g. "models/yolov8n.pt") is
      resolved against the project root, so it behaves the same regardless
      of the current working directory the CLI/tests were launched from.
    - An absolute path is left untouched.
    """
    from src.config import PROJECT_ROOT

    candidate = Path(model_name_or_path)
    if candidate.is_absolute():
        return str(candidate)
    if len(candidate.parts) == 1:
        return model_name_or_path  # bare model name, e.g. "yolov8n.pt"
    return str(PROJECT_ROOT / candidate)


class Detector:
    """
    Loads a YOLO model and runs person-only detection on frames or videos.

    Example:
        detector = Detector(model="yolov8n.pt", confidence=0.5, device="auto")
        detections = detector.detect_frame(frame, camera_id="CAM01", frame_number=0)
        result = detector.detect_video("data/videos/cam01.mp4")
        detector.close()
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        device: str = DEFAULT_DEVICE,
        imgsz: int = DEFAULT_IMGSZ,
        classes: Sequence[str] = DEFAULT_CLASSES,
    ):
        self.model_name_or_path = model
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = resolve_device(device)
        self.class_names_filter = tuple(classes)

        self._model: Any = None
        self._class_id_filter: Optional[List[int]] = None
        self._id_to_name: dict = {}

    @classmethod
    def from_config(cls, cfg) -> "Detector":
        """Build a Detector from the project's `detection:` config section."""
        det_cfg = cfg.get("detection", {}) if hasattr(cfg, "get") else {}
        return cls(
            model=det_cfg.get("model", DEFAULT_MODEL),
            confidence=det_cfg.get("confidence", DEFAULT_CONFIDENCE),
            iou=det_cfg.get("iou", DEFAULT_IOU),
            device=det_cfg.get("device", DEFAULT_DEVICE),
            imgsz=det_cfg.get("imgsz", DEFAULT_IMGSZ),
            classes=det_cfg.get("classes", list(DEFAULT_CLASSES)),
        )

    # -- model loading -----------------------------------------------------

    def _create_backend_model(self, model_name_or_path: str) -> Any:
        """
        Construct the underlying Ultralytics YOLO model object.

        Isolated into its own method (rather than inlined in load_model)
        specifically so tests can monkeypatch this single seam and exercise
        all the detection/parsing/IO logic without downloading real weights
        or running real inference.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "The 'ultralytics' package is not installed. "
                "Install it with: pip install ultralytics"
            ) from exc

        try:
            return YOLO(model_name_or_path)
        except Exception as exc:  # ultralytics raises a variety of exception types
            message = str(exc).lower()
            if "internet" in message or "connection" in message or "timed out" in message:
                raise DetectorError(
                    f"Could not load YOLO model '{model_name_or_path}': it is not cached "
                    "locally and downloading the pretrained weights failed (no internet "
                    "connection?). Download the weights manually or check your network."
                ) from exc
            raise DetectorError(f"Failed to load YOLO model '{model_name_or_path}': {exc}") from exc

    def load_model(self) -> None:
        """Load the YOLO model and resolve the person class id, if not already loaded."""
        if self._model is not None:
            return

        logger.info("Loading YOLO model '%s' on device '%s'", self.model_name_or_path, self.device)
        resolved_path = resolve_model_path(self.model_name_or_path)
        Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
        model = self._create_backend_model(resolved_path)

        names = getattr(model, "names", None) or {}
        # ultralytics exposes `names` as {int: str}; normalize defensively.
        self._id_to_name = {int(k): str(v) for k, v in dict(names).items()}

        wanted = {n.lower() for n in self.class_names_filter}
        class_ids = [cid for cid, name in self._id_to_name.items() if name.lower() in wanted]

        if not class_ids:
            raise DetectorError(
                f"None of the configured classes {list(self.class_names_filter)} were found "
                f"in the model's class list: {sorted(self._id_to_name.values())}"
            )

        self._class_id_filter = sorted(class_ids)
        self._model = model
        logger.info(
            "Model loaded. Filtering to class id(s) %s (%s)",
            self._class_id_filter,
            ", ".join(self.class_names_filter),
        )

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load_model()
        return self._model

    def close(self) -> None:
        """Release the loaded model reference."""
        self._model = None
        self._class_id_filter = None
        self._id_to_name = {}

    # -- frame-level detection ----------------------------------------------

    def detect_frame(
        self,
        frame: Any,
        camera_id: Optional[str] = None,
        frame_number: int = 0,
        timestamp_seconds: Optional[float] = None,
    ) -> List[Detection]:
        """
        Run person detection on a single OpenCV (BGR numpy array) frame.

        Returns a list of Detection objects. Every call is independent --
        no tracking or temporal state is kept between frames.
        """
        self.load_model()

        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self._class_id_filter,
            verbose=False,
        )
        return self._parse_results(results, camera_id, frame_number, timestamp_seconds)

    def _parse_results(
        self,
        results: Iterable[Any],
        camera_id: Optional[str],
        frame_number: int,
        timestamp_seconds: Optional[float],
    ) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy
            confs = boxes.conf
            clss = boxes.cls
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                conf_val = float(confs[i])
                cls_val = int(clss[i])
                class_name = self._id_to_name.get(cls_val, str(cls_val))
                detections.append(
                    Detection(
                        camera_id=camera_id,
                        frame_number=frame_number,
                        timestamp_seconds=timestamp_seconds,
                        class_id=cls_val,
                        class_name=class_name,
                        confidence=conf_val,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        return detections

    # -- video-level detection -----------------------------------------------

    def detect_video(
        self,
        video_path: Path,
        camera_id: Optional[str] = None,
        output_csv: Optional[Path] = None,
        annotate: bool = False,
        annotated_output_path: Optional[Path] = None,
        max_frames: Optional[int] = None,
    ) -> DetectionVideoResult:
        """
        Run person detection over every frame of a video, sequentially
        (frames are never all loaded into memory at once).

        Gracefully handles a missing file, an unreadable video, zero FPS,
        and model-loading errors -- these are reported via the returned
        DetectionVideoResult's `status`/`error` fields rather than raised.
        """
        video_path = Path(video_path)

        if camera_id is None:
            from src.data.video_validator import infer_camera_id

            camera_id = infer_camera_id(video_path.name)

        if not video_path.is_file():
            logger.warning("Video not found: %s", video_path)
            return DetectionVideoResult(
                video_path=str(video_path),
                camera_id=camera_id,
                status=STATUS_NOT_FOUND,
                error="File not found",
            )

        try:
            import cv2
        except ImportError:
            return DetectionVideoResult(
                video_path=str(video_path),
                camera_id=camera_id,
                status=STATUS_ERROR,
                error="opencv-python is not installed",
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            logger.warning("Video could not be opened: %s", video_path)
            return DetectionVideoResult(
                video_path=str(video_path),
                camera_id=camera_id,
                status=STATUS_UNREADABLE,
                error="File could not be opened by OpenCV (unsupported/corrupt codec or container)",
            )

        # Only load the (potentially expensive) YOLO model once we know the
        # video itself is actually readable.
        try:
            self.load_model()
        except DetectorError as exc:
            cap.release()
            logger.error("Model loading failed: %s", exc)
            return DetectionVideoResult(
                video_path=str(video_path),
                camera_id=camera_id,
                status=STATUS_ERROR,
                error=str(exc),
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_raw) if fps_raw and fps_raw > 0 else None
        if fps is None:
            logger.warning("Video reports 0/invalid FPS; timestamps will be unavailable: %s", video_path)

        writer = None
        if annotate:
            if annotated_output_path is None:
                annotated_output_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
            annotated_output_path = Path(annotated_output_path)
            annotated_output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(annotated_output_path), fourcc, fps or 25.0, (width or 0, height or 0)
            )

        result = DetectionVideoResult(
            video_path=str(video_path),
            camera_id=camera_id,
            status=STATUS_OK,
            fps=fps,
            width=width,
            height=height,
        )

        start_time = time.monotonic()
        frame_number = 0
        try:
            while True:
                if max_frames is not None and frame_number >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                timestamp = (frame_number / fps) if fps else None
                frame_detections = self.detect_frame(
                    frame, camera_id=camera_id, frame_number=frame_number, timestamp_seconds=timestamp
                )
                result.detections.extend(frame_detections)

                if writer is not None:
                    annotated_frame = draw_detections(frame, frame_detections)
                    writer.write(annotated_frame)

                frame_number += 1
        except Exception as exc:  # never let a bad frame crash the whole run
            logger.exception("Error while processing frame %s of %s", frame_number, video_path)
            result.status = STATUS_ERROR
            result.error = str(exc)
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        result.frames_processed = frame_number
        result.detection_count = len(result.detections)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Processed %s: %d frames, %d person detections in %.1fs",
            video_path.name,
            result.frames_processed,
            result.detection_count,
            elapsed,
        )

        if output_csv is not None and result.status != STATUS_ERROR:
            output_csv = Path(output_csv)
            write_detections_csv(result.detections, output_csv)
            result.output_csv_path = str(output_csv)
            logger.info("Wrote detections CSV: %s", output_csv)

        if writer is not None and result.status != STATUS_ERROR:
            result.annotated_video_path = str(annotated_output_path)
            logger.info("Wrote annotated video: %s", annotated_output_path)

        return result


def draw_detections(frame: Any, detections: List[Detection]) -> Any:
    """Draw person bounding boxes + confidence onto a copy of the frame."""
    import cv2

    annotated = frame.copy()
    for det in detections:
        pt1 = (int(det.x1), int(det.y1))
        pt2 = (int(det.x2), int(det.y2))
        cv2.rectangle(annotated, pt1, pt2, (0, 200, 0), 2)
        label = f"person {det.confidence:.2f}"
        cv2.putText(
            annotated, label, (pt1[0], max(pt1[1] - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1
        )
    return annotated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: run YOLO person detection on a CCTV video."
    )
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the detections CSV (default: outputs/detections/<CAM>_detections.csv)",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the main config YAML")
    parser.add_argument("--model", default=None, help="Override the configured YOLO model name/path")
    parser.add_argument("--confidence", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="Override IoU threshold")
    parser.add_argument("--device", default=None, help="Override device: auto | cpu | cuda")
    parser.add_argument("--annotate", action="store_true", help="Also write an annotated output video")
    parser.add_argument(
        "--annotated-output", default=None, help="Path for the annotated video (implies --annotate)"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (for quick tests)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from src.config import PROJECT_ROOT, load_config
    from src.logging_setup import setup_logging

    args = parse_args(argv)
    setup_logging(level="INFO")

    try:
        cfg = load_config(config_path=args.config, apply_env=False)
    except Exception as exc:
        print(f"[FATAL] Failed to load config: {exc}", file=sys.stderr)
        return 1

    det_cfg = cfg.get("detection", {})
    detector = Detector(
        model=args.model or det_cfg.get("model", DEFAULT_MODEL),
        confidence=args.confidence if args.confidence is not None else det_cfg.get("confidence", DEFAULT_CONFIDENCE),
        iou=args.iou if args.iou is not None else det_cfg.get("iou", DEFAULT_IOU),
        device=args.device or det_cfg.get("device", DEFAULT_DEVICE),
        imgsz=det_cfg.get("imgsz", DEFAULT_IMGSZ),
        classes=det_cfg.get("classes", list(DEFAULT_CLASSES)),
    )

    video_path = Path(args.input)

    output_csv = Path(args.output) if args.output else None
    if output_csv is None:
        from src.data.video_validator import infer_camera_id

        camera_id = infer_camera_id(video_path.name) or video_path.stem
        out_dir = det_cfg.get("output_dir", "outputs/detections")
        out_dir_path = Path(out_dir)
        if not out_dir_path.is_absolute():
            out_dir_path = PROJECT_ROOT / out_dir_path
        output_csv = out_dir_path / f"{camera_id}_detections.csv"

    annotated_output_path = Path(args.annotated_output) if args.annotated_output else None
    annotate = args.annotate or annotated_output_path is not None
    if annotate and annotated_output_path is None:
        from src.data.video_validator import infer_camera_id

        camera_id = infer_camera_id(video_path.name) or video_path.stem
        ann_dir = det_cfg.get("annotated_output_dir", "outputs/annotated")
        ann_dir_path = Path(ann_dir)
        if not ann_dir_path.is_absolute():
            ann_dir_path = PROJECT_ROOT / ann_dir_path
        annotated_output_path = ann_dir_path / f"{camera_id}_annotated.mp4"

    result = detector.detect_video(
        video_path,
        output_csv=output_csv,
        annotate=annotate,
        annotated_output_path=annotated_output_path,
        max_frames=args.max_frames,
    )
    detector.close()

    print(f"\nDetection status: {result.status}")
    print(f"Video:            {result.video_path}")
    print(f"Camera ID:        {result.camera_id}")
    if result.status == STATUS_OK:
        print(f"Frames processed: {result.frames_processed}")
        print(f"Detections:       {result.detection_count}")
        print(f"CSV output:       {result.output_csv_path}")
        if result.annotated_video_path:
            print(f"Annotated video:  {result.annotated_video_path}")
    if result.error:
        print(f"Note:             {result.error}")

    return 0 if result.status == STATUS_OK else 1


if __name__ == "__main__":
    sys.exit(main())
