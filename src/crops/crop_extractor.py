"""Extract sampled, quality-filtered crops from existing local ByteTrack tracks."""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
from src.crops.types import CropExtractionResult, CropMetadata, write_crops_csv

logger = logging.getLogger(__name__)
STATUS_OK, STATUS_NOT_FOUND, STATUS_UNREADABLE, STATUS_ERROR = "OK", "NOT_FOUND", "UNREADABLE", "ERROR"
DEFAULT_CROP_CONFIG = {"enabled": True, "output_dir": "outputs/crops", "min_width": 40, "min_height": 80, "min_area": 3200, "padding": .05, "save_every_n_frames": 5}
class CropExtractionError(ValueError): pass

def validate_crop_config(config: Optional[dict]) -> dict:
    value = {**DEFAULT_CROP_CONFIG, **(config or {})}
    if not isinstance(value["enabled"], bool): raise CropExtractionError("crops.enabled must be a boolean")
    if not isinstance(value["output_dir"], str) or not value["output_dir"].strip(): raise CropExtractionError("crops.output_dir must be a non-empty string")
    for key in ("min_width", "min_height", "min_area", "save_every_n_frames"):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 1: raise CropExtractionError(f"crops.{key} must be a positive integer")
    if not isinstance(value["padding"], (int, float)) or isinstance(value["padding"], bool) or not 0 <= value["padding"] <= 1: raise CropExtractionError("crops.padding must be a number in [0, 1]")
    return value

def should_save_frame(frame_number: int, every: int) -> bool: return frame_number >= 0 and frame_number % every == 0
def clip_box(box: Sequence[float], width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if len(box) != 4 or width <= 0 or height <= 0: return None
    try: x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError): return None
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)): return None
    result = max(0, int(np.floor(x1))), max(0, int(np.floor(y1))), min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    return result if result[2] > result[0] and result[3] > result[1] else None
def pad_box(box, padding, width, height):
    x1, y1, x2, y2 = box; return clip_box((x1-(x2-x1)*padding, y1-(y2-y1)*padding, x2+(x2-x1)*padding, y2+(y2-y1)*padding), width, height)

class CropExtractor:
    def __init__(self, crop_config=None, detector=None, tracker=None): self.config, self.detector, self.tracker = validate_crop_config(crop_config), detector, tracker
    @classmethod
    def from_config(cls, cfg, detector=None, tracker=None): return cls(dict(cfg.get("crops", {})), detector, tracker)
    def extract_track_crop(self, frame, track, source_video, output_directory):
        import cv2
        if frame is None or frame.ndim < 2: return None, "invalid_frame"
        height, width = frame.shape[:2]; original = (track.x1, track.y1, track.x2, track.y2)
        clipped = clip_box(original, width, height)
        if clipped is None: return None, "invalid_box"
        final = pad_box(clipped, self.config["padding"], width, height)
        if final is None: return None, "invalid_padded_box"
        x1,y1,x2,y2 = final; crop_width, crop_height = x2-x1, y2-y1
        if crop_width < self.config["min_width"]: return None, "min_width"
        if crop_height < self.config["min_height"]: return None, "min_height"
        if crop_width*crop_height < self.config["min_area"]: return None, "min_area"
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None, "empty_crop"
        crop_path = Path(output_directory) / f"track_{track.track_id:03d}" / f"frame_{track.frame_number:06d}.jpg"; crop_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(crop_path), crop): return None, "write_failed"
        return CropMetadata(track.camera_id or "UNKNOWN", track.track_id, track.frame_number, str(source_video), crop_path.relative_to(Path(output_directory).parent).as_posix(), *original, x1,y1,x2,y2,crop_width,crop_height,crop_width*crop_height,track.confidence), None
    def extract_video(self, video_path, camera_id=None, output_dir=None, max_frames=None):
        import cv2
        from src.data.video_validator import infer_camera_id
        from src.detection import Detector
        from src.tracking import Tracker
        video_path = Path(video_path); camera_id = camera_id or infer_camera_id(video_path.name) or video_path.stem; output = Path(output_dir or self.config["output_dir"]) / camera_id
        result = CropExtractionResult(str(video_path), camera_id, STATUS_OK, output_directory=str(output))
        if not video_path.is_file(): result.status, result.error = STATUS_NOT_FOUND, "File not found"; return result
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): cap.release(); result.status, result.error = STATUS_UNREADABLE, "File could not be opened by OpenCV"; return result
        detector, tracker = self.detector or Detector(), self.tracker or Tracker(); fps = cap.get(cv2.CAP_PROP_FPS) or 0; metadata=[]; number=0
        try:
            while max_frames is None or number < max_frames:
                ok, frame=cap.read()
                if not ok or frame is None: break
                tracks=tracker.update(detector.detect_frame(frame,camera_id,number,number/fps if fps else None)); result.tracks_seen += len(tracks)
                if should_save_frame(number,self.config["save_every_n_frames"]):
                    for track in tracks:
                        item, reason=self.extract_track_crop(frame,track,video_path,output)
                        if item: metadata.append(item); result.crops_saved += 1
                        else: result.crops_rejected += 1; result.rejection_counts[reason]=result.rejection_counts.get(reason,0)+1
                number += 1
        except Exception as exc: logger.exception("Crop extraction failed"); result.status,result.error=STATUS_ERROR,str(exc)
        finally: cap.release()
        result.frames_processed=number
        if result.status != STATUS_ERROR: result.metadata_csv_path=str(write_crops_csv(metadata,output/"crops.csv"))
        return result

def parse_args(argv=None):
    parser=argparse.ArgumentParser(description="Phase 5: extract person crops from local ByteTrack tracks."); parser.add_argument("--input",required=True); parser.add_argument("--max-frames",type=int); parser.add_argument("--camera-id"); parser.add_argument("--config",default="configs/config.yaml"); parser.add_argument("--output-dir"); return parser.parse_args(argv)
def main(argv=None):
    from src.config import PROJECT_ROOT,load_config
    from src.detection import Detector
    from src.logging_setup import setup_logging
    from src.tracking import Tracker
    args=parse_args(argv); setup_logging(); cfg=load_config(args.config,apply_env=False); extractor=CropExtractor.from_config(cfg,Detector.from_config(cfg),Tracker.from_config(cfg)); output=Path(args.output_dir or extractor.config["output_dir"]); output=output if output.is_absolute() else PROJECT_ROOT/output; result=extractor.extract_video(args.input,args.camera_id,output,args.max_frames)
    print(f"Crop extraction status: {result.status}\nCamera ID: {result.camera_id}\nFrames processed: {result.frames_processed}\nTracks seen: {result.tracks_seen}\nCrops saved: {result.crops_saved}\nCrops rejected: {result.crops_rejected}\nOutput directory: {result.output_directory}"); return 0 if result.status == STATUS_OK else 1
if __name__ == "__main__": sys.exit(main())
