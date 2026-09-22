import csv
from pathlib import Path
import numpy as np
import pytest
from src.config import load_config
from src.crops.crop_extractor import CropExtractor, CropExtractionError, clip_box, pad_box, should_save_frame, validate_crop_config
from src.tracking.types import TrackedPerson

def track(**changes):
    values=dict(camera_id="CAM01",frame_number=0,timestamp_seconds=0.0,track_id=7,x1=10.,y1=10.,x2=60.,y2=100.,confidence=.9,class_id=0,class_name="person"); values.update(changes); return TrackedPerson(**values)

def test_config_defaults_and_invalid_values():
    assert validate_crop_config({})["save_every_n_frames"] == 5
    with pytest.raises(CropExtractionError): validate_crop_config({"min_width": 0})
    with pytest.raises(CropExtractionError): validate_crop_config({"padding": 2})

def test_project_crop_config_loads_with_expected_defaults():
    config = load_config(apply_env=False)
    assert config.crops.output_dir == "outputs/crops"
    assert config.crops.save_every_n_frames == 5

def test_boxes_padding_and_sampling_are_deterministic():
    assert clip_box((-4, 5, 20, 30), 40, 40) == (0, 5, 20, 30)
    assert clip_box((5, 5, 5, 9), 40, 40) is None
    assert pad_box((0, 0, 20, 20), .5, 30, 30) == (0, 0, 30, 30)
    assert [n for n in range(7) if should_save_frame(n, 5)] == [0, 5]

def test_crop_writes_local_track_directory_and_metadata(tmp_path):
    extractor=CropExtractor({"min_width": 20,"min_height":40,"min_area":800,"padding":0,"output_dir":"unused"})
    item, reason=extractor.extract_track_crop(np.zeros((120,80,3),dtype=np.uint8),track(),Path("data/videos/cam01.mp4"),tmp_path/"CAM01")
    assert reason is None and item.track_id == 7 and item.camera_id == "CAM01"
    assert (tmp_path/"CAM01"/"track_007"/"frame_000000.jpg").exists()
    assert item.crop_path == "CAM01/track_007/frame_000000.jpg"

@pytest.mark.parametrize(("changes","reason"), [({"x1":70,"x2":60}, "invalid_box"), ({"x2":20}, "min_width"), ({"y2":30}, "min_height")])
def test_invalid_and_low_quality_crops_are_rejected(tmp_path, changes, reason):
    if changes.get("x1", 0) > changes.get("x2", 999):
        # TrackedPerson rightly rejects unordered data; use a lightweight invalid input instead.
        obj=type("Track",(),{**track().__dict__,**changes})()
    else: obj=track(**changes)
    _, actual=CropExtractor({"min_width":40,"min_height":80,"min_area":3200}).extract_track_crop(np.zeros((120,80,3),dtype=np.uint8),obj,Path("v.mp4"),tmp_path)
    assert actual == reason

def test_missing_video_reports_status(tmp_path):
    assert CropExtractor().extract_video(tmp_path/"missing.mp4").status == "NOT_FOUND"
