import csv

import cv2
import numpy as np
import pytest
import torch

from src.config import Config, load_config
from src.env_check import DeviceInfo
from src.reid.embedder import (
    STATUS_EMPTY, STATUS_NOT_FOUND, STATUS_OK, ReIDEmbedder, ReIDError,
    discover_crop_images, parse_args, parse_crop_metadata, resolve_device, validate_reid_config,
)


def make_crop(path, value=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((32, 16, 3), value, dtype=np.uint8))


class FakeOSNet:
    def to(self, device):
        self.device = str(device)
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        return torch.arange(batch.shape[0] * 4, dtype=torch.float32, device=batch.device).reshape(batch.shape[0], 4) + 1


def fake_embedder(**config):
    embedder = ReIDEmbedder({"device": "cpu", **config})
    embedder._model = FakeOSNet()
    return embedder


def test_reid_config_defaults_validation_and_project_section():
    assert validate_reid_config({})["image_width"] == 128
    assert ReIDEmbedder.from_config(Config({"reid": {"device": "cpu"}})).device == "cpu"
    assert load_config(apply_env=False).reid.output_dir == "outputs/embeddings"
    with pytest.raises(ReIDError):
        validate_reid_config({"device": "tpu"})
    with pytest.raises(ReIDError):
        validate_reid_config({"batch_size": 0})


def test_device_selection_cpu_and_cuda_fallback(monkeypatch):
    assert resolve_device("cpu") == "cpu"
    import src.env_check as env_check
    monkeypatch.setattr(env_check, "detect_device", lambda prefer_cuda=True: DeviceInfo(True, "cpu", False))
    assert resolve_device("cuda") == "cpu"


def test_discovery_uses_phase_five_layout_and_posix_crop_path(tmp_path):
    root = tmp_path / "CAM01"
    make_crop(root / "track_007" / "frame_000020.jpg")
    make_crop(root / "track_007" / "ignored.bmp")
    records = discover_crop_images(root)
    assert len(records) == 1
    assert (records[0].camera_id, records[0].track_id, records[0].frame_number) == ("CAM01", 7, 20)
    assert records[0].crop_path == "CAM01/track_007/frame_000020.jpg"


def test_metadata_parser_ignores_malformed_rows(tmp_path):
    metadata = tmp_path / "crops.csv"
    metadata.write_text(
        "camera_id,track_id,frame_number,crop_path\nCAM01,3,5,CAM01/track_003/frame_000005.jpg\nCAM01,bad,5,bad.jpg\n",
        encoding="utf-8",
    )
    parsed = parse_crop_metadata(metadata)
    assert parsed["CAM01/track_003/frame_000005.jpg"] == ("CAM01", 3, 5)
    assert "bad.jpg" not in parsed


def test_missing_and_empty_crop_directories_are_reported(tmp_path):
    assert fake_embedder().extract_directory(tmp_path / "missing").status == STATUS_NOT_FOUND
    assert fake_embedder().extract_directory(tmp_path / "empty").status == STATUS_NOT_FOUND
    empty = tmp_path / "CAM01"; empty.mkdir()
    assert fake_embedder().extract_directory(empty).status == STATUS_EMPTY


def test_embedding_extraction_writes_track_npy_and_metadata(tmp_path):
    crops = tmp_path / "CAM01"
    make_crop(crops / "track_002" / "frame_000005.jpg", 10)
    make_crop(crops / "track_002" / "frame_000010.png", 30)
    output = tmp_path / "embeddings"
    result = fake_embedder(batch_size=1).extract_directory(crops, output)
    assert result.status == STATUS_OK
    assert result.images_processed == result.embeddings_generated == 2
    assert result.embedding_dimension == 4
    array = np.load(output / "CAM01" / "track_002" / "embeddings.npy")
    assert array.shape == (2, 4)
    assert np.allclose(np.linalg.norm(array, axis=1), 1.0)
    with (output / "CAM01" / "embeddings.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["embedding_index"] for row in rows] == ["0", "1"]
    assert rows[0]["crop_path"] == "CAM01/track_002/frame_000005.jpg"


def test_normalization_can_be_disabled(tmp_path):
    crops = tmp_path / "CAM01"; make_crop(crops / "track_001" / "frame_000000.jpg")
    output = tmp_path / "embeddings"
    fake_embedder(normalize_embeddings=False).extract_directory(crops, output)
    vector = np.load(output / "CAM01" / "track_001" / "embeddings.npy")[0]
    assert np.linalg.norm(vector) != pytest.approx(1.0)


def test_unreadable_image_is_skipped(tmp_path):
    crops = tmp_path / "CAM01" / "track_001"; crops.mkdir(parents=True)
    (crops / "frame_000000.jpg").write_bytes(b"not an image")
    result = fake_embedder().extract_directory(crops.parent.parent, tmp_path / "embeddings")
    assert result.status == STATUS_OK
    assert result.images_skipped == 1
    assert result.embeddings_generated == 0


def test_cli_arguments_are_parsed():
    args = parse_args(["--input", "outputs/crops/CAM01", "--device", "cpu", "--batch-size", "2", "--max-images", "5"])
    assert args.device == "cpu" and args.batch_size == 2 and args.max_images == 5
