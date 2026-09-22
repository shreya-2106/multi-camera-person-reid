import csv

import pytest

from src.reid.types import EMBEDDING_CSV_FIELDNAMES, EmbeddingMetadata, write_embeddings_csv


def test_embedding_metadata_has_local_crop_association_fields():
    item = EmbeddingMetadata("CAM01", 7, 20, "CAM01/track_007/frame_000020.jpg", 0)
    assert set(item.to_dict()) == set(EMBEDDING_CSV_FIELDNAMES)


@pytest.mark.parametrize("changes", [{"camera_id": ""}, {"track_id": -1}, {"frame_number": -1}, {"embedding_index": -1}])
def test_embedding_metadata_rejects_invalid_values(changes):
    values = dict(camera_id="CAM01", track_id=1, frame_number=0, crop_path="CAM01/track_001/frame_000000.jpg", embedding_index=0)
    values.update(changes)
    with pytest.raises(ValueError):
        EmbeddingMetadata(**values)


def test_write_embeddings_csv_writes_header_and_rows(tmp_path):
    output = write_embeddings_csv([EmbeddingMetadata("CAM01", 1, 0, "CAM01/track_001/frame_000000.jpg", 0)], tmp_path / "nested" / "embeddings.csv")
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["camera_id"] == "CAM01"
    assert rows[0]["embedding_index"] == "0"
