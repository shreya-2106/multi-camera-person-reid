import numpy as np
import pytest

from src.reid.matcher import FAISSMatcher, MatcherError


def test_build_index_and_search():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metadata = [
        {
            "camera_id": "CAM01",
            "track_id": "1",
            "frame_number": "0",
            "crop_path": "CAM01/track_001/frame_000000.jpg",
            "embedding_index": "0",
        },
        {
            "camera_id": "CAM01",
            "track_id": "1",
            "frame_number": "5",
            "crop_path": "CAM01/track_001/frame_000005.jpg",
            "embedding_index": "1",
        },
        {
            "camera_id": "CAM02",
            "track_id": "7",
            "frame_number": "10",
            "crop_path": "CAM02/track_007/frame_000010.jpg",
            "embedding_index": "0",
        },
    ]

    matcher = FAISSMatcher(similarity_threshold=0.0)
    matcher.build_index(embeddings, metadata)

    results = matcher.search(embeddings[0], top_k=2)

    assert len(results) == 2
    assert results[0].camera_id == "CAM01"
    assert results[0].track_id == 1
    assert results[0].frame_number == 0
    assert results[0].similarity == pytest.approx(1.0)


def test_metadata_length_must_match_embeddings():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metadata = [
        {
            "camera_id": "CAM01",
            "track_id": "1",
            "frame_number": "0",
            "crop_path": "crop.jpg",
            "embedding_index": "0",
        }
    ]

    matcher = FAISSMatcher()

    with pytest.raises(MatcherError):
        matcher.build_index(embeddings, metadata)


def test_search_before_building_index_fails():
    matcher = FAISSMatcher()

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    with pytest.raises(MatcherError):
        matcher.search(query)


def test_metadata_mapping_for_multiple_tracks(tmp_path):
    embeddings_root = tmp_path / "embeddings"
    camera_dir = embeddings_root / "CAM01"

    track_001 = camera_dir / "track_001"
    track_002 = camera_dir / "track_002"

    track_001.mkdir(parents=True)
    track_002.mkdir(parents=True)

    np.save(
        track_001 / "embeddings.npy",
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    np.save(
        track_002 / "embeddings.npy",
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ],
            dtype=np.float32,
        ),
    )

    metadata_path = camera_dir / "embeddings.csv"

    metadata_path.write_text(
        "camera_id,track_id,frame_number,crop_path,embedding_index\n"
        "CAM01,1,0,CAM01/track_001/frame_000000.jpg,0\n"
        "CAM01,1,5,CAM01/track_001/frame_000005.jpg,1\n"
        "CAM01,2,20,CAM01/track_002/frame_000020.jpg,0\n"
        "CAM01,2,25,CAM01/track_002/frame_000025.jpg,1\n",
        encoding="utf-8",
    )

    matcher = FAISSMatcher(similarity_threshold=0.0)

    count = matcher.build_from_directory(embeddings_root)

    assert count == 4

    query = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float32,
    )

    results = matcher.search(query, top_k=2)

    assert results[0].track_id == 2
    assert results[0].frame_number == 20
    assert results[0].similarity == pytest.approx(1.0)

    assert results[1].track_id == 2
    assert results[1].frame_number == 25


def test_invalid_similarity_threshold():
    with pytest.raises(ValueError):
        FAISSMatcher(similarity_threshold=1.5)