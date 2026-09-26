import numpy as np
import pytest

from src.reid.track_matcher import TrackMatcher, TrackMatcherError


def test_compare_embeddings_returns_expected_statistics():
    matcher = TrackMatcher()

    track_a = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    track_b = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = matcher.compare_embeddings(track_a, track_b)

    assert result.mean_similarity == pytest.approx(0.5)
    assert result.max_similarity == pytest.approx(1.0)
    assert result.comparison_count == 4


def test_compare_embeddings_rejects_dimension_mismatch():
    matcher = TrackMatcher()

    track_a = np.ones((2, 512), dtype=np.float32)
    track_b = np.ones((3, 256), dtype=np.float32)

    with pytest.raises(TrackMatcherError):
        matcher.compare_embeddings(track_a, track_b)


def test_compare_embeddings_rejects_zero_norm_embedding():
    matcher = TrackMatcher()

    track_a = np.array(
        [
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    track_b = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(TrackMatcherError):
        matcher.compare_embeddings(track_a, track_b)


def test_load_track_embeddings_missing_file(tmp_path):
    matcher = TrackMatcher()

    missing_path = tmp_path / "missing.npy"

    with pytest.raises(TrackMatcherError):
        matcher.load_track_embeddings(missing_path)


def test_real_track_files_can_be_compared():
    matcher = TrackMatcher()

    result = matcher.compare_tracks(
        "outputs/embeddings/CAM01/track_001/embeddings.npy",
        "outputs/embeddings/CAM01/track_002/embeddings.npy",
    )

    assert result.comparison_count == 90
    assert 0.0 <= result.mean_similarity <= 1.0
    assert 0.0 <= result.max_similarity <= 1.0
    assert result.max_similarity >= result.mean_similarity