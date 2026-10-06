from __future__ import annotations

from typing import Any


def run_mock_analysis(
    *,
    camera_names: list[str],
) -> dict[str, Any]:
    """Return a deterministic mock result for Phase 15 UI testing."""

    cameras = [
        {
            "camera": camera_name,
            "status": "completed",
            "confidence": 0.94,
            "timeline": [
                {
                    "time": "00:12",
                    "event": "Person detected",
                },
                {
                    "time": "00:18",
                    "event": "Appearance matched",
                },
                {
                    "time": "00:27",
                    "event": "Camera transition",
                },
            ],
        }
        for camera_name in camera_names
    ]

    return {
        "match_found": True,
        "identity_label": "Global Person 1",
        "overall_confidence": 0.94,
        "cameras": cameras,
        "evidence": {
            "appearance_similarity": 0.91,
            "temporal_consistency": 0.96,
            "topology_valid": True,
        },
        "timeline": [
            {
                "camera": camera_name,
                "event": "Person observed",
            }
            for camera_name in camera_names
        ],
    }