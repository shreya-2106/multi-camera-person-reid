"""
Multi-Camera Person Re-Identification & Tracking System.

Top-level package for the project source code.

Sub-packages:
    detection  - Object/person detection (YOLO) [Phase 2+]
    tracking   - Single-camera multi-object tracking (ByteTrack) [Phase 2+]
    reid       - Re-identification embeddings (OSNet) [Phase 3+]
    matching   - Cross-camera identity matching (FAISS, etc.) [Phase 3+]
    topology   - Camera topology / spatial relationships [Phase 4+]
    temporal   - Temporal reasoning / time-window logic [Phase 4+]
    trajectory - Trajectory construction & smoothing [Phase 4+]
    database   - Persistence layer [Phase 5+]
    analytics  - Reporting & analytics [Phase 5+]
"""

__version__ = "0.1.0"
