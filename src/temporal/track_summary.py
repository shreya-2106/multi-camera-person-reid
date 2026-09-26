"""Build track-level temporal summaries from tracking CSV files."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .types import TrackSummary


class TrackSummaryError(RuntimeError):
    """Raised when track summaries cannot be created."""


REQUIRED_COLUMNS = {
    "camera_id",
    "frame_number",
    "timestamp_seconds",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
}


class TrackSummaryBuilder:
    """Build temporal summaries from frame-level tracking records."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def _read_rows(self) -> list[dict[str, str]]:
        """Read and validate tracking CSV rows."""
        if not self.csv_path.exists():
            raise TrackSummaryError(
                f"Tracking CSV does not exist: {self.csv_path}"
            )

        try:
            with self.csv_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                reader = csv.DictReader(handle)

                if reader.fieldnames is None:
                    raise TrackSummaryError(
                        "Tracking CSV does not contain a header"
                    )

                missing = REQUIRED_COLUMNS - set(reader.fieldnames)

                if missing:
                    raise TrackSummaryError(
                        "Tracking CSV is missing required columns: "
                        + ", ".join(sorted(missing))
                    )

                rows = list(reader)

        except OSError as exc:
            raise TrackSummaryError(
                f"Could not read tracking CSV: {self.csv_path}"
            ) from exc

        if not rows:
            raise TrackSummaryError(
                f"Tracking CSV contains no records: {self.csv_path}"
            )

        return rows

    @staticmethod
    def _parse_row(row: dict[str, str]) -> dict:
        """Convert one CSV row into validated numeric values."""
        try:
            camera_id = str(row["camera_id"]).strip()
            frame_number = int(row["frame_number"])
            timestamp_seconds = float(row["timestamp_seconds"])
            track_id = int(row["track_id"])

            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])

        except (KeyError, TypeError, ValueError) as exc:
            raise TrackSummaryError(
                f"Invalid tracking record: {row}"
            ) from exc

        if not camera_id:
            raise TrackSummaryError(
                "Tracking record contains an empty camera_id"
            )

        if frame_number < 0:
            raise TrackSummaryError(
                f"frame_number must be non-negative: {frame_number}"
            )

        if timestamp_seconds < 0.0:
            raise TrackSummaryError(
                "timestamp_seconds must be non-negative"
            )

        if track_id < 0:
            raise TrackSummaryError(
                f"track_id must be non-negative: {track_id}"
            )

        if x2 < x1 or y2 < y1:
            raise TrackSummaryError(
                f"Invalid bounding box: {(x1, y1, x2, y2)}"
            )

        return {
            "camera_id": camera_id,
            "frame_number": frame_number,
            "timestamp_seconds": timestamp_seconds,
            "track_id": track_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

    @staticmethod
    def _center(row: dict) -> tuple[float, float]:
        """Calculate the center point of a bounding box."""
        center_x = (row["x1"] + row["x2"]) / 2.0
        center_y = (row["y1"] + row["y2"]) / 2.0

        return center_x, center_y

    def build(self) -> list[TrackSummary]:
        """Build one TrackSummary for every local camera track."""
        rows = self._read_rows()

        grouped: defaultdict[
            tuple[str, int],
            list[dict],
        ] = defaultdict(list)

        for raw_row in rows:
            row = self._parse_row(raw_row)

            key = (
                row["camera_id"],
                row["track_id"],
            )

            grouped[key].append(row)

        summaries: list[TrackSummary] = []

        for (camera_id, track_id), track_rows in sorted(grouped.items()):
            track_rows.sort(
                key=lambda item: (
                    item["frame_number"],
                    item["timestamp_seconds"],
                )
            )

            first = track_rows[0]
            last = track_rows[-1]

            start_center_x, start_center_y = self._center(first)
            end_center_x, end_center_y = self._center(last)

            summary = TrackSummary(
                camera_id=camera_id,
                track_id=track_id,
                start_frame=first["frame_number"],
                end_frame=last["frame_number"],
                start_time=first["timestamp_seconds"],
                end_time=last["timestamp_seconds"],
                start_center_x=start_center_x,
                start_center_y=start_center_y,
                end_center_x=end_center_x,
                end_center_y=end_center_y,
                frames_seen=len(track_rows),
            )

            summaries.append(summary)

        return summaries