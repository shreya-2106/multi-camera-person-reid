from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api import searches


def _make_crop(root: Path, camera: str, track_id: int, filename: str) -> Path:
    directory = root / f"search_23" / "outputs" / "crops" / camera / f"track_{track_id:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / filename
    Image.new("RGB", (12, 18), (40, 80, 120)).save(image, format="JPEG")
    return image


def test_evidence_listing_returns_only_real_crops_for_accepted_global_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "analyses"
    _make_crop(root, "CAM01", 7, "frame_000001.jpg")
    _make_crop(root, "CAM02", 10, "frame_000002.jpg")
    _make_crop(root, "CAM02", 11, "frame_000003.jpg")
    monkeypatch.setattr(searches, "ANALYSIS_STORAGE_DIR", root)
    monkeypatch.setattr(searches, "_get_owned_search", lambda *_: object())
    monkeypatch.setattr(
        searches,
        "_stored_search_result",
        lambda *_: {
            "result": {
                "videos": [{"camera_id": "CAM01"}, {"camera_id": "CAM02"}],
                "global_identities": [
                    {"global_person_id": 4, "camera_id": "CAM01", "track_id": 7, "decision": "new_identity"},
                    {"global_person_id": 4, "camera_id": "CAM02", "track_id": 10, "decision": "accept"},
                    {"global_person_id": 5, "camera_id": "CAM02", "track_id": 11, "decision": "new_identity"},
                ],
            }
        },
    )

    response = searches.list_search_evidence(
        23,
        user=SimpleNamespace(id=8),
        db=object(),
    )

    assert [(image["camera_id"], image["track_id"]) for image in response["images"]] == [
        ("CAM01", 7),
        ("CAM02", 10),
    ]
    assert all(image["image_url"].startswith("/api/searches/23/evidence/") for image in response["images"])


def test_evidence_listing_returns_empty_when_generated_crops_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(searches, "ANALYSIS_STORAGE_DIR", tmp_path)
    monkeypatch.setattr(searches, "_get_owned_search", lambda *_: object())
    monkeypatch.setattr(
        searches,
        "_stored_search_result",
        lambda *_: {"result": {"videos": [{"camera_id": "CAM01"}], "global_identities": []}},
    )

    response = searches.list_search_evidence(
        23,
        user=SimpleNamespace(id=8),
        db=object(),
    )

    assert response == {"images": []}


@pytest.mark.parametrize("filename", ["..", "../../app.db", "frame_1.png", "frame_000001.jpg/../../secret"])
def test_evidence_image_rejects_invalid_filename(
    filename: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(searches, "_get_owned_search", lambda *_: object())

    with pytest.raises(HTTPException) as error:
        searches.get_search_evidence_image(
            23,
            "CAM01",
            7,
            filename,
            user=SimpleNamespace(id=8),
            db=object(),
        )

    assert error.value.status_code == 404


def test_evidence_image_requires_owned_search(
    monkeypatch: pytest.MonkeyPatch,
):
    def deny_missing_search(search_id, user_id, db):
        raise HTTPException(status_code=404, detail="Search not found.")

    monkeypatch.setattr(searches, "_get_owned_search", deny_missing_search)

    with pytest.raises(HTTPException) as error:
        searches.get_search_evidence_image(
            23,
            "CAM01",
            7,
            "frame_000001.jpg",
            user=SimpleNamespace(id=99),
            db=object(),
        )

    assert error.value.status_code == 404


def test_evidence_image_serves_only_valid_jpeg_inside_search_crop_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "analyses"
    image = _make_crop(root, "CAM01", 7, "frame_000001.jpg")
    monkeypatch.setattr(searches, "ANALYSIS_STORAGE_DIR", root)
    monkeypatch.setattr(searches, "_get_owned_search", lambda *_: object())
    monkeypatch.setattr(
        searches,
        "_stored_search_result",
        lambda *_: {
            "result": {
                "videos": [{"camera_id": "CAM01"}],
                "global_identities": [{"camera_id": "CAM01", "track_id": 7}],
            }
        },
    )

    response = searches.get_search_evidence_image(
        23,
        "CAM01",
        7,
        "frame_000001.jpg",
        user=SimpleNamespace(id=8),
        db=object(),
    )

    assert Path(response.path) == image
    assert response.media_type == "image/jpeg"


def test_evidence_image_rejects_corrupt_jpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "analyses"
    image = _make_crop(root, "CAM01", 7, "frame_000001.jpg")
    image.write_bytes(b"not a jpeg")
    monkeypatch.setattr(searches, "ANALYSIS_STORAGE_DIR", root)
    monkeypatch.setattr(searches, "_get_owned_search", lambda *_: object())
    monkeypatch.setattr(
        searches,
        "_stored_search_result",
        lambda *_: {
            "result": {
                "videos": [{"camera_id": "CAM01"}],
                "global_identities": [{"camera_id": "CAM01", "track_id": 7}],
            }
        },
    )

    with pytest.raises(HTTPException) as error:
        searches.get_search_evidence_image(
            23,
            "CAM01",
            7,
            "frame_000001.jpg",
            user=SimpleNamespace(id=8),
            db=object(),
        )

    assert error.value.status_code == 404


def test_evidence_http_endpoint_requires_authentication_and_serves_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "analyses"
    image = _make_crop(root, "CAM01", 7, "frame_000001.jpg")
    monkeypatch.setattr(searches, "ANALYSIS_STORAGE_DIR", root)
    monkeypatch.setattr(
        searches,
        "_stored_search_result",
        lambda *_: {
            "result": {
                "videos": [{"camera_id": "CAM01"}],
                "global_identities": [{"camera_id": "CAM01", "track_id": 7}],
            }
        },
    )

    class FakeDB:
        def scalar(self, _statement):
            return SimpleNamespace(id=23, user_id=8)

        def close(self):
            pass

    def get_db_override():
        yield FakeDB()

    app = FastAPI()
    app.include_router(searches.router)
    app.dependency_overrides[searches.get_db] = get_db_override

    with TestClient(app) as client:
        unauthenticated = client.get("/api/searches/23/evidence/CAM01/7/frame_000001.jpg")
        assert unauthenticated.status_code == 401

        app.dependency_overrides[searches.get_current_user] = lambda: SimpleNamespace(id=8)
        response = client.get("/api/searches/23/evidence/CAM01/7/frame_000001.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.content == image.read_bytes()


def test_evidence_http_endpoint_denies_other_users(
    monkeypatch: pytest.MonkeyPatch,
):
    class EmptyDB:
        def scalar(self, _statement):
            return None

        def close(self):
            pass

    def get_db_override():
        yield EmptyDB()

    app = FastAPI()
    app.include_router(searches.router)
    app.dependency_overrides[searches.get_db] = get_db_override
    app.dependency_overrides[searches.get_current_user] = lambda: SimpleNamespace(id=99)

    with TestClient(app) as client:
        response = client.get("/api/searches/23/evidence/CAM01/7/frame_000001.jpg")

    assert response.status_code == 404
