from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])


def test_dashboard_result_selectors_match_the_active_html():
    html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "dashboard" / "app.js").read_text(encoding="utf-8")
    parser = IdCollector()
    parser.feed(html)
    ids = set(parser.ids)
    selectors = set(re.findall(r'\$\(\s*["\']([^"\']+)["\']\s*\)', script))

    required = {
        "identity-label",
        "overall-confidence",
        "confidence-fill",
        "match-banner",
        "result-cameras",
        "evidence-grid",
        "timeline",
        "investigation-summary",
        "matched-person-list",
    }

    assert required <= ids
    assert required <= selectors
    assert len(parser.ids) == len(ids), "Dashboard contains duplicate element IDs."
    assert "result-identity" not in selectors
    assert "camera-results-body" not in selectors
    assert "result-timeline" not in selectors


def test_result_renderer_uses_real_match_decisions_and_crop_endpoint():
    script = (ROOT / "dashboard" / "app.js").read_text(encoding="utf-8")

    assert 'identity.decision === "accept"' in script
    assert "No accepted cross-camera match" in script
    assert "/evidence" in script
    assert "loading = \"lazy\"" in script


def test_matched_person_summary_uses_backend_counts_and_relative_track_times():
    script = (ROOT / "dashboard" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")

    assert "summary.matched_people_count" in script
    assert "summary.total_tracks" in script
    assert "summary.total_detections" in script
    assert "observation.first_time_seconds" in script
    assert "observation.last_time_seconds" in script
    assert "formatVideoTime" in script
    assert "absolute camera clock metadata" in html


def test_dashboard_styles_current_layout_classes():
    styles = (ROOT / "dashboard" / "style.css").read_text(encoding="utf-8")

    for selector in (
        "#app-screen > .content",
        ".page-heading",
        ".result-grid",
        ".camera-evidence-card",
        ".crop-gallery",
    ):
        assert selector in styles
