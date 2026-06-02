"""Tests for annex snapshot sync after cache updates.

Notes
-----
Uses temporary directories and mocked network fetches to verify snapshot
manifest maintenance without touching production cache paths.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from data.annex_snapshots import sync_all_snapshots_from_cache, sync_annex_snapshot_from_cache


def test_sync_annex_snapshot_from_cache_updates_manifest(tmp_path):
    """Copy a single annex CSV into snapshots and update the manifest.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory for cache and snapshot
        files.

    Notes
    -----
    A successful sync should mirror CSV contents and record the file in
    ``manifest.json`` under id ``snapshots``.
    """
    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snapshots"
    cache_dir.mkdir()
    cache_path = cache_dir / "annex_III.csv"
    cache_path.write_text("reference,data\n1,foo\n", encoding="utf-8")

    assert sync_annex_snapshot_from_cache("III", cache_path, snapshot_dir=snapshot_dir) is True

    dest = snapshot_dir / "annex_III.csv"
    assert dest.read_text(encoding="utf-8") == cache_path.read_text(encoding="utf-8")

    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "snapshots"
    assert "annex_III.csv" in manifest["files"]


def test_sync_all_snapshots_from_cache(tmp_path):
    """Sync all annex CSV files from cache and write a combined manifest.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory for cache and snapshot
        files.

    Notes
    -----
    Annex II and III cache files should both produce snapshot copies and
    content hashes.
    """
    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snapshots"
    cache_dir.mkdir()
    for annex in ("II", "III"):
        (cache_dir / f"annex_{annex}.csv").write_text(f"annex {annex}\n", encoding="utf-8")

    hashes = sync_all_snapshots_from_cache(snapshot_dir, source_cache=cache_dir)

    assert len(hashes) == 2
    assert (snapshot_dir / "manifest.json").exists()


def test_fetch_annex_csv_syncs_snapshot_on_refresh(tmp_path):
    """Refresh annex cache from the network and sync the snapshot copy.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory for cache and snapshot
        files.

    Notes
    -----
    When cached CSV is stale, ``fetch_annex_csv`` should download fresh content
    and trigger snapshot sync for the corresponding annex file.
    """
    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snapshots"
    cache_dir.mkdir()

    fresh_csv = '"Reference Number"\n"1","Test Substance"\n' + ("x" * 500)

    with (
        patch("data.cosing_api.CACHE_DIR", cache_dir),
        patch("data.cosing_api._is_fresh", return_value=False),
        patch("data.cosing_api.urllib.request.urlopen") as mock_urlopen,
        patch("data.annex_snapshots.ANNEX_SNAPSHOT_DIR", snapshot_dir),
    ):
        mock_urlopen.return_value.__enter__.return_value.read.return_value = fresh_csv.encode(
            "utf-8"
        )
        from data.cosing_api import fetch_annex_csv

        text = fetch_annex_csv("II")

    assert text == fresh_csv
    assert (snapshot_dir / "annex_II.csv").read_text(encoding="utf-8") == fresh_csv
    assert (snapshot_dir / "manifest.json").exists()
