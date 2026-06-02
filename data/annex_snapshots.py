"""Keep oracle annex snapshots in sync with the live CosIng cache."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, cast

from config.settings import ANNEX_SNAPSHOT_DIR, PROJECT_ROOT

ANNEXES = ("II", "III", "IV", "V", "VI")
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cosing_cache"


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file's contents.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(snapshot_dir: Path) -> dict[str, Any]:
    """Load the snapshot manifest, or return a default empty structure.

    Parameters
    ----------
    snapshot_dir : pathlib.Path
        Directory containing ``manifest.json``.

    Returns
    -------
    dict
        Manifest with ``id`` and ``files`` keys mapping filenames to digests.
    """
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        return {"id": snapshot_dir.name, "files": {}}
    return cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))


def _write_manifest(snapshot_dir: Path, files: dict[str, str]) -> None:
    """Write ``manifest.json`` with the given file digest map.

    Parameters
    ----------
    snapshot_dir : pathlib.Path
        Snapshot root directory.
    files : dict[str, str]
        Mapping of snapshot filenames to SHA-256 hex digests.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"id": snapshot_dir.name, "files": files}
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def sync_annex_snapshot_from_cache(
    annex: str,
    cache_path: Path,
    *,
    snapshot_dir: Path | None = None,
) -> bool:
    """Copy one refreshed cache file into the oracle snapshot dir and update manifest.

    Parameters
    ----------
    annex : str
        Annex identifier (``II``–``VI``).
    cache_path : pathlib.Path
        Source cached CSV from the live CosIng client.
    snapshot_dir : pathlib.Path or None, optional
        Destination snapshot root (default: ``ANNEX_SNAPSHOT_DIR``).

    Returns
    -------
    bool
        ``True`` if the cache file existed and was copied; ``False`` otherwise.
    """
    if not cache_path.exists():
        return False

    dest_dir = snapshot_dir or ANNEX_SNAPSHOT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"annex_{annex}.csv"
    shutil.copy2(cache_path, dest)

    manifest = _read_manifest(dest_dir)
    manifest["files"][dest.name] = _sha256(dest)
    _write_manifest(dest_dir, manifest["files"])
    return True


def sync_all_snapshots_from_cache(
    snapshot_dir: Path | None = None,
    *,
    source_cache: Path | None = None,
) -> dict[str, str]:
    """Copy all annex cache files into the oracle snapshot dir.

    Parameters
    ----------
    snapshot_dir : pathlib.Path or None, optional
        Destination snapshot root (default: ``ANNEX_SNAPSHOT_DIR``).
    source_cache : pathlib.Path or None, optional
        Directory containing ``annex_*.csv`` cache files (default: ``DEFAULT_CACHE_DIR``).

    Returns
    -------
    dict[str, str]
        Mapping of copied snapshot filenames to their SHA-256 digests.
    """
    cache_root = source_cache or DEFAULT_CACHE_DIR
    dest_dir = snapshot_dir or ANNEX_SNAPSHOT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    for annex in ANNEXES:
        cache_path = cache_root / f"annex_{annex}.csv"
        if not cache_path.exists():
            continue
        sync_annex_snapshot_from_cache(annex, cache_path, snapshot_dir=dest_dir)
        dest = dest_dir / f"annex_{annex}.csv"
        hashes[dest.name] = _sha256(dest)

    return hashes
