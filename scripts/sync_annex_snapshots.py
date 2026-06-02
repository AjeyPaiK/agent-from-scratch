#!/usr/bin/env python3
"""Copy cached annex CSVs into the oracle snapshot directory.

CLI entry point that syncs all annex cache files from ``data/.cosing_cache`` into
the pinned snapshot directory used by scoring oracles.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from config.settings import ANNEX_SNAPSHOT_DIR
from data.annex_snapshots import sync_all_snapshots_from_cache


def main() -> None:
    """Parse arguments and sync annex snapshots from the CosIng cache."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=ANNEX_SNAPSHOT_DIR,
        help="Snapshot directory (default: config ANNEX_SNAPSHOT_DIR)",
    )
    args = parser.parse_args()
    hashes = sync_all_snapshots_from_cache(args.dest)
    print(f"Synced {len(hashes)} annex files to {args.dest}")


if __name__ == "__main__":
    main()
