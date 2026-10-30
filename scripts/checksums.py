"""
SHA-256 of the input rasters.

    python scripts/checksums.py --write     # after placing the published rasters: writes data/checksums.sha256
    python scripts/checksums.py             # verifies data/raw against data/checksums.sha256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hazardfuse.config import REPO_ROOT  # noqa: E402
from hazardfuse.utils import sha256  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--list", default=str(REPO_ROOT / "data" / "checksums.sha256"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    ddir, lst = Path(a.data_dir), Path(a.list)
    if a.write:
        files = sorted(p for p in ddir.glob("*.tif*"))
        lst.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
        print(f"{len(files)} files -> {lst}")
        return 0
    if not lst.exists():
        print(f"{lst} not found; run with --write on the published rasters first.")
        return 1
    ok = True
    for line in lst.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        p = ddir / name
        status = "missing" if not p.exists() else ("OK" if sha256(p) == digest else "DIFFERENT")
        ok &= status == "OK"
        print(f"{status:<9} {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
