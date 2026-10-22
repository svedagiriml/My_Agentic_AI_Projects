"""Download the knowledge-base PDFs listed in data/sources.yaml into data/pdfs/.

Usage:
    python scripts/download_pdfs.py [--force] [--upload gs://BUCKET/pdfs/]

`--upload` copies the files to GCS so the Cloud Run ingestion job (and the
Eventarc CDC trigger) can pick them up.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import httpx
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources.yaml"
OUT_DIR = ROOT / "data" / "pdfs"
HEADERS = {
    # Several bank sites reject requests without a browser-like UA.
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Accept": "application/pdf,*/*",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--upload", help="optional gs:// prefix to copy files to")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = yaml.safe_load(MANIFEST.read_text())["documents"]
    failures = 0
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as client:
        for d in docs:
            target = OUT_DIR / f"{d['id']}.pdf"
            if target.exists() and not args.force:
                print(f"[skip] {target.name} exists")
                continue
            try:
                r = client.get(d["url"])
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    raise ValueError("response is not a PDF (site may require a browser)")
                target.write_bytes(r.content)
                print(f"[ok]   {target.name}  ({len(r.content)/1024:.0f} KB)")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[fail] {d['id']}: {exc}\n       open {d['url']} in a browser and save it as {target}")

    if args.upload:
        subprocess.run(["gsutil", "-m", "cp", str(OUT_DIR / "*.pdf"), args.upload], check=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
