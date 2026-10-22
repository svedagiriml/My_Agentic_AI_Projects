"""Tiny storage abstraction: local paths or gs:// URIs (Google Cloud Storage)."""
from __future__ import annotations

import pathlib
from functools import lru_cache


def _is_gcs(uri: str) -> bool:
    return uri.startswith("gs://")


def _split(uri: str) -> tuple[str, str]:
    bucket, _, blob = uri[5:].partition("/")
    return bucket, blob


@lru_cache
def _gcs_client():  # type: ignore[no-untyped-def]
    from google.cloud import storage

    return storage.Client()


def read_bytes(uri: str) -> bytes | None:
    if _is_gcs(uri):
        b, name = _split(uri)
        blob = _gcs_client().bucket(b).blob(name)
        return blob.download_as_bytes() if blob.exists() else None
    p = pathlib.Path(uri)
    return p.read_bytes() if p.exists() else None


def read_text(uri: str) -> str | None:
    data = read_bytes(uri)
    return data.decode("utf-8") if data is not None else None


def write_text(uri: str, text: str) -> None:
    if _is_gcs(uri):
        b, name = _split(uri)
        _gcs_client().bucket(b).blob(name).upload_from_string(text, content_type="text/plain; charset=utf-8")
        return
    p = pathlib.Path(uri)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)  # atomic on POSIX


def list_files(prefix: str, suffix: str = ".pdf") -> list[str]:
    if _is_gcs(prefix):
        b, name = _split(prefix.rstrip("/") + "/")
        return sorted(f"gs://{b}/{bl.name}" for bl in _gcs_client().list_blobs(b, prefix=name)
                      if bl.name.lower().endswith(suffix))
    p = pathlib.Path(prefix)
    return sorted(str(f) for f in p.glob(f"*{suffix}")) if p.exists() else []


def basename(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]
