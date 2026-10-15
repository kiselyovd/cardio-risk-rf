"""The source fetcher verifies bytes and reuses only the expected original CSV."""

import hashlib
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fetch_data", ROOT / "scripts/fetch_data.py")
assert SPEC and SPEC.loader
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


def test_changed_archive_is_rejected() -> None:
    with pytest.raises(ValueError, match="archive changed"):
        fetch.verified_csv(b"changed")


def test_fetch_is_verified_and_idempotent(tmp_path, monkeypatch) -> None:
    csv = b"fixture,only\n1,2\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("cardio_train.csv", csv)
    payload = buffer.getvalue()
    monkeypatch.setattr(fetch, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch, "CSV_SHA256", hashlib.sha256(csv).hexdigest())

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    out = tmp_path / "cardio.csv"
    monkeypatch.setattr(sys, "argv", ["fetch_data", "--out", str(out)])
    fetch.main()
    assert out.read_bytes() == csv

    def forbidden_network(*args, **kwargs):
        raise AssertionError("Verified existing data must not be downloaded again")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", forbidden_network)
    fetch.main()
    out.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing data preserved"):
        fetch.main()
    assert out.read_bytes() == b"changed"
