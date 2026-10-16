"""Offline regression tests for the upstream propofol CSV downloader."""
from io import BytesIO
from unittest.mock import patch

import pytest

from download_data import FILES, download_data


def test_download_and_verify_cached(tmp_path):
    with patch("download_data.urlopen", side_effect=lambda url, timeout: BytesIO(url.encode())) as fetch:
        download_data(tmp_path)
        assert fetch.call_count == len(FILES)
        assert all((tmp_path / name).stat().st_size > 0 for name in FILES)
        download_data(tmp_path)
        assert fetch.call_count == 2 * len(FILES)
        assert not list(tmp_path.glob("*.part"))


def test_cleanup_on_empty_response(tmp_path):
    with patch("download_data.urlopen", return_value=BytesIO(b"")):
        with pytest.raises(ValueError, match="Empty upstream file"):
            download_data(tmp_path)
    assert not list(tmp_path.iterdir())


def test_cleanup_on_interrupted_download(tmp_path):
    class Broken(BytesIO):
        def read(self, n=-1):
            raise OSError("Interrupted")

    with patch("download_data.urlopen", return_value=Broken(b"data")):
        with pytest.raises(OSError, match="Interrupted"):
            download_data(tmp_path)
    assert not list(tmp_path.iterdir())


def test_reject_empty_cached_file(tmp_path):
    cached = tmp_path / FILES[0]
    cached.touch()
    with patch("download_data.urlopen", return_value=BytesIO(b"valid remote data")) as fetch:
        with pytest.raises(ValueError, match="Cached upstream file differs"):
            download_data(tmp_path)
    assert fetch.call_count == 1
    assert cached.stat().st_size == 0

def test_reject_modified_cached_file(tmp_path):
    with patch("download_data.urlopen", side_effect=lambda url, timeout: BytesIO(url.encode())):
        download_data(tmp_path)
        cached = tmp_path / FILES[0]
        cached.write_bytes(b"modified")
        with pytest.raises(ValueError, match="Cached upstream file differs"):
            download_data(tmp_path)
    assert cached.read_bytes() == b"modified"
    assert not list(tmp_path.glob("*.part"))
