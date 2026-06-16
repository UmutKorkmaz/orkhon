"""Unit tests for the corpus downloader (no network: urlopen is mocked)."""

from __future__ import annotations

import io
from contextlib import contextmanager

import orkhon.data.download as dl


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _patch_stream(monkeypatch, payload: bytes) -> None:
    @contextmanager
    def fake_urlopen(req, context=None):
        yield _FakeResp(payload)

    # download splits chunks itself; a single BytesIO.read(n) is enough.
    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)


def test_splits_on_endoftext_and_flattens(tmp_path, monkeypatch):
    payload = (
        "Once upon a time\nthere was a cat.<|endoftext|>"
        "The   end\t\tof   another   story.<|endoftext|>"
        "trailing tale with no final marker"
    ).encode("utf-8")
    _patch_stream(monkeypatch, payload)
    out = tmp_path / "stories.txt"
    n = dl.download_tinystories(out, split="valid")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert n == 3
    assert lines[0] == "Once upon a time there was a cat."  # newline flattened
    assert lines[1] == "The end of another story."  # runs of ws collapsed
    assert lines[2] == "trailing tale with no final marker"  # tail flushed


def test_max_stories_caps_output(tmp_path, monkeypatch):
    payload = ("a<|endoftext|>" * 10).encode("utf-8")
    _patch_stream(monkeypatch, payload)
    out = tmp_path / "capped.txt"
    n = dl.download_tinystories(out, split="train", max_stories=3)
    assert n == 3
    assert out.read_text().splitlines() == ["a", "a", "a"]


def test_unknown_split_raises(tmp_path):
    try:
        dl.download_tinystories(tmp_path / "x.txt", split="nope")
    except ValueError as e:
        assert "split" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown split")
