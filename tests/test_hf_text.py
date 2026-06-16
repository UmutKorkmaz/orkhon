"""Unit tests for HF text streaming (no network: datasets.load_dataset mocked)."""

from __future__ import annotations

import datasets

import orkhon.data.hf_text as hf


def _patch_stream(monkeypatch, rows):
    """Replace ``datasets.load_dataset`` with a fake that yields ``rows``.

    The module imports ``load_dataset`` lazily *from* ``datasets`` inside the
    function, so patching ``datasets.load_dataset`` is what gets resolved.
    Records the call kwargs so tests can assert streaming wiring.
    """
    calls = {}

    def fake_load_dataset(dataset, name=None, split="train", streaming=False):
        calls["dataset"] = dataset
        calls["name"] = name
        calls["split"] = split
        calls["streaming"] = streaming
        return iter(rows)

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return calls


def test_writes_one_doc_per_line_and_flattens(tmp_path, monkeypatch):
    rows = [
        {"text": "Hello   world\nthis\t\tis  fine." + "x" * 200},
        {"text": "Second   document\nwith   newlines." + "y" * 200},
    ]
    _patch_stream(monkeypatch, rows)
    out = tmp_path / "train.txt"

    n = hf.stream_hf_text("some/dataset", out, min_chars=10)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert n == 2
    assert len(lines) == 2
    assert lines[0] == "Hello world this is fine." + "x" * 200
    assert lines[1] == "Second document with newlines." + "y" * 200


def test_skips_short_docs(tmp_path, monkeypatch):
    rows = [
        {"text": "too short"},  # below min_chars
        {"text": "k" * 300},  # kept
        {"text": ""},  # falsy, skipped
        {"text": "   \n  "},  # flattens to empty, below threshold
    ]
    _patch_stream(monkeypatch, rows)
    out = tmp_path / "train.txt"

    n = hf.stream_hf_text("some/dataset", out, min_chars=200)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert n == 1
    assert lines == ["k" * 300]


def test_honors_max_docs(tmp_path, monkeypatch):
    rows = [{"text": "doc " + "z" * 300} for _ in range(10)]
    _patch_stream(monkeypatch, rows)
    out = tmp_path / "train.txt"

    n = hf.stream_hf_text("some/dataset", out, min_chars=10, max_docs=3)

    assert n == 3
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_custom_text_field(tmp_path, monkeypatch):
    rows = [{"content": "c" * 300}, {"content": "d" * 300}]
    _patch_stream(monkeypatch, rows)
    out = tmp_path / "train.txt"

    n = hf.stream_hf_text("some/dataset", out, text_field="content", min_chars=10)

    assert n == 2


def test_passes_streaming_and_config(tmp_path, monkeypatch):
    rows = [{"text": "a" * 300}]
    calls = _patch_stream(monkeypatch, rows)
    out = tmp_path / "train.txt"

    hf.stream_hf_text("foo/bar", out, name="cfg-1", split="validation", min_chars=10)

    assert calls["dataset"] == "foo/bar"
    assert calls["name"] == "cfg-1"
    assert calls["split"] == "validation"
    assert calls["streaming"] is True


def test_on_progress_callback(tmp_path, monkeypatch):
    rows = [{"text": "p" * 300} for _ in range(hf._PROGRESS_EVERY)]
    _patch_stream(monkeypatch, rows)
    out = tmp_path / "train.txt"

    seen = []
    n = hf.stream_hf_text(
        "some/dataset", out, min_chars=10, on_progress=seen.append
    )

    assert n == hf._PROGRESS_EVERY
    assert seen == [hf._PROGRESS_EVERY]


def test_download_fineweb_edu_uses_descriptor(tmp_path, monkeypatch):
    rows = [{"text": "w" * 300}, {"text": "v" * 300}]
    calls = _patch_stream(monkeypatch, rows)
    out = tmp_path / "fineweb.txt"

    n = hf.download_fineweb_edu(out, max_docs=2)

    assert n == 2
    assert calls["dataset"] == "HuggingFaceFW/fineweb-edu"
    assert calls["name"] == "sample-10BT"
    assert calls["streaming"] is True


def test_fineweb_edu_descriptor_shape():
    assert hf.FINEWEB_EDU == {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "text_field": "text",
    }
