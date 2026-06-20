"""Tests for unified assistant SFT data generation."""

from __future__ import annotations

from scripts.make_unified_assistant_data import make_split
from orkhon.data.old_turkic import rune_to_latin


def test_unified_split_has_required_categories():
    rows = make_split("test", seed=123, n_kokturk=20, n_capability=8, n_qa=12)
    categories = {row["category"] for row in rows}

    assert "kokturk_transliteration" in categories
    assert any(c.startswith("capability") for c in categories)
    assert "old_turkic_scope" in categories or "old_turkic_scope_tr" in categories
    assert any(c.startswith("math") for c in categories)


def test_kokturk_rows_are_correct_by_construction():
    rows = make_split("test", seed=456, n_kokturk=40, n_capability=0, n_qa=0)
    for row in rows:
        assert row["category"] == "kokturk_transliteration"
        assert row["expected"] == rune_to_latin(row["runic"])
        assert row["messages"][1]["content"] == row["expected"]


def test_train_val_test_seeds_create_disjoint_kokturk_samples():
    train = make_split("train", seed=10, n_kokturk=50, n_capability=0, n_qa=0)
    val = make_split("val", seed=11, n_kokturk=50, n_capability=0, n_qa=0)
    test = make_split("test", seed=12, n_kokturk=50, n_capability=0, n_qa=0)

    train_runic = {row["runic"] for row in train}
    val_runic = {row["runic"] for row in val}
    test_runic = {row["runic"] for row in test}

    assert train_runic.isdisjoint(val_runic)
    assert train_runic.isdisjoint(test_runic)
    assert val_runic.isdisjoint(test_runic)
