"""CLI smoke tests: the typer app wires up and every --help works.

These tests must stay fast: they only exercise argument parsing / help output
(no real training, no model loading). Heavier end-to-end coverage lives in the
smoke script run by the integrator.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from orkhon.cli import app

runner = CliRunner()

# The top-level command tree the CLI must expose.
_EXPECTED_COMMANDS = ("tokenizer", "data", "train", "eval", "chat", "serve", "export")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _invoke(args: list[str]):
    # Typer renders help through Rich. In narrow CI consoles, Rich can truncate
    # option names to ellipses before tests inspect the output.
    return runner.invoke(
        app,
        args,
        env={"COLUMNS": "160", "NO_COLOR": "1"},
        terminal_width=160,
        color=False,
    )


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_root_help_exits_zero_and_lists_subcommands() -> None:
    result = _invoke(["--help"])
    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    for name in _EXPECTED_COMMANDS:
        assert name in output, f"missing '{name}' in:\n{result.output}"


def test_train_pretrain_help_exits_zero() -> None:
    result = _invoke(["train", "pretrain", "--help"])
    assert result.exit_code == 0, result.output
    # The --config option must be advertised.
    assert "--config" in _plain(result.output)


def test_no_args_shows_help_not_crash() -> None:
    # no_args_is_help=True => help text is shown (typer exits 2, the standard
    # "missing command" code) rather than raising a traceback.
    result = _invoke([])
    assert result.exit_code in (0, 2), result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Usage" in _plain(result.output)


def test_train_subgroup_help_lists_stages() -> None:
    result = _invoke(["train", "--help"])
    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    for stage in ("pretrain", "sft", "dpo"):
        assert stage in output


def test_data_subgroup_help_lists_verbs() -> None:
    result = _invoke(["data", "--help"])
    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    for verb in ("synth", "prepare"):
        assert verb in output


def test_export_hf_help_exits_zero() -> None:
    result = _invoke(["export", "hf", "--help"])
    assert result.exit_code == 0, result.output
    assert "--checkpoint" in _plain(result.output)


def test_tokenizer_train_help_exits_zero() -> None:
    result = _invoke(["tokenizer", "train", "--help"])
    assert result.exit_code == 0, result.output
    assert "--vocab-size" in _plain(result.output)


def test_tokenizer_fertility_help_exits_zero() -> None:
    result = _invoke(["tokenizer", "fertility", "--help"])
    assert result.exit_code == 0, result.output
    assert "--min-turkish-bpt" in _plain(result.output)


def test_eval_help_exits_zero() -> None:
    result = _invoke(["eval", "--help"])
    assert result.exit_code == 0, result.output
    assert "--checkpoint" in _plain(result.output)
