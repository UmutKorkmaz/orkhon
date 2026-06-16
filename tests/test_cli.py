"""CLI smoke tests: the typer app wires up and every --help works.

These tests must stay fast: they only exercise argument parsing / help output
(no real training, no model loading). Heavier end-to-end coverage lives in the
smoke script run by the integrator.
"""

from __future__ import annotations

from typer.testing import CliRunner

from orkhon.cli import app

runner = CliRunner()

# The top-level command tree the CLI must expose.
_EXPECTED_COMMANDS = ("tokenizer", "data", "train", "eval", "chat", "serve", "export")


def test_root_help_exits_zero_and_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in _EXPECTED_COMMANDS:
        assert name in result.output, f"missing '{name}' in:\n{result.output}"


def test_train_pretrain_help_exits_zero() -> None:
    result = runner.invoke(app, ["train", "pretrain", "--help"])
    assert result.exit_code == 0, result.output
    # The --config option must be advertised.
    assert "--config" in result.output


def test_no_args_shows_help_not_crash() -> None:
    # no_args_is_help=True => help text is shown (typer exits 2, the standard
    # "missing command" code) rather than raising a traceback.
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2), result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Usage" in result.output


def test_train_subgroup_help_lists_stages() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0, result.output
    for stage in ("pretrain", "sft", "dpo"):
        assert stage in result.output


def test_data_subgroup_help_lists_verbs() -> None:
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0, result.output
    for verb in ("synth", "prepare"):
        assert verb in result.output


def test_export_hf_help_exits_zero() -> None:
    result = runner.invoke(app, ["export", "hf", "--help"])
    assert result.exit_code == 0, result.output
    assert "--checkpoint" in result.output


def test_tokenizer_train_help_exits_zero() -> None:
    result = runner.invoke(app, ["tokenizer", "train", "--help"])
    assert result.exit_code == 0, result.output
    assert "--vocab-size" in result.output


def test_tokenizer_fertility_help_exits_zero() -> None:
    result = runner.invoke(app, ["tokenizer", "fertility", "--help"])
    assert result.exit_code == 0, result.output
    assert "--min-turkish-bpt" in result.output


def test_eval_help_exits_zero() -> None:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0, result.output
    assert "--checkpoint" in result.output
