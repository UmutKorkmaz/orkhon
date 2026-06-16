"""Evaluation surface: perplexity, multiple-choice benchmarks, smoke gate, reports."""

from orkhon.eval.benchmarks import TASKS, builtin_examples, load_task
from orkhon.eval.chat_eval import load_golden_set, score_golden_set
from orkhon.eval.loglikelihood import (
    choice_loglikelihoods,
    loglikelihood,
    multiple_choice,
    run_multiple_choice,
)
from orkhon.eval.perplexity import evaluate
from orkhon.eval.report import read_report, write_report
from orkhon.eval.smoke import run_smoke_eval

__all__ = [
    "evaluate",
    "run_smoke_eval",
    "score_golden_set",
    "load_golden_set",
    "write_report",
    "read_report",
    # multiple-choice benchmarks
    "loglikelihood",
    "choice_loglikelihoods",
    "multiple_choice",
    "run_multiple_choice",
    "TASKS",
    "load_task",
    "builtin_examples",
]
