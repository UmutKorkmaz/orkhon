"""Training metrics sink: JSONL always-on, optional W&B / TensorBoard fan-out.

A drop-in replacement for :class:`~orkhon.utils.logging.JsonlMetrics` with the same
``log(step, **fields)`` / ``close()`` protocol, but it mirrors every logged record to
optional experiment-tracking backends. Backends are imported lazily so W&B / TB are
strictly optional (``pip install orkhon[monitor]`` style); absence is silent.

Selection (first match wins): an explicit ``backend`` arg → the ``ORKHON_MONITOR``
env var (``wandb``, ``tensorboard``, ``all``) → JSONL only.

JSONL is *always* written (the local-of-record, no-dependency log), so a run is
fully reproducible without any account.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from orkhon.utils.logging import JsonlMetrics


class MetricsSink:
    """Fan-out metrics logger with the :class:`JsonlMetrics` interface."""

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        project: str | None = None,
        run_name: str | None = None,
        config: dict | None = None,
        backend: str | None = None,
    ) -> None:
        self.jsonl = JsonlMetrics(jsonl_path)
        self._wandb = None
        self._tb = None
        self._tb_writer = None

        choice = (backend or os.environ.get("ORKHON_MONITOR", "")).lower()
        want_wandb = choice in ("wandb", "all")
        want_tb = choice in ("tensorboard", "tb", "all")

        if want_wandb:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(project=project or "orkhon", name=run_name,
                           config=config or {}, dir=str(Path(jsonl_path).parent))
            except Exception as e:  # pragma: no cover - optional dep / auth
                self._wandb = None
                print(f"[monitor] wandb disabled ({type(e).__name__}: {e})")

        if want_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(log_dir=str(Path(jsonl_path).parent / "tb"))
                self._tb = True
            except Exception as e:  # pragma: no cover - optional dep
                self._tb_writer = None
                print(f"[monitor] tensorboard disabled ({type(e).__name__}: {e})")

    def log(self, step: int, **fields: Any) -> None:
        """Mirror a record to every enabled backend."""
        self.jsonl.log(step, **fields)
        if self._wandb is not None:
            self._wandb.log({"step": step, **fields})
        if self._tb_writer is not None:
            for k, v in fields.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)

    def close(self) -> None:
        self.jsonl.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:  # pragma: no cover
                pass
        if self._tb_writer is not None:
            self._tb_writer.close()

    def __enter__(self) -> "MetricsSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def build_metrics_sink(
    out_dir: str | Path,
    *,
    project: str | None = None,
    run_name: str | None = None,
    config: dict | None = None,
    backend: str | None = None,
    enabled: bool = True,
) -> MetricsSink | None:
    """Construct a MetricsSink (or None for non-main ranks)."""
    if not enabled:
        return None
    return MetricsSink(
        Path(out_dir) / "metrics.jsonl",
        project=project, run_name=run_name, config=config, backend=backend,
    )
