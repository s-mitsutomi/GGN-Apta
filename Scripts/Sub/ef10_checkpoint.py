"""Opt-in EF@10% checkpoint tracking.

The hit count and enrichment use a stable top-10% ranking. At a fixed validation size
and prevalence, maximizing EF@10% is the same decision as maximizing the
integer top-10% hit count, so checkpoint patience uses that count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class CheckpointNotFound(FileNotFoundError):
    """The requested checkpoint mode has no files in the model directory."""


def percent_k(n: int, pct: float) -> int:
    return max(1, int(round(float(n) * float(pct))))


def ef10_stats(y_true, y_score, pct: float = 0.10):
    """Return ``(hits, enrichment, k)`` for the top ``pct`` of scores.

    Hits and enrichment are NaN when either class is absent. Ties keep the
    earlier row, matching ``np.argsort(..., kind="mergesort")``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = int(y_true.shape[0])
    if n == 0 or y_score.shape[0] != n:
        return float("nan"), float("nan"), 0
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    k = min(percent_k(n, pct), n)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), k
    order = np.argsort(-y_score, kind="mergesort")
    hits = int(np.sum(y_true[order[:k]] == 1))
    prevalence = n_pos / float(n)
    enrichment = (hits / float(k)) / prevalence
    return hits, float(enrichment), k


@dataclass
class HitPolicy:
    """Independent top-10% hit early-stopping policy.

    The first observation is saved and counts as one non-improvement, matching
    the trainer's AUPR counter on epoch 0. A later observation improves only
    when its hit count is at least one above the best saved count. Ties keep
    the earlier epoch. Once patience expires, the policy stays frozen.
    """

    patience: int
    best_hits: float | None = None
    wait: int = 0
    frozen: bool = False
    saved_epoch: int | None = None

    def observe(self, epoch: int, hits) -> bool:
        """Return True when this observation should overwrite the checkpoint."""
        if self.frozen:
            return False
        finite = isinstance(hits, (int, np.integer, float, np.floating)) and hits == hits
        if self.best_hits is None:
            self.best_hits = float(hits) if finite else float("nan")
            self.saved_epoch = int(epoch)
            self.wait = 1
            if self.wait >= self.patience:
                self.frozen = True
            return True
        if finite and float(hits) >= float(self.best_hits) + 1.0:
            self.best_hits = float(hits)
            self.wait = 0
            self.saved_epoch = int(epoch)
            return True
        self.wait += 1
        if self.wait >= self.patience:
            self.frozen = True
        return False


def aupr_checkpoint_should_save(epoch: int, metric_improved: bool, aupr_frozen: bool) -> bool:
    """Same rule as the trainer: epoch 0 or a real improvement, unless frozen."""
    return ((int(epoch) == 0) or bool(metric_improved)) and (not aupr_frozen)


def trajectory_should_stop(
    track_ef10: bool,
    es: int,
    patience: int,
    aupr_frozen: bool,
    ef_frozen: bool,
) -> bool:
    """Stop immediately on the AUPR counter when EF tracking is off.

    When tracking is on, stop only after both policies have frozen.
    """
    if not track_ef10:
        return int(es) >= int(patience)
    return bool(aupr_frozen and ef_frozen)


def fold_number(path: Path) -> int:
    return int(path.stem.split("fold")[-1].lstrip("_").split("_")[0])


def _is_es(path: Path) -> bool:
    return path.stem.endswith("_es")


def _is_es_ef10(path: Path) -> bool:
    return path.stem.endswith("_es_ef10")


def resolve_fold_checkpoints(model_dir: Path, mode: str) -> list[Path]:
    """Return one checkpoint per fold for ``es``, ``es_ef10``, or ``final``.

    ``es`` is only ``*_es.pt``. ``es_ef10`` is only ``*_es_ef10.pt``. ``final``
    excludes both. A fold with two files for the requested mode is an error,
    so a mixed set cannot be loaded as one ensemble.
    """
    if mode not in {"es", "es_ef10", "final"}:
        raise ValueError(f"checkpoint_mode must be es, es_ef10, or final, got {mode!r}")
    model_dir = Path(model_dir)
    candidates = sorted(model_dir.glob("model_*_fold*.pt"))
    if mode == "es":
        chosen = [path for path in candidates if _is_es(path)]
    elif mode == "es_ef10":
        chosen = [path for path in candidates if _is_es_ef10(path)]
    else:
        chosen = [
            path
            for path in candidates
            if not _is_es(path) and not _is_es_ef10(path)
        ]
    by_fold: dict[int, Path] = {}
    for path in chosen:
        fold = fold_number(path)
        if fold in by_fold:
            raise ValueError(
                f"multiple {mode} checkpoints for fold {fold}: "
                f"{by_fold[fold].name} and {path.name}"
            )
        by_fold[fold] = path
    if not by_fold:
        raise CheckpointNotFound(f"no {mode} checkpoints in {model_dir}")
    return [by_fold[fold] for fold in sorted(by_fold)]
