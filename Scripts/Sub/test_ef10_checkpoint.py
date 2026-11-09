#!/usr/bin/env python3
"""Checks for opt-in EF@10% checkpoint tracking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ef10_checkpoint import (
    CheckpointNotFound,
    HitPolicy,
    aupr_checkpoint_should_save,
    resolve_fold_checkpoints,
    trajectory_should_stop,
)


class PolicyTests(unittest.TestCase):
    def test_freeze_does_not_revive(self):
        policy = HitPolicy(patience=5)
        saved = []
        for epoch, hits in enumerate([1, 1, 1, 1, 1, 9]):
            if policy.observe(epoch, hits):
                saved.append(epoch)
        self.assertEqual(saved, [0])
        self.assertTrue(policy.frozen)
        self.assertEqual(policy.saved_epoch, 0)
        self.assertFalse(policy.observe(6, 20))
        self.assertEqual(policy.saved_epoch, 0)

    def test_one_extra_hit_resets_and_ties_keep_earlier(self):
        policy = HitPolicy(patience=5)
        self.assertTrue(policy.observe(0, 1))
        self.assertTrue(policy.observe(1, 2))
        self.assertFalse(policy.observe(2, 2))
        self.assertEqual(policy.saved_epoch, 1)
        self.assertEqual(policy.best_hits, 2.0)
        self.assertFalse(policy.frozen)

    def test_disabled_path_stops_on_aupr_counter(self):
        self.assertTrue(trajectory_should_stop(False, 5, 5, False, False))
        self.assertFalse(trajectory_should_stop(True, 5, 5, True, False))
        self.assertTrue(trajectory_should_stop(True, 5, 5, True, True))
        self.assertTrue(aupr_checkpoint_should_save(0, False, False))
        self.assertFalse(aupr_checkpoint_should_save(3, True, True))

    def test_virtual_policies_stay_frozen(self):
        auprs = [0.10, 0.10, 0.10, 0.10, 0.10, 0.90]
        hits = [1, 1, 1, 1, 1, 8]
        aupr_saved, ef_saved, stopped = simulate_dual(auprs, hits, patience=5)
        self.assertEqual(aupr_saved, 0)
        self.assertEqual(ef_saved, 0)
        self.assertEqual(stopped, 4)


class CheckpointResolutionTests(unittest.TestCase):
    def test_modes_do_not_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fold in (1, 2):
                (root / f"model_affinity_x_fold{fold}.pt").write_bytes(b"final")
                (root / f"model_affinity_x_fold{fold}_es.pt").write_bytes(b"es")
                (root / f"model_affinity_x_fold{fold}_es_ef10.pt").write_bytes(b"ef")
            es = resolve_fold_checkpoints(root, "es")
            ef = resolve_fold_checkpoints(root, "es_ef10")
            final = resolve_fold_checkpoints(root, "final")
            self.assertTrue(all(path.stem.endswith("_es") for path in es))
            self.assertFalse(any(path.stem.endswith("_es_ef10") for path in es))
            self.assertTrue(all(path.stem.endswith("_es_ef10") for path in ef))
            self.assertTrue(all(path.name.endswith(f"_fold{i}.pt") for i, path in enumerate(final, start=1)))
            (root / "model_affinity_x_fold1_es_copy.pt").write_bytes(b"dup")
            (root / "model_affinity_extra_fold1_es.pt").write_bytes(b"dup")
            with self.assertRaises(ValueError):
                resolve_fold_checkpoints(root, "es")

    def test_es_fallback_is_only_for_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model_affinity_x_fold1.pt").write_bytes(b"final")
            with self.assertRaises(CheckpointNotFound):
                resolve_fold_checkpoints(root, "es")
            self.assertEqual(
                resolve_fold_checkpoints(root, "final")[0].name,
                "model_affinity_x_fold1.pt",
            )


def simulate_dual(auprs, hits, patience):
    es = 0
    best = None
    aupr_frozen = False
    aupr_saved = None
    ef_policy = HitPolicy(patience=patience)
    ef_saved = None
    stopped = None
    min_delta = 0.001
    for epoch, (aupr, hit) in enumerate(zip(auprs, hits)):
        metric_improved = False
        if not aupr_frozen:
            if best is None:
                best = aupr
            elif aupr > best + min_delta:
                best = aupr
                metric_improved = True
            if aupr > best:
                best = aupr
            if metric_improved:
                es = 0
            else:
                es += 1
        if aupr_checkpoint_should_save(epoch, metric_improved, aupr_frozen):
            aupr_saved = epoch
        if ef_policy.observe(epoch, hit):
            ef_saved = epoch
        if (not aupr_frozen) and es >= patience:
            aupr_frozen = True
        if trajectory_should_stop(True, es, patience, aupr_frozen, ef_policy.frozen):
            stopped = epoch
            break
    return aupr_saved, ef_saved, stopped


if __name__ == "__main__":
    unittest.main()
