#!/usr/bin/env python3
"""Grouped cross-validation helpers for homology-aware GGN-Apta folds.

The independent train/test split already assigns whole MMseqs2 clusters.
Within train, fragment-level StratifiedKFold can put
the same UniProt accession or the same MMseqs2 cluster into different folds.

This module assigns folds at the cluster (default) or UniProt level so that
every fragment of every accession in a group stays in one fold.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5
KFOLD_RANDOM_STATE = 42

DEFAULT_TRAIN_FASTA = (
    "Fasta/AFDB_split_by_homology/seed42/"
    "AFDB_concatenated_train80pct_homology_id30.fasta.gz"
)
DEFAULT_TEST_FASTA = (
    "Fasta/AFDB_split_by_homology/seed42/"
    "AFDB_concatenated_test20pct_homology_id30.fasta.gz"
)
DEFAULT_APTAMER_LIST = "Lists/aptamer_target_proteins_human_2024-08-01.txt"
DEFAULT_CLUSTER_MEMBERS = "Tables/Homology_split/mmseqs_id30/cluster_members.tsv"
DEFAULT_PROTEIN_MEMBERSHIP = (
    "Tables/Homology_split/seed42/protein_split_membership_id30.tsv"
)
DEFAULT_MANIFEST = "Tables/Homology_split/uniprot_full_sequence_manifest.tsv"
DEFAULT_CURRENT_MEMBERSHIP = (
    "Results/Protein_predictions/Homology_id30/fold_membership_seed42/"
    "train80_fold_membership.tsv"
)
DEFAULT_OUTPUT_DIR = (
    "Results/Protein_predictions/Homology_id30/"
    "fold_membership_seed42_grouped_cluster"
)

LONG_MEMBERSHIP_FIELDS = [
    "protein_id",
    "uniprot",
    "cluster_id",
    "group_id",
    "fold",
    "split",
    "dataset_index",
    "label",
    "n_fragments_uniprot",
]


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def normalize_to_uniprot(name: str) -> str:
    name = name.split("/")[-1]
    for ext in (".pdb.gz", ".cif.gz", ".fasta.gz", ".pdb", ".cif", ".fasta"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    match = re.match(r"AF-([A-Z0-9]+)-F\d+", name)
    if match:
        return match.group(1)
    return name


def extract_fragment_number(name: str) -> Optional[int]:
    name = name.split("/")[-1]
    for ext in (".pdb.gz", ".cif.gz", ".fasta.gz", ".pdb", ".cif", ".fasta"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    match = re.match(r"AF-[A-Z0-9]+-F(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def read_fasta_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line[1:].split()[0].strip())
    return ids


def load_positive_uniprots(path: Path) -> set[str]:
    positives: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if name and not name.startswith("#") and name != "TargetID":
                positives.add(name)
    return positives


def filter_multi_fragment_positives(
    positive_set: set[str],
    dataset_ids: Iterable[str],
) -> tuple[set[str], set[str]]:
    uniprot_to_fragments: dict[str, set[int]] = {}
    for protein_id in dataset_ids:
        uniprot = normalize_to_uniprot(protein_id)
        frag = extract_fragment_number(protein_id)
        uniprot_to_fragments.setdefault(uniprot, set())
        if frag is not None:
            uniprot_to_fragments[uniprot].add(frag)
    multi = {
        uniprot
        for uniprot in positive_set
        if len(uniprot_to_fragments.get(uniprot, set())) > 1
    }
    return positive_set - multi, multi


def labels_for_ids(protein_ids: Iterable[str], positive_set: set[str]) -> np.ndarray:
    return np.asarray(
        [1 if normalize_to_uniprot(pid) in positive_set else 0 for pid in protein_ids],
        dtype=np.int64,
    )


def load_cluster_members(path: Path) -> dict[str, str]:
    """Map UniProt accession -> MMseqs2 cluster_id."""
    uniprot_to_cluster: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cluster_id", "member"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"cluster members TSV missing {required}: {path}")
        for row in reader:
            member = row["member"].strip()
            cluster_id = row["cluster_id"].strip()
            if not member or not cluster_id:
                continue
            prev = uniprot_to_cluster.get(member)
            if prev is not None and prev != cluster_id:
                raise ValueError(
                    f"UniProt {member} mapped to multiple clusters: {prev} vs {cluster_id}"
                )
            uniprot_to_cluster[member] = cluster_id
    return uniprot_to_cluster


def load_protein_split_membership(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            uid = row["uniprot_id"].strip()
            rows[uid] = row
    return rows


def load_manifest_n_fragments(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            counts[row["uniprot_id"].strip()] = int(row["n_fragments"])
    return counts


def load_long_membership(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Empty membership TSV: {path}")
        required = {"protein_id", "fold", "split"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Membership TSV missing {required}: {path}")
        return [dict(row) for row in reader]


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def fragment_annotations(
    protein_ids: list[str],
    uniprot_to_cluster: dict[str, str],
    protein_split: dict[str, dict[str, str]],
    labels: np.ndarray,
    group_level: str,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    if group_level not in {"cluster", "uniprot"}:
        raise ValueError(f"group_level must be cluster or uniprot, got {group_level}")
    rows: list[dict[str, object]] = []
    missing_cluster: list[str] = []
    split_mismatch: list[str] = []
    uniprot_counts: dict[str, int] = defaultdict(int)
    for pid in protein_ids:
        uniprot_counts[normalize_to_uniprot(pid)] += 1

    for idx, pid in enumerate(protein_ids):
        uniprot = normalize_to_uniprot(pid)
        cluster_id = uniprot_to_cluster.get(uniprot)
        if not cluster_id:
            cluster_id = uniprot
            missing_cluster.append(uniprot)
        split_row = protein_split.get(uniprot)
        split_name = split_row["split"] if split_row else "unknown"
        if split_name not in {"train", "unknown"}:
            split_mismatch.append(uniprot)
        group_id = cluster_id if group_level == "cluster" else uniprot
        rows.append(
            {
                "protein_id": pid,
                "uniprot": uniprot,
                "cluster_id": cluster_id,
                "group_id": group_id,
                "dataset_index": idx,
                "label": int(labels[idx]),
                "n_fragments_uniprot": int(uniprot_counts[uniprot]),
                "protein_split": split_name,
            }
        )
    return rows, sorted(set(missing_cluster)), sorted(set(split_mismatch))


def assign_groups_to_folds(
    group_ids: list[str],
    group_labels: np.ndarray,
    n_folds: int,
    random_state: int,
) -> dict[str, int]:
    """StratifiedKFold on groups. Returns group_id -> val fold (1-based)."""
    if len(group_ids) != int(group_labels.size):
        raise ValueError("group_ids and group_labels length mismatch")
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("Duplicate group IDs")
    n_pos = int(group_labels.sum())
    n_neg = int(group_labels.size - n_pos)
    if n_pos < n_folds:
        raise ValueError(
            f"Only {n_pos} positive groups for {n_folds} folds; "
            "cannot stratify at the group level."
        )
    if n_neg < n_folds:
        raise ValueError(
            f"Only {n_neg} negative groups for {n_folds} folds; "
            "cannot stratify at the group level."
        )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_of: dict[str, int] = {}
    indices = np.arange(len(group_ids))
    for fold_idx, (_, val_idx) in enumerate(skf.split(indices, group_labels), start=1):
        for i in val_idx.tolist():
            fold_of[group_ids[i]] = fold_idx
    if len(fold_of) != len(group_ids):
        raise RuntimeError("Fold assignment missed one or more groups")
    return fold_of


def build_grouped_folds(
    protein_ids: list[str],
    labels: np.ndarray,
    uniprot_to_cluster: dict[str, str],
    protein_split: dict[str, dict[str, str]],
    group_level: str = "cluster",
    n_folds: int = N_FOLDS,
    random_state: int = KFOLD_RANDOM_STATE,
) -> dict[str, object]:
    annotated, missing_cluster, split_mismatch = fragment_annotations(
        protein_ids,
        uniprot_to_cluster,
        protein_split,
        labels,
        group_level,
    )
    groups: dict[str, dict[str, object]] = {}
    for row in annotated:
        rec = groups.setdefault(
            str(row["group_id"]),
            {
                "group_id": row["group_id"],
                "cluster_ids": set(),
                "uniprots": set(),
                "dataset_indices": [],
                "n_positives": 0,
            },
        )
        rec["cluster_ids"].add(row["cluster_id"])
        rec["uniprots"].add(row["uniprot"])
        rec["dataset_indices"].append(int(row["dataset_index"]))
        rec["n_positives"] += int(row["label"])

    group_ids = sorted(groups)
    group_labels = np.asarray(
        [1 if groups[gid]["n_positives"] > 0 else 0 for gid in group_ids],
        dtype=np.int64,
    )
    fold_of_group = assign_groups_to_folds(
        group_ids, group_labels, n_folds=n_folds, random_state=random_state
    )
    val_fold = np.zeros(len(protein_ids), dtype=np.int64)
    for gid, rec in groups.items():
        fold_num = fold_of_group[gid]
        rec["val_fold"] = fold_num
        for idx in rec["dataset_indices"]:
            val_fold[idx] = fold_num

    long_rows: list[dict[str, object]] = []
    fold_split_ids: dict[tuple[int, str], list[str]] = {}
    for fold_num in range(1, n_folds + 1):
        val_idx = np.where(val_fold == fold_num)[0]
        train_idx = np.where(val_fold != fold_num)[0]
        fold_split_ids[(fold_num, "train")] = [protein_ids[i] for i in train_idx.tolist()]
        fold_split_ids[(fold_num, "val")] = [protein_ids[i] for i in val_idx.tolist()]
        for split_name, idx_array in (("train", train_idx), ("val", val_idx)):
            for dataset_index in idx_array.tolist():
                src = annotated[dataset_index]
                long_rows.append(
                    {
                        "protein_id": src["protein_id"],
                        "uniprot": src["uniprot"],
                        "cluster_id": src["cluster_id"],
                        "group_id": src["group_id"],
                        "fold": fold_num,
                        "split": split_name,
                        "dataset_index": int(dataset_index),
                        "label": int(src["label"]),
                        "n_fragments_uniprot": int(src["n_fragments_uniprot"]),
                    }
                )

    cluster_rows = []
    for gid in group_ids:
        rec = groups[gid]
        cluster_rows.append(
            {
                "group_id": gid,
                "val_fold": rec["val_fold"],
                "is_positive_group": int(rec["n_positives"] > 0),
                "n_positives": int(rec["n_positives"]),
                "n_fragments": len(rec["dataset_indices"]),
                "n_uniprots": len(rec["uniprots"]),
                "n_clusters": len(rec["cluster_ids"]),
                "cluster_ids": ";".join(sorted(rec["cluster_ids"])),
                "uniprot_ids": ";".join(sorted(rec["uniprots"])),
            }
        )

    return {
        "long_rows": long_rows,
        "cluster_rows": cluster_rows,
        "fold_split_ids": fold_split_ids,
        "val_fold": val_fold,
        "annotated": annotated,
        "n_groups": len(group_ids),
        "n_positive_groups": int(group_labels.sum()),
        "missing_cluster_uniprots": missing_cluster,
        "split_mismatch_uniprots": split_mismatch,
        "fold_of_group": fold_of_group,
    }


def verify_membership_integrity(
    long_rows: list[dict],
    protein_ids: list[str],
    n_folds: int,
) -> dict[str, object]:
    """Confirm no UniProt accession or MMseqs2 cluster crosses folds."""
    if not long_rows:
        raise ValueError("No membership rows to verify")

    by_fold_split: dict[tuple[int, str], list[dict]] = defaultdict(list)
    val_fold_by_pid: dict[str, set[int]] = defaultdict(set)
    val_fold_by_uniprot: dict[str, set[int]] = defaultdict(set)
    val_fold_by_cluster: dict[str, set[int]] = defaultdict(set)
    for row in long_rows:
        fold_num = int(float(row["fold"]))
        split_name = str(row["split"])
        by_fold_split[(fold_num, split_name)].append(row)
        if split_name == "val":
            val_fold_by_pid[str(row["protein_id"])].add(fold_num)
            val_fold_by_uniprot[str(row.get("uniprot") or normalize_to_uniprot(row["protein_id"]))].add(
                fold_num
            )
            cluster_id = str(row.get("cluster_id") or "")
            if cluster_id:
                val_fold_by_cluster[cluster_id].add(fold_num)

    crossing_accessions = [
        {"uniprot": uid, "val_folds": ",".join(str(f) for f in sorted(folds))}
        for uid, folds in sorted(val_fold_by_uniprot.items())
        if len(folds) > 1
    ]
    crossing_clusters = [
        {"cluster_id": cid, "val_folds": ",".join(str(f) for f in sorted(folds))}
        for cid, folds in sorted(val_fold_by_cluster.items())
        if len(folds) > 1
    ]
    crossing_fragments = [
        {"protein_id": pid, "val_folds": ",".join(str(f) for f in sorted(folds))}
        for pid, folds in sorted(val_fold_by_pid.items())
        if len(folds) > 1
    ]

    same_fold_accession_leaks = []
    same_fold_cluster_leaks = []
    per_fold = []
    expected_ids = set(protein_ids)
    for fold_num in range(1, n_folds + 1):
        train_rows = by_fold_split.get((fold_num, "train"), [])
        val_rows = by_fold_split.get((fold_num, "val"), [])
        train_ids = [str(r["protein_id"]) for r in train_rows]
        val_ids = [str(r["protein_id"]) for r in val_rows]
        train_set = set(train_ids)
        val_set = set(val_ids)
        covered = train_set | val_set
        train_uniprot = {str(r.get("uniprot") or normalize_to_uniprot(r["protein_id"])) for r in train_rows}
        val_uniprot = {str(r.get("uniprot") or normalize_to_uniprot(r["protein_id"])) for r in val_rows}
        train_cluster = {str(r.get("cluster_id") or "") for r in train_rows if r.get("cluster_id")}
        val_cluster = {str(r.get("cluster_id") or "") for r in val_rows if r.get("cluster_id")}
        acc_overlap = sorted(train_uniprot & val_uniprot)
        clu_overlap = sorted(train_cluster & val_cluster)
        for uid in acc_overlap:
            same_fold_accession_leaks.append({"fold": fold_num, "uniprot": uid})
        for cid in clu_overlap:
            same_fold_cluster_leaks.append({"fold": fold_num, "cluster_id": cid})
        per_fold.append(
            {
                "fold": fold_num,
                "n_train_fragments": len(train_ids),
                "n_val_fragments": len(val_ids),
                "n_train_uniprot": len(train_uniprot),
                "n_val_uniprot": len(val_uniprot),
                "n_train_clusters": len(train_cluster),
                "n_val_clusters": len(val_cluster),
                "n_pos_train": sum(int(float(r.get("label", 0))) for r in train_rows),
                "n_pos_val": sum(int(float(r.get("label", 0))) for r in val_rows),
                "n_accession_train_val_overlap": len(acc_overlap),
                "n_cluster_train_val_overlap": len(clu_overlap),
                "covers_all_train_fragments": covered == expected_ids,
                "fragment_train_val_overlap": len(train_set & val_set),
            }
        )

    ok = (
        not crossing_accessions
        and not crossing_clusters
        and not crossing_fragments
        and not same_fold_accession_leaks
        and not same_fold_cluster_leaks
        and all(item["covers_all_train_fragments"] for item in per_fold)
        and all(item["fragment_train_val_overlap"] == 0 for item in per_fold)
        and all(item["n_val_fragments"] > 0 and item["n_train_fragments"] > 0 for item in per_fold)
        and all(item["n_pos_val"] > 0 for item in per_fold)
    )
    return {
        "ok": ok,
        "n_folds": n_folds,
        "n_long_rows": len(long_rows),
        "n_train_fragments": len(protein_ids),
        "n_accessions_crossing_folds": len(crossing_accessions),
        "n_clusters_crossing_folds": len(crossing_clusters),
        "n_fragments_crossing_folds": len(crossing_fragments),
        "n_accession_train_val_same_fold": len(same_fold_accession_leaks),
        "n_cluster_train_val_same_fold": len(same_fold_cluster_leaks),
        "crossing_accessions": crossing_accessions,
        "crossing_clusters": crossing_clusters,
        "same_fold_accession_leaks": same_fold_accession_leaks,
        "same_fold_cluster_leaks": same_fold_cluster_leaks,
        "per_fold": per_fold,
    }


def annotate_membership_with_groups(
    long_rows: list[dict],
    uniprot_to_cluster: dict[str, str],
) -> list[dict[str, object]]:
    annotated = []
    for row in long_rows:
        pid = str(row["protein_id"])
        uniprot = str(row.get("uniprot") or normalize_to_uniprot(pid))
        cluster_id = str(row.get("cluster_id") or uniprot_to_cluster.get(uniprot, uniprot))
        out = dict(row)
        out["uniprot"] = uniprot
        out["cluster_id"] = cluster_id
        annotated.append(out)
    return annotated


def multi_fragment_uniprot_report(
    protein_ids: list[str],
    labels: np.ndarray,
    positive_set: set[str],
    excluded_positives: set[str],
    manifest_n_fragments: dict[str, int],
) -> dict[str, object]:
    counts: dict[str, int] = defaultdict(int)
    pos_by_uniprot: dict[str, int] = {}
    for pid, label in zip(protein_ids, labels.tolist()):
        uid = normalize_to_uniprot(pid)
        counts[uid] += 1
        pos_by_uniprot[uid] = int(label)
    multi_in_fasta = sorted(uid for uid, n in counts.items() if n > 1)
    multi_listed = sorted(uid for uid in multi_in_fasta if uid in positive_set)
    multi_unlisted = sorted(uid for uid in multi_in_fasta if uid not in positive_set)
    manifest_multi = sorted(
        uid for uid, n in manifest_n_fragments.items() if n > 1 and uid in counts
    )
    return {
        "n_uniprot_in_train_fasta": len(counts),
        "n_multi_fragment_uniprot_in_train_fasta": len(multi_in_fasta),
        "n_multi_fragment_listed_in_train_fasta": len(multi_listed),
        "n_multi_fragment_unlisted_in_train_fasta": len(multi_unlisted),
        "multi_fragment_listed_in_train_fasta": multi_listed,
        "excluded_multi_fragment_positives": sorted(excluded_positives),
        "n_manifest_multi_fragment_uniprot_present": len(manifest_multi),
        "note": (
            "train_fixed.py drops listed multi-fragment accessions from the "
            "positive set. Unlisted multi-fragment accessions remain in FASTA "
            "and were previously free to split across fragment-level folds."
        ),
    }


def load_kfold_index_splits(
    membership_tsv: Path,
    protein_ids: list[str],
    k_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read a long membership TSV into sklearn-style (train_idx, val_idx) splits."""
    pid_to_index = {pid: i for i, pid in enumerate(protein_ids)}
    if len(pid_to_index) != len(protein_ids):
        raise ValueError("Duplicate protein IDs in training dataset")
    fold_to_split: dict[int, dict[str, list[int]]] = {
        fold: {"train": [], "val": []} for fold in range(1, k_folds + 1)
    }
    membership_tsv = Path(membership_tsv).expanduser()
    if not membership_tsv.is_file():
        raise FileNotFoundError(f"Fold membership TSV not found: {membership_tsv}")
    with membership_tsv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_id", "fold", "split"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Membership TSV missing {required}: {membership_tsv}")
        for row in reader:
            pid = row["protein_id"].strip()
            fold_num = int(float(row["fold"]))
            split_name = row["split"].strip()
            if split_name not in {"train", "val"}:
                continue
            if fold_num not in fold_to_split:
                raise ValueError(
                    f"Unexpected fold {fold_num} in {membership_tsv} (k_folds={k_folds})"
                )
            if pid not in pid_to_index:
                raise ValueError(f"Membership protein_id not in train dataset: {pid}")
            idx = pid_to_index[pid]
            if row.get("dataset_index"):
                expected = int(float(row["dataset_index"]))
                if expected != idx:
                    raise ValueError(
                        f"dataset_index mismatch for {pid}: TSV={expected} dataset={idx}"
                    )
            fold_to_split[fold_num][split_name].append(idx)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = set(range(len(protein_ids)))
    for fold_num in range(1, k_folds + 1):
        train_idx = np.array(sorted(set(fold_to_split[fold_num]["train"])), dtype=np.int64)
        val_idx = np.array(sorted(set(fold_to_split[fold_num]["val"])), dtype=np.int64)
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError(f"Empty train/val assignment for fold {fold_num}")
        overlap = set(train_idx.tolist()) & set(val_idx.tolist())
        if overlap:
            raise ValueError(f"Train/val index overlap in fold {fold_num}: {len(overlap)}")
        covered = set(train_idx.tolist()) | set(val_idx.tolist())
        if covered != all_idx:
            raise ValueError(
                f"Fold {fold_num} covers {len(covered)}/{len(protein_ids)} proteins"
            )
        splits.append((train_idx, val_idx))
    return splits


def val_fold_from_long_rows(long_rows: list[dict], protein_ids: list[str]) -> np.ndarray:
    val_fold = np.zeros(len(protein_ids), dtype=np.int64)
    pid_to_index = {pid: i for i, pid in enumerate(protein_ids)}
    for row in long_rows:
        if str(row["split"]) != "val":
            continue
        pid = str(row["protein_id"])
        if pid not in pid_to_index:
            continue
        val_fold[pid_to_index[pid]] = int(float(row["fold"]))
    return val_fold
