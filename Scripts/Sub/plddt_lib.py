#!/usr/bin/env python3
"""Shared AlphaFold pLDDT helpers.

AFDB PDB CA B-factors (PDB columns 61-66) are pLDDT on a 0-100 scale.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
HELIX_CODES = {"H", "G", "I"}
SHEET_CODES = {"E", "B"}
PLDDT_BINS = ("lt50", "50to70", "70to90", "ge90")
BIN_LABELS = {
    "lt50": "pLDDT <50",
    "50to70": "pLDDT 50–70",
    "70to90": "pLDDT 70–90",
    "ge90": "pLDDT ≥90",
}
KEY_FEATURES = [
    "dssp_sheet_frac",
    "dssp_max_sheet_run_frac",
    "dssp_avg_asa_norm",
    "dssp_exposed_frac_norm",
    "instability_index",
    "geo_radius_gyration",
    "geo_compactness_rg_over_n_cuberoot",
]
PLDDT_SCORE_FEATURES = [
    "plddt_mean",
    "frac_plddt_lt50",
    "frac_plddt_lt70",
    "structure_coverage",
]
SMOKE_INSULIN = "AF-P01308-F1-model_v4_A"
SMOKE_IDR = "AF-Q9UPX6-F1-model_v4_A"


def open_text(path: Path):
    if path.suffix == ".gz" or str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def strip_chain_suffix(protein_id: str) -> str:
    parts = protein_id.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 1:
        return parts[0]
    return protein_id


def chain_from_protein_id(protein_id: str) -> Optional[str]:
    parts = protein_id.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 1:
        return parts[1]
    return None


def protein_to_structure_id(protein_id: str) -> str:
    if protein_id.endswith("_A") or protein_id.endswith("_B"):
        return protein_id[:-2]
    return protein_id


def uniprot_from_protein_id(protein_id: str) -> str:
    if "|" in protein_id:
        parts = protein_id.split("|")
        if len(parts) >= 2:
            return parts[1]
    if protein_id.startswith("AF-") and protein_id.count("-") >= 2:
        return protein_id.split("-")[1]
    return protein_id


def resolve_pdb_path(pdb_dir: Path, protein_id: str) -> Optional[Path]:
    stem = strip_chain_suffix(protein_id)
    for suffix in (".pdb", ".pdb.gz"):
        candidate = pdb_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def parse_fasta_ids(fasta_path: Path) -> List[str]:
    ids: List[str] = []
    with open_text(fasta_path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def parse_ca_plddt_from_pdb(
    pdb_path: Path,
    prefer_chain: Optional[str] = None,
) -> Tuple[List[Tuple[int, float]], Dict[str, object]]:
    chain_rows: Dict[str, List[Tuple[int, float]]] = {}
    chain_seen: Dict[str, set] = {}

    with open_text(pdb_path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip() if len(line) > 26 else ""
            key = (resseq, icode)
            try:
                plddt = float(line[60:66].strip())
            except ValueError:
                continue
            if chain_id not in chain_rows:
                chain_rows[chain_id] = []
                chain_seen[chain_id] = set()
            if key in chain_seen[chain_id]:
                continue
            chain_seen[chain_id].add(key)
            ridx = len(chain_rows[chain_id]) + 1
            chain_rows[chain_id].append((ridx, plddt))

    if not chain_rows:
        return [], {"n_chains": 0}

    if prefer_chain and prefer_chain in chain_rows:
        chosen_chain = prefer_chain
    else:
        chosen_chain = max(chain_rows.keys(), key=lambda c: len(chain_rows[c]))

    rows = chain_rows[chosen_chain]
    meta = {
        "n_chains": len(chain_rows),
        "chosen_chain": chosen_chain,
        "n_ca": len(rows),
    }
    return rows, meta


def terminal_low_runs(plddt: Sequence[float], thresh: float = 50.0) -> Tuple[int, int]:
    nterm = 0
    for val in plddt:
        if val < thresh:
            nterm += 1
        else:
            break
    cterm = 0
    for val in reversed(list(plddt)):
        if val < thresh:
            cterm += 1
        else:
            break
    return nterm, cterm


def termini_keep_mask(
    plddt: np.ndarray,
    *,
    thresh: float = 50.0,
    min_run: int = 20,
) -> Tuple[np.ndarray, int, int]:
    nterm, cterm = terminal_low_runs(plddt.tolist(), thresh=thresh)
    if nterm < min_run:
        nterm = 0
    if cterm < min_run:
        cterm = 0
    mask = np.ones(len(plddt), dtype=bool)
    if nterm:
        mask[:nterm] = False
    if cterm:
        mask[len(plddt) - cterm :] = False
    return mask, nterm, cterm


def summarize_plddt(plddt: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(plddt, dtype=float)
    n = int(arr.size)
    if n == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("pLDDT array is empty or contains non-finite values")
    nterm, cterm = terminal_low_runs(arr.tolist(), thresh=50.0)
    return {
        "plddt_mean": float(np.mean(arr)),
        "plddt_median": float(np.median(arr)),
        "plddt_min": float(np.min(arr)),
        "frac_plddt_lt50": float(np.mean(arr < 50.0)),
        "frac_plddt_lt70": float(np.mean(arr < 70.0)),
        "frac_plddt_ge90": float(np.mean(arr >= 90.0)),
        "n_res": float(n),
        "n_plddt_ge70": float(np.sum(arr >= 70.0)),
        "nterm_plddt_lt50_run": float(nterm),
        "cterm_plddt_lt50_run": float(cterm),
        "disorder_frac": float(np.mean(arr < 50.0)),
    }


def plddt_bin_label(value: float) -> str:
    if value < 50.0:
        return "lt50"
    if value < 70.0:
        return "50to70"
    if value < 90.0:
        return "70to90"
    return "ge90"


def load_coverage_by_fragment(manifest_path: Path) -> Dict[str, Tuple[int, int]]:
    """Map fragment protein_id -> (fragment_length, full_seq_len)."""
    out: Dict[str, Tuple[int, int]] = {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            full_len = int(float(row["full_seq_len"]))
            ids = [x.strip() for x in row["fragment_ids"].split(";") if x.strip()]
            lengths = [int(float(x)) for x in row["fragment_lengths"].split(";") if x.strip()]
            if len(ids) != len(lengths):
                raise ValueError(f"Fragment id/length mismatch for {row.get('uniprot_id')}")
            for fid, flen in zip(ids, lengths):
                out[fid] = (flen, full_len)
    return out


def structure_coverage(protein_id: str, n_res: int, coverage_map: Dict[str, Tuple[int, int]]) -> float:
    rec = coverage_map.get(protein_id)
    if rec is None:
        rec = coverage_map.get(strip_chain_suffix(protein_id))
    if rec is None:
        return float("nan")
    frag_len, full_len = rec
    if full_len <= 0:
        return float("nan")
    use_len = n_res if n_res > 0 else frag_len
    return float(use_len) / float(full_len)


def count_ca_tsv_rows(ca_path: Path) -> Optional[int]:
    if not ca_path.exists():
        return None
    n = 0
    with open(ca_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for _ in reader:
            n += 1
    return n


def write_plddt_tsv(path: Path, rows: Iterable[Tuple[int, float]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["residue_index", "plddt"])
        for ridx, plddt in rows:
            writer.writerow([ridx, f"{plddt:.4f}"])


def read_plddt_tsv(path: Path) -> np.ndarray:
    vals: List[float] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            vals.append(float(row["plddt"]))
    return np.asarray(vals, dtype=float)


def longest_true_run(mask: Iterable[bool]) -> int:
    best = 0
    current = 0
    for val in mask:
        if val:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def extended_dssp_from_residues(
    seq: str,
    ss_list: List[str],
    exposed_mask: List[bool],
    n: float,
) -> Dict[str, float]:
    charged = set("KRHDE")
    aromatic = set("FWYH")
    sheet_mask = [s in SHEET_CODES for s in ss_list]
    charged_exposed = sum(
        1 for aa, exposed in zip(seq, exposed_mask) if exposed and aa in charged
    )
    aromatic_exposed = sum(
        1 for aa, exposed in zip(seq, exposed_mask) if exposed and aa in aromatic
    )
    max_sheet_run = longest_true_run(sheet_mask)
    n_term = max(1, int(round(0.1 * n)))
    c_term = max(1, int(round(0.1 * n)))
    n_exposed_frac = sum(1 for v in exposed_mask[:n_term] if v) / n_term
    c_exposed_frac = sum(1 for v in exposed_mask[-c_term:] if v) / c_term
    return {
        "dssp_exposed_charged_frac": charged_exposed / n,
        "dssp_exposed_aromatic_frac": aromatic_exposed / n,
        "dssp_max_sheet_run_frac": max_sheet_run / n,
        "dssp_n_term_exposed_frac": n_exposed_frac,
        "dssp_c_term_exposed_frac": c_exposed_frac,
    }


def load_dssp_residue_arrays(
    dssp_dir: Path,
    protein_id: str,
    seq: str,
) -> Optional[Dict[str, object]]:
    structure_id = protein_to_structure_id(protein_id)
    npy_path = dssp_dir / f"{structure_id}_dssp.npy"
    if not npy_path.exists():
        npy_path = dssp_dir / f"{protein_id}_dssp.npy"
    if not npy_path.exists():
        return None
    try:
        arr = np.load(npy_path)
    except OSError:
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 14:
        return None
    clean_seq = "".join(aa for aa in seq.upper() if aa in AA_ORDER)
    n_res = min(arr.shape[0], len(clean_seq))
    if n_res == 0:
        return None
    seq_use = clean_seq[:n_res]
    arr = arr[:n_res]
    asa = arr[:, 4]
    ss_oh = arr[:, 5:14]
    ss_list = []
    for idx in range(n_res):
        state = int(np.argmax(ss_oh[idx]))
        ss_list.append(["H", "B", "E", "G", "I", "T", "S", "C", "C"][min(state, 8)])
    return {"ss_list": ss_list, "asa": np.asarray(asa, dtype=float), "seq": seq_use, "n": n_res}


def dssp_features_from_mask(
    ss_list: List[str],
    asa: np.ndarray,
    seq: str,
    mask: np.ndarray,
) -> Optional[Dict[str, float]]:
    keep = np.asarray(mask, dtype=bool)
    if keep.size != len(ss_list) or keep.size != len(asa) or keep.size != len(seq):
        n = min(keep.size, len(ss_list), len(asa), len(seq))
        keep = keep[:n]
        ss_list = ss_list[:n]
        asa = asa[:n]
        seq = seq[:n]
    ss_use = [s for s, m in zip(ss_list, keep) if m]
    asa_use = asa[keep]
    seq_use = "".join(aa for aa, m in zip(seq, keep) if m)
    n = float(len(ss_use))
    if n < 3:
        return None
    helix = sum(1 for s in ss_use if s in HELIX_CODES) / n
    sheet = sum(1 for s in ss_use if s in SHEET_CODES) / n
    coil = sum(1 for s in ss_use if s not in HELIX_CODES and s not in SHEET_CODES) / n
    exposed = asa_use >= 0.25
    exposed_mask = [bool(v) for v in exposed]
    out = {
        "dssp_helix_frac": helix,
        "dssp_sheet_frac": sheet,
        "dssp_coil_frac": coil,
        "dssp_avg_asa_norm": float(np.mean(asa_use)),
        "dssp_exposed_frac_norm": float(np.mean(exposed)),
        "dssp_n_confident": n,
    }
    out.update(extended_dssp_from_residues(seq_use, ss_use, exposed_mask, n))
    return out


def parse_ca_coords(ca_path: Path) -> Optional[np.ndarray]:
    coords: List[List[float]] = []
    try:
        with open(ca_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                coords.append([float(row["x"]), float(row["y"]), float(row["z"])])
    except (OSError, KeyError, ValueError):
        return None
    if len(coords) < 3:
        return None
    return np.array(coords, dtype=float)


def compute_geometry_features(coords: np.ndarray) -> Dict[str, float]:
    n = coords.shape[0]
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    distances = np.linalg.norm(centered, axis=1)
    rg = float(np.sqrt(np.mean(distances**2)))
    end_to_end = float(np.linalg.norm(coords[-1] - coords[0]))
    gt = centered.T @ centered / n
    eigvals = np.sort(np.linalg.eigvalsh(gt))[::-1]
    l1, l2, l3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    asphericity = l1 - 0.5 * (l2 + l3)
    return {
        "geo_radius_gyration": rg,
        "geo_end_to_end_distance": end_to_end,
        "geo_asphericity": asphericity,
        "geo_max_extent": float(np.max(distances)),
        "geo_compactness_rg_over_n_cuberoot": rg / (n ** (1.0 / 3.0)),
        "geo_end_to_end_over_n": end_to_end / n,
        "geo_n_ca": float(n),
    }
