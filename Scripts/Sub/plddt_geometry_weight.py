#!/usr/bin/env python3
"""Geometry-only pLDDT downweighting for GGN-Apta residue graphs.

Keeps every residue (sequence / ProtT5 / ESM-2 unchanged). Downweights
unreliable AlphaFold geometry:

- node scalar: dihedrals (first 6 dims) and DSSP (last dssp_dim dims)
- node vector: orientations + sidechain proxy
- edges: scale features by min(w_i, w_j), or rebuild k-NN on high-pLDDT CAs

pLDDT <50 is treated as disordered / unreliable (AFDB guideline). Residues
in 50–70 can be downweighted with the soft ramp. Do not drop nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import plddt_lib as plddtlib

DIHEDRAL_DIM = 6
DEFAULT_DSSP_DIM = 14
DEFAULT_THRESH = 50.0
DEFAULT_SOFT_LO = 40.0
DEFAULT_SOFT_HI = 60.0
DEFAULT_KNN_KEEP_MIN = 0.5
WEIGHT_MODES = ("hard", "soft")
EDGE_MODES = ("scale", "knn_highconf", "none")


def resolve_plddt_tsv(plddt_dir: Path, protein_id: str) -> Path:
    plddt_dir = Path(plddt_dir)
    direct = plddt_dir / f"{protein_id}.tsv"
    if direct.is_file():
        return direct
    stem = plddtlib.strip_chain_suffix(protein_id)
    alt = plddt_dir / f"{stem}.tsv"
    if alt.is_file():
        return alt
    raise FileNotFoundError(
        f"pLDDT TSV not found for {protein_id}. Tried:\n"
        f"  1. {direct}\n"
        f"  2. {alt}"
    )


def residue_weights(
    plddt: np.ndarray,
    mode: str = "hard",
    thresh: float = DEFAULT_THRESH,
    soft_lo: float = DEFAULT_SOFT_LO,
    soft_hi: float = DEFAULT_SOFT_HI,
) -> np.ndarray:
    """Return per-residue weights in [0, 1]."""
    arr = np.asarray(plddt, dtype=float)
    mode_l = str(mode).strip().lower()
    if mode_l not in WEIGHT_MODES:
        raise ValueError(f"Unknown pLDDT weight mode {mode!r}. Expected {WEIGHT_MODES}.")
    if mode_l == "hard":
        return (arr >= float(thresh)).astype(np.float32)
    lo = float(soft_lo)
    hi = float(soft_hi)
    if hi <= lo:
        raise ValueError(f"soft_hi must be > soft_lo (got lo={lo}, hi={hi}).")
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_residue_weights(
    plddt_dir: Path,
    protein_id: str,
    n_res: int,
    mode: str = "hard",
    thresh: float = DEFAULT_THRESH,
    soft_lo: float = DEFAULT_SOFT_LO,
    soft_hi: float = DEFAULT_SOFT_HI,
    cache: Optional[dict] = None,
) -> np.ndarray:
    """Load pLDDT and return weights aligned to n_res graph nodes."""
    key = (str(plddt_dir), protein_id, int(n_res), mode, float(thresh), float(soft_lo), float(soft_hi))
    if cache is not None and key in cache:
        return cache[key]
    path = resolve_plddt_tsv(Path(plddt_dir), protein_id)
    plddt = plddtlib.read_plddt_tsv(path)
    if plddt.shape[0] != int(n_res):
        raise ValueError(
            f"pLDDT length mismatch for {protein_id}: "
            f"n_plddt={plddt.shape[0]} n_graph={n_res} file={path}"
        )
    weights = residue_weights(
        plddt, mode=mode, thresh=thresh, soft_lo=soft_lo, soft_hi=soft_hi
    )
    if cache is not None:
        cache[key] = weights
    return weights


def knn_coords_for_plddt(
    x_ca: torch.Tensor,
    weights: np.ndarray,
    edge_mode: str = "scale",
    knn_keep_min: float = DEFAULT_KNN_KEEP_MIN,
) -> torch.Tensor:
    """Optionally hide low-pLDDT CAs from k-NN (set to +inf). Nodes stay in the graph."""
    mode = str(edge_mode).strip().lower()
    if mode not in EDGE_MODES:
        raise ValueError(f"Unknown pLDDT edge mode {edge_mode!r}. Expected {EDGE_MODES}.")
    if mode != "knn_highconf":
        return x_ca
    w = torch.as_tensor(weights, device=x_ca.device, dtype=x_ca.dtype)
    if w.numel() != x_ca.shape[0]:
        raise ValueError(
            f"k-NN pLDDT weight length {w.numel()} != n_ca {x_ca.shape[0]}"
        )
    keep = w >= float(knn_keep_min)
    if int(keep.sum()) < 2:
        return x_ca
    masked = x_ca.clone()
    masked[~keep] = float("inf")
    return masked


def scale_geometry_features(
    node_s: torch.Tensor,
    node_v: torch.Tensor,
    edge_s: torch.Tensor,
    edge_v: torch.Tensor,
    edge_index: torch.Tensor,
    weights: np.ndarray,
    dihedral_dim: int = DIHEDRAL_DIM,
    dssp_dim: int = DEFAULT_DSSP_DIM,
    edge_mode: str = "scale",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale geometry channels. Sequence embeddings in the middle of node_s are kept."""
    w = torch.as_tensor(weights, device=node_s.device, dtype=node_s.dtype).view(-1)
    if w.numel() != node_s.shape[0]:
        raise ValueError(
            f"Geometry weight length {w.numel()} != n_nodes {node_s.shape[0]}"
        )
    w_node = w.view(-1, 1)
    node_s = node_s.clone()
    d_dim = int(dihedral_dim)
    if d_dim > 0:
        node_s[:, :d_dim] = node_s[:, :d_dim] * w_node
    ss_dim = int(dssp_dim)
    if ss_dim > 0:
        if node_s.shape[-1] < d_dim + ss_dim:
            raise ValueError(
                f"node_s dim {node_s.shape[-1]} too small for "
                f"dihedrals={d_dim} + dssp={ss_dim}"
            )
        node_s[:, -ss_dim:] = node_s[:, -ss_dim:] * w_node
    node_v = node_v * w.view(-1, 1, 1)

    mode = str(edge_mode).strip().lower()
    if mode not in EDGE_MODES:
        raise ValueError(f"Unknown pLDDT edge mode {edge_mode!r}. Expected {EDGE_MODES}.")
    if mode in ("scale", "knn_highconf") and edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        ew = torch.minimum(w[src], w[dst]).view(-1, 1)
        edge_s = edge_s * ew
        edge_v = edge_v * ew.view(-1, 1, 1)
    return node_s, node_v, edge_s, edge_v


def weighted_global_mean_pool(
    x: torch.Tensor,
    batch: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted GAP. Graphs with all-zero weights fall back to uniform mean.

    Reduce in float32 and drop non-finite nodes. AMP + hard 0/1 weights can
    otherwise do ``0 * Inf -> NaN``, which CUDA BCE rejects.
    """
    from torch_scatter import scatter_add
    from torch_geometric.nn import global_mean_pool as gap

    if batch is None:
        batch = x.new_zeros(x.shape[0], dtype=torch.long)
    if weight.numel() != x.shape[0]:
        raise ValueError(f"pool weight length {weight.numel()} != n_nodes {x.shape[0]}")

    x32 = x.to(dtype=torch.float32)
    w = weight.to(dtype=torch.float32, device=x.device).view(-1, 1)
    finite_x = torch.isfinite(x32)
    finite_w = torch.isfinite(w) & (w > 0)
    x32 = torch.where(finite_x, x32, torch.zeros_like(x32))
    w = torch.where(finite_w, w, torch.zeros_like(w))
    # Exclude a node if any hidden channel is non-finite.
    if x32.ndim > 1:
        w = w * finite_x.all(dim=1, keepdim=True).to(w.dtype)

    n_graph = int(batch.max().item()) + 1 if x32.shape[0] else 0
    num = scatter_add(x32 * w, batch, dim=0, dim_size=n_graph)
    den = scatter_add(w, batch, dim=0, dim_size=n_graph)
    out = num / den.clamp(min=eps)
    empty = den.view(-1) < eps
    if bool(empty.any()):
        uniform = gap(x32, batch)
        out = torch.where(empty.view(-1, 1), uniform, out)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.to(dtype=x.dtype)


def plddt_active(args) -> bool:
    return bool(getattr(args, "plddt_geometry_downweight", False)) or bool(
        getattr(args, "plddt_pool_weight", False)
    )
