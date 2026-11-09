#!/usr/bin/env python3
# Variant of train_fixed_lr_scheduler_no_ddp_fsdp_allow_sched_factor_1.py that applies
# training-time edge dropout using augment_eps as the edge keep probability
# (lower = stronger augmentation). Validation and inference use the full graph (no edge dropout).
import sys
import os
import gc
import datetime
from pathlib import Path

# Add Scripts/Sub directory to Python path to import local modules (gvp, pool)
# This allows the script to find gvp.py and pool.py in the same directory
# Use absolute path to ensure it works regardless of working directory
script_dir = Path(__file__).resolve().parent

# Ensure the directory is in sys.path
script_dir_str = str(script_dir)
if script_dir_str not in sys.path:
    sys.path.insert(0, script_dir_str)

from typing import Any
import time
import argparse
import math
import random
import re
from contextlib import nullcontext
from pathlib import Path
import logging
import warnings
import ml_collections
import copy
from Bio import SeqIO
import numpy as np
from ef10_checkpoint import (
    HitPolicy,
    aupr_checkpoint_should_save,
    ef10_stats,
    trajectory_should_stop,
)
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
# Import autocast - use new API if available, fallback to old API
try:
    from torch.amp import autocast
    AUTOCAST_DEVICE_TYPE = True  # New API requires device_type parameter
except ImportError:
    from torch.cuda.amp import autocast
    AUTOCAST_DEVICE_TYPE = False  # Old API doesn't require device_type
from torch.cuda.amp import GradScaler
from torch.autograd import Variable
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

# Note: The following classes were previously imported from model.py and nt_xent.py,
# but have been incorporated directly into this script.

# Import dependencies for model classes
from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
import plddt_geometry_weight as plddt_geo
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from torch.nn import functional as F
from pool import GraphMultisetTransformer
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
import torch.nn as nn

# NOTE: This script is single-process only (no multi-process distributed training).


# -----------------------------------------------------------------------------
# NT-Xent Contrastive Loss (single-process; no all-gather)
# -----------------------------------------------------------------------------

class NT_Xent(nn.Module):
    """Normalized Temperature-scaled Cross Entropy (NT-Xent) loss
    for contrastive learning."""

    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        """Build a mask to remove positives and self-similarities."""
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """Compute NT-Xent loss for a batch of positive pairs."""
        N = 2 * self.batch_size * self.world_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N, dtype=torch.long, device=positive_samples.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in binary classification.
    
    Focal Loss automatically down-weights easy negatives and focuses on hard examples,
    which is perfect for extreme imbalance (e.g., 164:1 ratio).
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0). Higher gamma focuses more on hard examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """Compute Focal Loss.
        
        Args:
            inputs: Predicted probabilities (after sigmoid) or logits
            targets: Ground truth binary labels
        """
        # Convert to float32 to avoid dtype mismatches with mixed precision
        inputs = inputs.float()
        targets = targets.float()
        
        # Ensure inputs are probabilities (apply sigmoid if needed)
        # Check if inputs are in [0, 1] range (probabilities) or logits
        if inputs.min() < 0 or inputs.max() > 1:
            # Inputs are logits, apply sigmoid
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs
        # CUDA BCE asserts on NaN / out-of-range. Weighted GAP under AMP can
        # emit NaN; min/max then treats the tensor as already-probabilities.
        probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # Compute BCE loss (both probs and targets are now float32)
        bce_loss = F.binary_cross_entropy(probs, targets, reduction="none")
        
        # Compute p_t: probability of true class
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting for positive class
        alpha_t = torch.where(targets == 1, self.alpha, 1.0)
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        if self.reduction == "none":
            return focal_loss
        return focal_loss.mean()


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric focal loss for stronger false-negative control."""

    def __init__(
        self,
        alpha_pos=1.0,
        alpha_neg=1.0,
        gamma_pos=0.0,
        gamma_neg=2.0,
        eps=1e-8,
        reduction="mean",
    ):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        if inputs.min() < 0 or inputs.max() > 1:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs

        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)
        pos_loss = -self.alpha_pos * targets * ((1.0 - probs) ** self.gamma_pos) * torch.log(probs)
        neg_loss = -self.alpha_neg * (1.0 - targets) * (probs ** self.gamma_neg) * torch.log(1.0 - probs)
        loss = pos_loss + neg_loss
        if self.reduction == "none":
            return loss
        return loss.mean()


class FocalTverskyLoss(nn.Module):
    """Focal-Tversky loss (batch-level) for FN-sensitive optimization."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        if inputs.min() < 0 or inputs.max() > 1:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)
        tp = torch.sum(targets * probs)
        fp = torch.sum((1.0 - targets) * probs)
        fn = torch.sum(targets * (1.0 - probs))
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1.0 - tversky) ** self.gamma


class PairwiseRankBCELoss(nn.Module):
    """Pairwise logistic ranking loss with a small BCE calibration term."""

    def __init__(self, margin=0.0, lambda_cal=0.25, eps=1e-6):
        super().__init__()
        self.margin = float(margin)
        self.lambda_cal = float(lambda_cal)
        self.eps = float(eps)

    def forward(self, inputs, targets):
        probs = inputs.float().reshape(-1).clamp(self.eps, 1.0 - self.eps)
        targets = targets.float().reshape(-1)
        logits = torch.logit(probs)
        pos_logits = logits[targets > 0.5]
        neg_logits = logits[targets <= 0.5]
        if pos_logits.numel() > 0 and neg_logits.numel() > 0:
            pair_differences = pos_logits[:, None] - neg_logits[None, :]
            rank_loss = F.softplus(self.margin - pair_differences).mean()
        else:
            rank_loss = logits.sum() * 0.0
        calibration_loss = F.binary_cross_entropy(probs, targets)
        return rank_loss + self.lambda_cal * calibration_loss


# Model Classes (previously from model.py)
# -----------------------------------------------------------------------------

class GVP_encoder(nn.Module):
    """GVP-GNN encoder for protein structure graph processing."""

    def __init__(
        self,
        node_in_dim,
        node_h_dim,
        edge_in_dim,
        edge_h_dim,
        seq_in=False,
        num_layers=3,
        drop_rate=0.1,
        use_residual=False,
        residual_alpha=1.0,
    ):
        super(GVP_encoder, self).__init__()
        activations = (F.relu, None)
        self.seq_in = seq_in
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha
        
        if self.seq_in:
            self.W_s = nn.Embedding(21, 21)
            node_in_dim = (node_in_dim[0] + 21, node_in_dim[1])

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=True),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=True),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(
                node_h_dim,
                edge_h_dim,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
            )
            for _ in range(num_layers)
        )

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0), activations=activations, vector_gate=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(ns, ns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None, pertubed=False):
        """Forward pass through the GVP encoder."""
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            prev_h_V = h_V
            if pertubed:
                h_V = layer(h_V, edge_index, h_E, pertubed=True)
            else:
                h_V = layer(h_V, edge_index, h_E)
            if self.use_residual:
                h_V = (
                    prev_h_V[0] + self.residual_alpha * h_V[0],
                    prev_h_V[1] + self.residual_alpha * h_V[1],
                )

        out = self.W_out(h_V)
        return self.dense(out)


class GGN_GO(nn.Module):
    """Graph-based neural network for protein function prediction."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        augment_eps,
        dropout,
        out_dim,
        pooling="MTP",
        pertub=False,
        use_residual=False,
        residual_alpha=1.0,
    ):
        super(GGN_GO, self).__init__()
        self.pertub = pertub
        
        self.GVP_encoder = GVP_encoder(
            node_in_dim=(input_dim, 3),
            node_h_dim=(hidden_dim, 16),
            edge_in_dim=(32, 1),
            edge_h_dim=(32, 1),
            seq_in=True,
            num_layers=num_layers,
            drop_rate=dropout,
            use_residual=use_residual,
            residual_alpha=residual_alpha,
        )

        self.pooling = pooling
        if self.pooling == "MTP":
            self.pool = GraphMultisetTransformer(
                hidden_dim,
                256,
                hidden_dim,
                None,
                10000,
                0.25,
                ["GMPool_G", "GMPool_G"],
                num_heads=8,
                layer_norm=True,
            )
        elif self.pooling == "GMP":
            self.pool = gmp
        else:
            self.pool = gap

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, out_dim),
            nn.Sigmoid(),
        )

        self.softmax = nn.Softmax(dim=-1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.pool_use_plddt_weight = False

    def encode_graph(
        self, h_V, edge_index, h_E, seq, batch=None, perturbed=False, node_weight=None
    ):
        """Encode and pool a protein graph into a graph-level embedding."""
        h_V_ = self.GVP_encoder(h_V, edge_index, h_E, seq, pertubed=perturbed)

        if self.pooling == "MTP":
            return self.pool(h_V_, batch, edge_index.long())
        if (
            self.pooling == "GAP"
            and getattr(self, "pool_use_plddt_weight", False)
            and node_weight is not None
        ):
            return plddt_geo.weighted_global_mean_pool(h_V_, batch, node_weight)
        return self.pool(h_V_, batch)

    def set_dropout_rate(self, p):
        """Update dropout probabilities across all Dropout modules."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = float(p)

    def forward(self, h_V, edge_index, h_E, seq, batch=None, node_weight=None):
        """Forward pass through the complete network."""
        x = self.encode_graph(
            h_V, edge_index, h_E, seq, batch=batch, perturbed=False, node_weight=node_weight
        )

        if self.pertub:
            x_pertub = self.encode_graph(
                h_V, edge_index, h_E, seq, batch=batch, perturbed=True, node_weight=node_weight
            )
            y_pred = self.readout(x)
            return y_pred, x, x_pertub
        else:
            y_pred = self.readout(x)
            return y_pred


def _plddt_pool_weight(model, batch):
    """Node weights for optional pLDDT-weighted GAP. None unless enabled."""
    if batch is None or not getattr(model, "pool_use_plddt_weight", False):
        return None
    return getattr(batch, "plddt_w", None)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CONFIG = ml_collections.ConfigDict(
    {
        "debug": False,
        "max_epochs": 100,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.01,  # Minimum improvement required to reset patience counter
        "batch_size": 16,
        "k_pos_override": None,
        "device": "cuda:0",
        "optimizer": {
            "lr": 5e-5,  # Will be updated from args.learning_rate
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-5,  # ADDED: L2 regularization to prevent overfitting
        },
        "gradient_clip_max_norm": 1.0,  # Will be updated from args.gradient_clip_max_norm
        "hidden_dim": 256,  # Will be updated from args.hidden_dim
        "num_layers": 3,  # Will be updated from args.num_layers
        "dropout": 0.1,  # Will be updated from args.dropout
        "dropout_schedule": "none",  # none, linear
        "dropout_final": 0.1,
        "dropout_schedule_start_epoch": 0,
        "augment_eps": 0.95,  # Will be updated from args.augment_eps
        "augment_eps_view2": None,  # Optional second-view keep probability for contrastive training
        "use_residual": False,
        "residual_alpha": 1.0,
        "scheduler": {
            "step_size": 30,
            "gamma": 0.75,
        },
        "scheduler_factor": None,  # Factor for ReduceLROnPlateau scheduler (set from args)
        "scheduler_patience": 5,   # Patience (epochs) for ReduceLROnPlateau (set from args)
        "lr_schedule_type": "plateau",  # plateau, exponential, cosine, onecycle, constant
        "exp_gamma": 0.95,
        "cosine_min_lr": 1e-6,
        "onecycle_max_lr": None,
        "onecycle_pct_start": 0.1,
        "onecycle_div_factor": 25.0,
        "onecycle_final_div_factor": 1e4,
        "model_save_path": "Model/model_",
        "loss_save_path": "Model/loss_",
        "test_result_path": "Model/test_",
        "use_mixed_precision": False,
        "seed_base": 1234,
        "deterministic": True,
        "deterministic_warn_only": True,
        "loss_type": "focal",  # focal, asymmetric_focal, focal_tversky, pairwise_rank_bce
        "asym_alpha_pos": 1.0,
        "asym_alpha_neg": 1.0,
        "asym_gamma_pos": 0.0,
        "asym_gamma_neg": 2.0,
        "tversky_alpha": 0.3,
        "tversky_beta": 0.7,
        "tversky_gamma": 1.33,
        "pairwise_margin": 0.0,
        "pairwise_lambda_cal": 0.25,
        "loss_use_pos_weight_scale": True,
        "negative_rounds": 1,
        "negative_rotation_seed": 20260718,
        "contrast_multi_view": False,
        "feature_mask_prob": 0.0,
        "feature_mask_value": 0.0,
        "ema_enabled": False,
        "ema_decay": 0.999,
        "swa_enabled": False,
        "swa_start_epoch": 70,
        "eval_use_ema": True,
        "eval_use_swa": False,
        "plddt_geometry_downweight": False,
        "plddt_pool_weight": False,
    }
)


def get_config():
    """Return a deep copy of the default configuration."""
    config = copy.deepcopy(CONFIG)
    return config


def build_model(config, output_dim, device):
    """Construct a GGN_GO model on the given device."""
    model = GGN_GO(
        2324,                  # input node feature dimension
        config.hidden_dim,     # hidden embedding size
        config.num_layers,     # number of message-passing layers
        config.augment_eps,    # edge dropout / keep probability
        config.dropout,        # dropout rate in GNN layers
        output_dim,            # output dimension (1 logit per protein)
        config.pooling,        # graph-level pooling type ("MTP" or "GMP")
        pertub=config.contrast,  # enable contrastive perturbation branch
        use_residual=config.use_residual,
        residual_alpha=config.residual_alpha,
    ).to(device)
    model.pool_use_plddt_weight = bool(getattr(config, "plddt_pool_weight", False))
    return model


def build_optimizer(config, model):
    """Construct the Adam optimizer for the given model."""
    return torch.optim.Adam(
        params=model.parameters(),
        **config.optimizer,
    )


def build_scaler(config):
    """Construct GradScaler for mixed precision, or return None."""
    if config.use_mixed_precision and torch.cuda.is_available():
        return GradScaler()
    return None


def set_all_seeds(seed):
    """Set Python, NumPy, and PyTorch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled=True, warn_only=True):
    """Configure deterministic behavior across CUDA/cuDNN/PyTorch."""
    if enabled:
        # Required by deterministic cuBLAS kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def make_worker_init_fn(base_seed):
    """Create a deterministic worker init function for DataLoader workers."""
    def _seed_worker(worker_id):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _seed_worker


# -----------------------------------------------------------------------------
# GPU selection helper
# -----------------------------------------------------------------------------

def set_gpu_ids(gpu_ids):
    """
    Set CUDA_VISIBLE_DEVICES to restrict which GPUs are visible.
    
    This must be called before PyTorch initializes CUDA
    (before any torch.cuda operations).
    Once set, CUDA_VISIBLE_DEVICES makes only the specified GPUs visible,
    and they are renumbered starting from 0. For example, if you set
    CUDA_VISIBLE_DEVICES="2,3", then GPU 2 becomes visible as cuda:0
    and GPU 3 becomes visible as cuda:1.
    
    Args:
        gpu_ids: Comma-separated string of GPU IDs (e.g., "0,1,2,3") or None
    """
    if gpu_ids is not None and gpu_ids.strip():
        # Set CUDA_VISIBLE_DEVICES before PyTorch initializes CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids.strip()
        # Note: We can't use logging here because logging might not be initialized yet
        # The logging will happen in the train() function after initialization


def save_model_state_dict(model, save_path, config, rank, world_size):
    """Save model state dict (single-process).

    This script runs in a single process, so we always save a regular
    `state_dict()` from the model.
    """
    # Create directory if it doesn't exist
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        logging.info(f"[rank0] Saving model to {save_path}...")
        for handler in logging.getLogger().handlers:
            handler.flush()

    torch.save(model.state_dict(), save_path)

    if rank == 0:
        logging.info(f"[rank0] Model saved successfully to {save_path}")
        for handler in logging.getLogger().handlers:
            handler.flush()


def attach_ef10_curves(
    payload,
    config,
    val_ef10,
    val_ef10_hits,
    ef_policy,
    aupr_policy_frozen,
    aupr_saved_epoch,
):
    """Add EF@10% curves only when the opt-in flag is on."""
    if not bool(getattr(config, "track_ef10_checkpoint", False)):
        return payload
    payload["val_ef10"] = list(val_ef10)
    payload["val_ef10_hits"] = list(val_ef10_hits)
    payload["ef10_saved_epoch"] = ef_policy.saved_epoch
    payload["ef10_best_hits"] = ef_policy.best_hits
    payload["ef10_policy_frozen"] = bool(ef_policy.frozen)
    payload["aupr_policy_frozen"] = bool(aupr_policy_frozen)
    payload["aupr_saved_epoch"] = aupr_saved_epoch
    return payload


# -----------------------------------------------------------------------------
# Dataset utilities (previously from utils.py)
# -----------------------------------------------------------------------------

def extract_dataset(fasta_file):
    """
    Extract sequences from a FASTA file into a dictionary.
    
    Supports both plain text and gzip-compressed (.gz) FASTA files.
    Uses Biopython's SeqIO for robust parsing.
    
    Args:
        fasta_file: Path to FASTA file (can be .fasta, .fa, .fasta.gz, .fa.gz, etc.)
    
    Returns:
        Dictionary mapping sequence names (IDs) to sequences as strings
    """
    fasta_data = {}
    
    # SeqIO.parse automatically handles gzip-compressed files when opened in binary mode
    # Check if file is gzipped by extension
    if fasta_file.endswith('.gz'):
        import gzip
        handle = gzip.open(fasta_file, 'rt')
    else:
        handle = open(fasta_file, 'r')
    
    try:
        for record in SeqIO.parse(handle, "fasta"):
            # Use record.id as the key (first word of header, before space)
            # and record.seq as a string for the sequence
            fasta_data[record.id] = str(record.seq)
    finally:
        handle.close()
    
    return fasta_data


def apply_edge_dropout(edge_index, edge_s, edge_v, keep_prob, device):
    """
    Training-time augmentation: randomly keep each edge with probability keep_prob.
    augment_eps in config is the keep probability; lower value = stronger augmentation.
    Returns (edge_index, edge_s, edge_v) subset; if keep_prob >= 1.0 returns inputs unchanged.
    """
    if keep_prob >= 1.0:
        if keep_prob > 1.0:
            logging.warning(
                "apply_edge_dropout: keep_prob=%.4f > 1.0 is invalid (expected in [0, 1]); "
                "no edge dropout applied.",
                keep_prob,
            )
        return edge_index, edge_s, edge_v
    num_edges = edge_index.shape[1]
    keep = (
        (torch.rand(num_edges, device=device, dtype=torch.float32) < keep_prob)
        .nonzero(as_tuple=True)[0]
    )
    if keep.numel() == 0:
        # keep at least one edge to avoid empty graph (fallback: keep first edge)
        keep = torch.zeros(1, device=device, dtype=torch.long)
    edge_index_new = edge_index[:, keep]
    edge_s_new = edge_s[keep]
    edge_v_new = edge_v[keep]
    return edge_index_new, edge_s_new, edge_v_new


def apply_feature_mask(node_s, mask_prob, mask_value=0.0):
    """Randomly mask scalar node features for contrastive view augmentation."""
    if mask_prob <= 0.0:
        return node_s
    mask = torch.rand_like(node_s) < float(mask_prob)
    return node_s.masked_fill(mask, float(mask_value))


def get_autocast_context(scaler):
    """Return an autocast context when mixed precision is active."""
    if scaler is None:
        return nullcontext()
    if AUTOCAST_DEVICE_TYPE:
        return autocast(device_type="cuda")
    return autocast()


def build_classification_loss(config):
    """Build classification loss module from config."""
    loss_type = str(getattr(config, "loss_type", "focal")).lower()
    if loss_type == "focal":
        return FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            reduction="none",
        )
    if loss_type == "asymmetric_focal":
        return AsymmetricFocalLoss(
            alpha_pos=config.asym_alpha_pos,
            alpha_neg=config.asym_alpha_neg,
            gamma_pos=config.asym_gamma_pos,
            gamma_neg=config.asym_gamma_neg,
            reduction="none",
        )
    if loss_type == "focal_tversky":
        return FocalTverskyLoss(
            alpha=config.tversky_alpha,
            beta=config.tversky_beta,
            gamma=config.tversky_gamma,
        )
    if loss_type == "pairwise_rank_bce":
        return PairwiseRankBCELoss(
            margin=config.pairwise_margin,
            lambda_cal=config.pairwise_lambda_cal,
        )
    raise ValueError(
        f"Unsupported --loss_type={config.loss_type}. "
        "Choose from: focal, asymmetric_focal, focal_tversky, pairwise_rank_bce."
    )


def compute_supervised_loss(loss_fn, y_pred, y_true, pos_weight_tensor, use_pos_weight_scale=True):
    """Compute supervised loss supporting both samplewise and scalar loss functions."""
    raw_loss = loss_fn(y_pred, y_true)
    if isinstance(raw_loss, torch.Tensor) and raw_loss.ndim > 0:
        if use_pos_weight_scale:
            raw_loss = raw_loss * (1.0 + (pos_weight_tensor - 1.0) * y_true)
        return raw_loss.mean()
    return raw_loss


def compute_dropout_for_epoch(config, epoch_idx):
    """Compute dropout value for the current epoch."""
    base = float(config.dropout)
    schedule = str(getattr(config, "dropout_schedule", "none")).lower()
    if schedule == "none":
        return base
    if schedule != "linear":
        raise ValueError(
            f"Unsupported --dropout_schedule={config.dropout_schedule}. "
            "Choose from: none, linear."
        )
    start_epoch = int(getattr(config, "dropout_schedule_start_epoch", 0))
    if epoch_idx < start_epoch:
        return base
    final = float(getattr(config, "dropout_final", base))
    denom = max(1, int(config.max_epochs) - 1 - start_epoch)
    progress = min(1.0, max(0.0, float(epoch_idx - start_epoch) / float(denom)))
    return base + (final - base) * progress


def _copy_model_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _update_ema_state(ema_state, model, decay):
    current = model.state_dict()
    if ema_state is None:
        return _copy_model_state(model)
    with torch.no_grad():
        for k, v in current.items():
            if not torch.is_tensor(v):
                ema_state[k] = v
                continue
            if torch.is_floating_point(v):
                ema_state[k].mul_(decay).add_(v.detach(), alpha=(1.0 - decay))
            else:
                ema_state[k] = v.detach().clone()
    return ema_state


def _update_swa_state(swa_state, model, count):
    current = model.state_dict()
    if swa_state is None:
        return _copy_model_state(model), 1
    with torch.no_grad():
        for k, v in current.items():
            if not torch.is_tensor(v):
                swa_state[k] = v
                continue
            if torch.is_floating_point(v):
                swa_state[k].mul_(float(count) / float(count + 1)).add_(
                    v.detach(), alpha=1.0 / float(count + 1)
                )
            else:
                swa_state[k] = v.detach().clone()
    return swa_state, count + 1


def maybe_swap_eval_weights(model, config, ema_state, swa_state, epoch_idx):
    """Swap in EMA/SWA weights for validation/checkpointing when enabled."""
    use_ema = bool(getattr(config, "ema_enabled", False) and getattr(config, "eval_use_ema", True))
    use_swa = bool(getattr(config, "swa_enabled", False) and getattr(config, "eval_use_swa", False))
    if use_ema and ema_state is not None:
        backup = _copy_model_state(model)
        model.load_state_dict(ema_state, strict=True)
        return backup, "ema"
    if use_swa and swa_state is not None and epoch_idx >= int(getattr(config, "swa_start_epoch", 0)):
        backup = _copy_model_state(model)
        model.load_state_dict(swa_state, strict=True)
        return backup, "swa"
    return None, "base"


def get_data_args(
    split="train",
    fasta_file=None,
    prottrans_output_path=None,
    esm2_output_path=None,
    pdb_dir=None,
    proc_pdb_output_path=None,
    dssp_output_path=None,
    num_workers=None,
    disable_prottrans=False,
    disable_esm2=False,
    disable_dssp=False,
    disable_dihedrals=False,
    disable_orientations=False,
    disable_sidechains=False,
    disable_rbf=False,
    disable_pos_embeddings=False,
    disable_edge_direction=False,
    plddt_geometry_downweight=False,
    plddt_dir=None,
    plddt_weight_mode="hard",
    plddt_thresh=50.0,
    plddt_soft_lo=40.0,
    plddt_soft_hi=60.0,
    plddt_edge_mode="scale",
    plddt_pool_weight=False,
    plddt_knn_keep_min=0.5,
):
    """
    Get data configuration with default paths for training or validation split.
    
    Args:
        split: Either "train" or "val" to specify which split's default FASTA file to use.
        fasta_file: Optional path to FASTA file. If None, uses default based on split.
        prottrans_output_path: Optional path to ProtTrans embeddings. If None, uses default.
        esm2_output_path: Optional path to ESM2 embeddings. If None, uses default.
        pdb_dir: Optional path to PDB directory. If None, uses default.
        proc_pdb_output_path: Optional path to processed PDB data. If None, uses default.
        dssp_output_path: Optional path to DSSP features. If None, uses default.
        num_workers: Optional number of workers. If None, uses default (8).
    
    Returns:
        Args object with file paths for the specified data split.
    
    Example usage:
        # Using defaults:
        train_args = get_data_args("train")
        val_args = get_data_args("val")
        
        # With custom paths:
        train_args = get_data_args(
            "train",
            fasta_file="/path/to/train.fasta",
            num_workers=16
        )
    """
    class Args:
        def __init__(self):
            # FASTA file path differs by split
            if fasta_file is not None:
                self.fasta_file = fasta_file
            elif split == "train":
                self.fasta_file = "../data/nrPDB-GO_train_sequences.fasta"
            else:  # val
                self.fasta_file = "../data/nrPDB-GO_val_sequences.fasta"
            
            # All other paths are shared between train and val
            self.prottrans_output_path = (
                prottrans_output_path or "../data/bpDataSet/prottrans/"
            )
            self.esm2_output_path = esm2_output_path or "../data/bpDataSet/esm2/"
            self.pdb_dir = pdb_dir or "../data/pdb_dir/"
            self.proc_pdb_output_path = (
                proc_pdb_output_path or "../data/bpDataSet/structure_data/"
            )
            self.dssp_output_path = dssp_output_path or "../data/bpDataSet/dssp/"
            self.num_workers = num_workers or 8
            self.disable_prottrans = disable_prottrans
            self.disable_esm2 = disable_esm2
            self.disable_dssp = disable_dssp
            self.disable_dihedrals = disable_dihedrals
            self.disable_orientations = disable_orientations
            self.disable_sidechains = disable_sidechains
            self.disable_rbf = disable_rbf
            self.disable_pos_embeddings = disable_pos_embeddings
            self.disable_edge_direction = disable_edge_direction
            self.plddt_geometry_downweight = plddt_geometry_downweight
            self.plddt_dir = plddt_dir
            self.plddt_weight_mode = plddt_weight_mode
            self.plddt_thresh = plddt_thresh
            self.plddt_soft_lo = plddt_soft_lo
            self.plddt_soft_hi = plddt_soft_hi
            self.plddt_edge_mode = plddt_edge_mode
            self.plddt_pool_weight = plddt_pool_weight
            self.plddt_knn_keep_min = plddt_knn_keep_min
    
    return Args()


# Convenience aliases for backward compatibility
def get_arg_train(**kwargs):
    """Get training data configuration (alias for get_data_args('train'))."""
    return get_data_args("train", **kwargs)


def get_arg_val(**kwargs):
    """Get validation data configuration (alias for get_data_args('val'))."""
    return get_data_args("val", **kwargs)


# -----------------------------------------------------------------------------
# ProteinGraphDataset class (previously from protein_graph_dataset.py)
# -----------------------------------------------------------------------------

import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import my_Features as features


# Note: GO annotation loading has been removed as this script uses binary
# aptamer affinity labels instead of GO term annotations.


def _normalize(tensor, dim=-1):
    """Normalizes a torch.Tensor along dimension dim without nans."""
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    Returns an RBF embedding of torch.Tensor D along a new axis=-1.
    From https://github.com/jingraham/neurips19-graph-protein-design
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


class BatchSampler(data.Sampler):
    """
    A torch.utils.data.Sampler which samples batches according to
    a maximum number of graph nodes.
    From https://github.com/jingraham/neurips19-graph-protein-design
    """

    def __init__(self, node_counts, max_nodes=3000, shuffle=True,
                 with_replacement=False, stop_when_exhausted=False):
        self.node_counts = node_counts
        self.idx = [
            i for i in range(len(node_counts)) if node_counts[i] <= max_nodes
        ]
        self.shuffle = shuffle
        self.with_replacement = with_replacement
        self.stop_when_exhausted = stop_when_exhausted
        self.max_nodes = max_nodes
        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle:
            random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self):
        if not self.batches:
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        if not self.batches:
            self._form_batches()
        for batch in self.batches:
            yield batch


class ProteinGraphDataset(data.Dataset):
    """
    A map-style torch.utils.data.Dataset which transforms protein structures
    into featurized graphs.
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    """

    def __init__(
        self,
        dataset,
        index,
        args,
        task_list,
        num_positional_embeddings=16,
        top_k=50,
        num_rbf=16,
        device="cpu",
    ):
        super(ProteinGraphDataset, self).__init__()

        # Filter dataset to specified indices (for train/val/test splits)
        self.dataset = {}
        index = set(index)  # Convert to set for fast lookup
        for i, ID in enumerate(dataset):
            if i in index:
                self.dataset[ID] = dataset[ID]
        self.IDs = list(self.dataset.keys())

        # Store file paths for input data and pre-computed features
        self.fasta_file = args.fasta_file
        self.output_prottrans = args.prottrans_output_path
        self.output_esm2 = args.esm2_output_path
        self.proc_pdb_output_path = args.proc_pdb_output_path
        self.pdb_dir = args.pdb_dir
        self.output_dssp = args.dssp_output_path
        self.disable_prottrans = getattr(args, "disable_prottrans", False)
        self.disable_esm2 = getattr(args, "disable_esm2", False)
        self.disable_dssp = getattr(args, "disable_dssp", False)
        self.disable_dihedrals = getattr(args, "disable_dihedrals", False)
        self.disable_orientations = getattr(args, "disable_orientations", False)
        self.disable_sidechains = getattr(args, "disable_sidechains", False)
        self.disable_rbf = getattr(args, "disable_rbf", False)
        self.disable_pos_embeddings = getattr(args, "disable_pos_embeddings", False)
        self.disable_edge_direction = getattr(args, "disable_edge_direction", False)
        self.plddt_geometry_downweight = getattr(args, "plddt_geometry_downweight", False)
        self.plddt_dir = getattr(args, "plddt_dir", None)
        self.plddt_weight_mode = getattr(args, "plddt_weight_mode", "hard")
        self.plddt_thresh = float(getattr(args, "plddt_thresh", 50.0))
        self.plddt_soft_lo = float(getattr(args, "plddt_soft_lo", 40.0))
        self.plddt_soft_hi = float(getattr(args, "plddt_soft_hi", 60.0))
        self.plddt_edge_mode = getattr(args, "plddt_edge_mode", "scale")
        self.plddt_pool_weight = getattr(args, "plddt_pool_weight", False)
        self.plddt_knn_keep_min = float(getattr(args, "plddt_knn_keep_min", 0.5))
        self._plddt_weight_cache = {}
        self.task_list = task_list

        # Graph construction and feature encoding hyperparameters
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        
        # Pre-compute number of residues per protein (for batching)
        self.node_counts = [len(self.dataset[ID][0]) for ID in self.IDs]

        # Amino acid encoding: single-letter code -> integer index
        self.letter_to_num = {
            "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7,
            "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
            "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19, "X": 20
        }
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        
        # Cache for ESM2 embedding dimension (inferred from first successful load)
        self._esm2_dim = None
        self._dssp_dim = None

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        return self._featurize_as_graph(idx)

    def _featurize_as_graph(self, idx):
        """Convert a protein structure into a graph representation."""
        name = self.IDs[idx]
        
        with torch.no_grad():
            coords = np.load(
                self.proc_pdb_output_path + name + ".npy", allow_pickle=True
            )
            coords = torch.as_tensor(
                coords, device=self.device, dtype=torch.float32
            )
            seq = torch.as_tensor(
                [self.letter_to_num[aa] for aa in self.dataset[name]],
                device=self.device,
                dtype=torch.long,
            )
            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf
            X_ca = coords[:, 1]
            plddt_w_np = None
            X_ca_knn = X_ca
            if self.plddt_geometry_downweight or self.plddt_pool_weight:
                if not self.plddt_dir:
                    raise ValueError(
                        "plddt_dir is required when pLDDT geometry or pool weighting is enabled"
                    )
                plddt_w_np = plddt_geo.load_residue_weights(
                    self.plddt_dir,
                    name,
                    int(X_ca.shape[0]),
                    mode=self.plddt_weight_mode,
                    thresh=self.plddt_thresh,
                    soft_lo=self.plddt_soft_lo,
                    soft_hi=self.plddt_soft_hi,
                    cache=self._plddt_weight_cache,
                )
                if self.plddt_geometry_downweight:
                    X_ca_knn = plddt_geo.knn_coords_for_plddt(
                        X_ca,
                        plddt_w_np,
                        edge_mode=self.plddt_edge_mode,
                        knn_keep_min=self.plddt_knn_keep_min,
                    )
            edge_index = torch_cluster.knn_graph(X_ca_knn, k=self.top_k)
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            geo_edge_feat = self._get_geo_edge_feat(X_ca, edge_index)
            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            if self.disable_dihedrals:
                dihedrals = torch.zeros_like(dihedrals)
            if self.disable_orientations:
                orientations = torch.zeros_like(orientations)
            if self.disable_sidechains:
                sidechains = torch.zeros_like(sidechains)
            if self.disable_rbf:
                rbf = torch.zeros_like(rbf)
            if self.disable_pos_embeddings:
                pos_embeddings = torch.zeros_like(pos_embeddings)

            if self.disable_prottrans:
                prottrans_feat = torch.zeros(
                    len(self.dataset[name]), 1024, device=self.device, dtype=torch.float32
                )
            else:
                prottrans_feat = torch.tensor(
                    np.load(self.output_prottrans + name + ".npy", allow_pickle=True)
                )
            # ESM2 embeddings may be missing for some proteins
            # If missing, create a zero tensor with the expected shape
            if self.disable_esm2:
                if self._esm2_dim is None:
                    self._esm2_dim = 1280
                esm_feat = torch.zeros(
                    len(self.dataset[name]),
                    self._esm2_dim,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                esm2_path = self.output_esm2 + name + ".npy"
                try:
                    esm_feat = torch.tensor(np.load(esm2_path, allow_pickle=True))
                    # Cache ESM2 embedding dimension for future use
                    if self._esm2_dim is None and len(esm_feat.shape) >= 2:
                        self._esm2_dim = esm_feat.shape[-1]
                except FileNotFoundError:
                    # ESM2 embedding is missing - create zero tensor with expected shape
                    seq_len = len(self.dataset[name])  # Sequence length from dataset
                    # Use cached dimension if available, otherwise try to infer from other features
                    if self._esm2_dim is None:
                        # Try to infer from ProtTrans (common ESM2 models: 650M=1280, 3B=2560)
                        # Default to 1280 (ESM2-650M) as a reasonable guess
                        self._esm2_dim = 1280
                        logging.warning(
                            "ESM2 embedding missing for %s. Using default dimension %d. "
                            "This will be corrected once ESM2 embeddings are added.",
                            name, self._esm2_dim
                        )
                    else:
                        logging.warning(
                            "ESM2 embedding missing for %s. "
                            "Using zero tensor (shape: [%d, %d]). "
                            "This will be corrected once ESM2 embeddings are added.",
                            name, seq_len, self._esm2_dim
                        )
                    # Create zero tensor with shape [seq_len, esm2_dim]
                    esm_feat = torch.zeros(
                        seq_len,
                        self._esm2_dim,
                        device=self.device,
                        dtype=torch.float32,
                    )
            # DSSP files may not have chain identifiers in their filenames
            # Try with full name first, then without chain identifier if needed
            if self.disable_dssp:
                if self._dssp_dim is None:
                    self._dssp_dim = 14
                dssp = torch.zeros(
                    len(self.dataset[name]),
                    self._dssp_dim,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                dssp_path = self.output_dssp + name + "_dssp.npy"
                try:
                    dssp = torch.tensor(np.load(dssp_path, allow_pickle=True))
                    if self._dssp_dim is None and len(dssp.shape) >= 2:
                        self._dssp_dim = dssp.shape[-1]
                except FileNotFoundError:
                    # Try without chain identifier (strip _A, _B, etc. from end of name)
                    # Pattern: _[A-Z] or _[0-9] at the end before file extension
                    name_no_chain = re.sub(r'_[A-Z0-9]+$', '', name)
                    if name_no_chain != name:
                        dssp_path_no_chain = self.output_dssp + name_no_chain + "_dssp.npy"
                        try:
                            dssp = torch.tensor(np.load(dssp_path_no_chain, allow_pickle=True))
                            if self._dssp_dim is None and len(dssp.shape) >= 2:
                                self._dssp_dim = dssp.shape[-1]
                        except FileNotFoundError:
                            raise FileNotFoundError(
                                f"DSSP file not found for {name}. Tried:\n"
                                f"  1. {dssp_path}\n"
                                f"  2. {dssp_path_no_chain}"
                            )
                    else:
                        raise FileNotFoundError(f"DSSP file not found: {dssp_path}")

            node_s = torch.cat([dihedrals, prottrans_feat, esm_feat, dssp], dim=-1).to(
                torch.float32
            )
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            if self.disable_edge_direction:
                edge_v = torch.zeros_like(edge_v)

            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )
            if plddt_w_np is not None and self.plddt_geometry_downweight:
                dssp_dim = (
                    int(dssp.shape[-1])
                    if getattr(dssp, "ndim", 0) >= 2
                    else plddt_geo.DEFAULT_DSSP_DIM
                )
                node_s, node_v, edge_s, edge_v = plddt_geo.scale_geometry_features(
                    node_s,
                    node_v,
                    edge_s,
                    edge_v,
                    edge_index,
                    plddt_w_np,
                    dihedral_dim=plddt_geo.DIHEDRAL_DIM,
                    dssp_dim=dssp_dim,
                    edge_mode=self.plddt_edge_mode,
                )

            # Create dummy y tensor (not used - labels come from
            # aptamer list in training loop)
            # Shape: [1] for binary classification
            y = torch.zeros(1, device=self.device, dtype=torch.float32)

        data = torch_geometric.data.Data(
            x=X_ca,
            seq=seq,
            name=name,
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            edge_index=edge_index,
            mask=mask,
            y=y,
        )
        if plddt_w_np is not None:
            data.plddt_w = torch.as_tensor(
                plddt_w_np, device=self.device, dtype=torch.float32
            )
        return data

    def _dihedrals(self, X, eps=1e-7):
        """Compute backbone dihedral angles (phi, psi, omega) as node features."""
        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]
        n_2 = _normalize(torch.linalg.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = _normalize(torch.linalg.cross(u_1, u_0, dim=-1), dim=-1)
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(
        self, edge_index, num_embeddings=None, period_range=[2, 1000]
    ):
        """Compute sinusoidal positional embeddings for graph edges."""
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        """Compute backbone orientation vectors for each residue."""
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        """Compute sidechain orientation vectors for each residue."""
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.linalg.cross(c, n, dim=-1))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

    def _get_geo_edge_feat(self, X_ca, edge_index):
        """Compute geometric edge features: local coordinate frames,
        translations, and rotations."""
        u = torch.ones_like(X_ca)
        u[1:] = X_ca[1:] - X_ca[:-1]
        u = F.normalize(u, dim=-1)
        b = torch.ones_like(X_ca)
        b[:-1] = u[:-1] - u[1:]
        b = F.normalize(b, dim=-1)
        n = torch.ones_like(X_ca)
        n[:-1] = torch.linalg.cross(u[:-1], u[1:], dim=-1)
        n = F.normalize(n, dim=-1)
        local_frame = torch.stack([b, n, torch.linalg.cross(b, n, dim=-1)], dim=-1)
        node_j, node_i = edge_index
        t = F.normalize(X_ca[node_j] - X_ca[node_i], dim=-1)
        t = torch.einsum("ijk,ij->ik", local_frame[node_i], t)
        r = torch.matmul(
            local_frame[node_i].transpose(-1, -2), local_frame[node_j]
        )
        Q = self._quaternions(r)
        return torch.cat([t, Q], dim=-1)

    def _quaternions(self, R):
        """Convert 3x3 rotation matrices to quaternions using Shepperd's method."""
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(
                1
                + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)
            )
        )
        _R = lambda i, j: R[:, i, j]
        signs = torch.sign(
            torch.stack(
                [_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1
            )
        )
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q


class StratifiedPosNegBatchSampler(data.Sampler):
    """
    Batch sampler that enforces a fixed number of positives (k_pos)
    and negatives (batch_size - k_pos) in each batch.

    Sampling is with replacement, so positives/negatives can be reused
    across batches within an epoch. This guarantees at least k_pos
    positives per batch as long as there is at least one positive
    in the training set.
    """

    def __init__(
        self,
        pos_indices,
        neg_indices,
        batch_size,
        k_pos,
        num_batches,
        shuffle=True,
        with_replacement=False,
        stop_when_exhausted=False,
        balanced_epoch=False,
        negative_rounds=1,
        rotation_seed=20260718,
    ):
        self.pos_indices = list(pos_indices)
        self.neg_indices = list(neg_indices)
        self.batch_size = batch_size
        self.k_pos = max(1, min(k_pos, batch_size - 1))  # ensure at least 1 pos, 1 neg
        self.num_batches = num_batches
        self.shuffle = shuffle
        self.with_replacement = with_replacement
        self.stop_when_exhausted = stop_when_exhausted
        self.balanced_epoch = balanced_epoch
        self.negative_rounds = max(1, int(negative_rounds))
        self.rotation_seed = int(rotation_seed)
        self._epoch_index = 0
        rotation_rng = random.Random(self.rotation_seed)
        self._negative_order = list(self.neg_indices)
        rotation_rng.shuffle(self._negative_order)
        self._negative_offset = 0
        self._negative_cycle = 0

    def __len__(self):
        if (
            self.negative_rounds > 1
            and not self.with_replacement
            and self.stop_when_exhausted
            and not self.balanced_epoch
        ):
            batches_per_round = int(math.ceil(len(self.pos_indices) / self.k_pos))
            return min(self.num_batches, batches_per_round * self.negative_rounds)
        return self.num_batches

    def __iter__(self):
        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            # Fallback: no stratification possible, yield empty iterator
            return iter([])

        # Important: __iter__ is called once *per epoch* by the DataLoader.
        # We therefore keep all per-epoch sampling state local to this method
        # (no persistent self._pos_pool_idx / self._neg_pool_idx), so that
        # each epoch starts with a fresh pass over the pools even when
        # stop_when_exhausted=True.

        if (
            self.negative_rounds > 1
            and not self.with_replacement
            and self.stop_when_exhausted
            and not self.balanced_epoch
        ):
            yield from self._iter_rotating_negative_rounds()
            return

        pos_indices = list(self.pos_indices)
        neg_indices = list(self.neg_indices)

        # Balanced epoch sampling: oversample minority class to match majority
        if self.balanced_epoch:
            num_pos = len(pos_indices)
            num_neg = len(neg_indices)
            # Determine which class is minority and oversample it
            if num_pos < num_neg:
                # Oversample positives to match negatives
                repeat_factor = int(math.ceil(num_neg / num_pos))
                pos_indices = (pos_indices * repeat_factor)[:num_neg]  # Expand to match negatives
            elif num_neg < num_pos:
                # Oversample negatives to match positives
                repeat_factor = int(math.ceil(num_pos / num_neg))
                neg_indices = (neg_indices * repeat_factor)[:num_pos]  # Expand to match positives
            # Now both pools have equal size, use 50/50 split per batch
            k_pos = self.batch_size // 2
            k_neg = self.batch_size - k_pos
        else:
            k_neg = self.batch_size - self.k_pos

        if self.shuffle:
            random.shuffle(pos_indices)
            random.shuffle(neg_indices)

        pos_pool_idx = 0
        neg_pool_idx = 0

        for _ in range(self.num_batches):
            # --- Positives ---
            if self.balanced_epoch:
                # Balanced epoch: sample without replacement, cycle when exhausted
                if pos_pool_idx + k_pos > len(pos_indices):
                    pos_pool_idx = 0
                    if self.shuffle:
                        random.shuffle(pos_indices)
                batch_pos = pos_indices[pos_pool_idx : pos_pool_idx + k_pos]
                pos_pool_idx += k_pos
            elif self.with_replacement:
                batch_pos = random.choices(pos_indices, k=self.k_pos)
            else:
                # Without replacement: either cycle (default) or stop the epoch
                if pos_pool_idx + self.k_pos > len(pos_indices):
                    if self.stop_when_exhausted:
                        # End this epoch once positives are exhausted
                        break
                    else:
                        pos_pool_idx = 0
                        if self.shuffle:
                            random.shuffle(pos_indices)
                batch_pos = pos_indices[pos_pool_idx : pos_pool_idx + self.k_pos]
                pos_pool_idx += self.k_pos

            # --- Negatives ---
            if self.balanced_epoch:
                # Balanced epoch: sample without replacement, cycle when exhausted
                if neg_pool_idx + k_neg > len(neg_indices):
                    neg_pool_idx = 0
                    if self.shuffle:
                        random.shuffle(neg_indices)
                batch_neg = neg_indices[neg_pool_idx : neg_pool_idx + k_neg]
                neg_pool_idx += k_neg
            elif self.with_replacement:
                batch_neg = random.choices(neg_indices, k=k_neg)
            else:
                if neg_pool_idx + k_neg > len(neg_indices):
                    if self.stop_when_exhausted:
                        # End this epoch once negatives are exhausted
                        break
                    else:
                        neg_pool_idx = 0
                        if self.shuffle:
                            random.shuffle(neg_indices)
                batch_neg = neg_indices[neg_pool_idx : neg_pool_idx + k_neg]
                neg_pool_idx += k_neg

            batch = batch_pos + batch_neg
            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def _iter_rotating_negative_rounds(self):
        """Repeat each positive against disjoint, deterministically rotating negatives."""
        epoch_rng = random.Random(self.rotation_seed + self._epoch_index)
        self._epoch_index += 1
        k_neg = self.batch_size - self.k_pos
        emitted = 0
        max_batches = len(self)

        for _ in range(self.negative_rounds):
            positives = list(self.pos_indices)
            if self.shuffle:
                epoch_rng.shuffle(positives)
            for start in range(0, len(positives), self.k_pos):
                if emitted >= max_batches:
                    return
                batch_pos = positives[start : start + self.k_pos]
                if len(batch_pos) < self.k_pos:
                    continue
                batch_neg = []
                while len(batch_neg) < k_neg:
                    remaining = len(self._negative_order) - self._negative_offset
                    take = min(k_neg - len(batch_neg), remaining)
                    batch_neg.extend(
                        self._negative_order[
                            self._negative_offset : self._negative_offset + take
                        ]
                    )
                    self._negative_offset += take
                    if self._negative_offset == len(self._negative_order):
                        self._negative_cycle += 1
                        cycle_rng = random.Random(
                            self.rotation_seed + 1000003 * self._negative_cycle
                        )
                        cycle_rng.shuffle(self._negative_order)
                        self._negative_offset = 0
                batch = batch_pos + batch_neg
                if self.shuffle:
                    epoch_rng.shuffle(batch)
                emitted += 1
                yield batch


# -----------------------------------------------------------------------------
# Utilities for loading binary labels from an aptamer list
# -----------------------------------------------------------------------------
# The training script expects a **plain text file** that lists one UniProt
# accession per line, e.g.:
#     P12345
#     Q8ABC0
#     ...
#
# Pass that file via --aptamer_list_file /path/to/positives.txt.

def load_positive_proteins(path):
    """Load UniProt IDs with known high-affinity aptamers.

    The file is expected to contain one UniProt accession per line, for
    example:

        P12345
        Q8ABC0
        ...
    """
    positives = set()
    with open(path, "r") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            positives.add(name)
    return positives

def _normalize_to_uniprot(name: str) -> str:
    """Normalize various header/file naming schemes to a bare UniProt ID.

    Handles AlphaFold-style names such as:
        AF-P69905-F1-model_v4.pdb.gz   -> P69905
        AF-P69905-F1-model_v4.fasta.gz -> P69905

    Falls back to returning the input if no special pattern matches.
    """
    # Strip directory part if present
    name = name.split("/")[-1]

    # Remove common file extensions
    for ext in (".pdb.gz", ".cif.gz", ".fasta.gz", ".pdb", ".cif", ".fasta"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break

    # AlphaFold pattern: AF-<UNIPROT>-F<number>-...
    m = re.match(r"AF-([A-Z0-9]+)-F\d+", name)
    if m:
        return m.group(1)

    # Assume it's already a UniProt accession
    return name


def _extract_fragment_number(name: str) -> int:
    """Extract fragment number from AlphaFold-style names.
    
    Examples:
        AF-P69905-F1-model_v4 -> 1
        AF-P69905-F2-model_v4 -> 2
        P69905 -> None (no fragment)
    
    Returns:
        Fragment number as int, or None if no fragment pattern found.
    """
    # Strip directory part if present
    name = name.split("/")[-1]
    
    # Remove common file extensions
    for ext in (".pdb.gz", ".cif.gz", ".fasta.gz", ".pdb", ".cif", ".fasta"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    
    # AlphaFold pattern: AF-<UNIPROT>-F<number>-...
    m = re.match(r"AF-[A-Z0-9]+-F(\d+)", name)
    if m:
        return int(m.group(1))
    
    # No fragment pattern found
    return None


def _filter_multi_fragment_positives(positive_set, dataset_ids):
    """Filter out UniProt IDs that have multiple fragments in the dataset.
    
    For positive samples, if a UniProt ID appears with multiple fragment numbers
    (e.g., AF-P69905-F1-... and AF-P69905-F2-...), exclude that UniProt ID
    from the positive set to avoid label ambiguity.
    
    Args:
        positive_set: Set of UniProt IDs that are positive
        dataset_ids: List of all protein IDs in the dataset
    
    Returns:
        Filtered positive_set (with multi-fragment IDs removed)
    """
    # Build a map: UniProt ID -> set of fragment numbers
    uniprot_to_fragments = {}
    for protein_id in dataset_ids:
        normalized_id = _normalize_to_uniprot(protein_id)
        frag_num = _extract_fragment_number(protein_id)
        
        if normalized_id not in uniprot_to_fragments:
            uniprot_to_fragments[normalized_id] = set()
        if frag_num is not None:
            uniprot_to_fragments[normalized_id].add(frag_num)
    
    # Find positive UniProt IDs that have multiple fragments
    multi_fragment_positives = set()
    for uniprot_id in positive_set:
        fragments = uniprot_to_fragments.get(uniprot_id, set())
        # If there are 2+ distinct fragment numbers, it's multi-fragment
        if len(fragments) > 1:
            multi_fragment_positives.add(uniprot_id)
    
    # Remove multi-fragment positives from the set
    filtered_positive_set = positive_set - multi_fragment_positives
    
    return filtered_positive_set, multi_fragment_positives

def labels_from_batch_names(names, positive_set, device):
    """Create a binary label tensor (1 = has high-affinity aptamer).

    Both the batch names and the positive list are compared after
    normalizing to bare UniProt accessions.
    """
    labels = [
        1.0 if _normalize_to_uniprot(name) in positive_set else 0.0
        for name in names
    ]
    return torch.tensor(labels, device=device, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train(
    config,
    suffix,
    positive_list_file,
    train_fasta_file=None,
    val_fasta_file=None,
    prottrans_output_path=None,
    esm2_output_path=None,
    pdb_dir=None,
    proc_pdb_output_path=None,
    dssp_output_path=None,
    num_workers=None,
    disable_prottrans=False,
    disable_esm2=False,
    disable_dssp=False,
    disable_dihedrals=False,
    disable_orientations=False,
    disable_sidechains=False,
    disable_rbf=False,
    disable_pos_embeddings=False,
    disable_edge_direction=False,
    plddt_geometry_downweight=False,
    plddt_dir=None,
    plddt_weight_mode="hard",
    plddt_thresh=50.0,
    plddt_soft_lo=40.0,
    plddt_soft_hi=60.0,
    plddt_edge_mode="scale",
    plddt_pool_weight=False,
    plddt_knn_keep_min=0.5,
    k_folds=5,
    use_kfold=True,
    gpu_ids=None,
    fold_membership_tsv=None,
):
    """
    Train the model for binary aptamer affinity prediction.
    
    Args:
        config: Configuration object
        suffix: Suffix for saved models
        positive_list_file: Path to file listing proteins with known
            high-affinity aptamers
        train_fasta_file: Path to training FASTA file (optional, uses default if None)
        val_fasta_file: Path to validation FASTA file (optional, uses default if None)
        prottrans_output_path: Path to ProtTrans embeddings (optional,
            uses default if None)
        esm2_output_path: Path to ESM2 embeddings (optional, uses default if None)
        pdb_dir: Path to PDB directory (optional, uses default if None)
        proc_pdb_output_path: Path to processed PDB data (optional, uses default if None)
        dssp_output_path: Path to DSSP features (optional, uses default if None)
        num_workers: Number of data loader workers (optional, uses default if None)
        k_folds: Number of folds for cross-validation (default: 5)
        use_kfold: Whether to use K-fold cross-validation (default: True)
        gpu_ids: Comma-separated string of GPU IDs to use
            (e.g., "0,1,2,3") or None to use all
        fold_membership_tsv: Optional long-format membership TSV. When set,
            folds are taken from that file instead of fragment-level
            StratifiedKFold. Use this for cluster- or accession-grouped CV.
    """
    # Set GPU visibility before initializing distributed training
    # This must be done before PyTorch initializes CUDA
    if gpu_ids is not None:
        set_gpu_ids(gpu_ids)

    # ------------------------------------------------------------------
    # Simplified (single-process) execution
    # ------------------------------------------------------------------
    # Enforce single-process contrastive mode as well (no cross-process all-gather).
    if getattr(config, "contrast_world_size", 1) != 1:
        raise ValueError(
            "contrast_world_size > 1 requires multi-process support, which is not "
            "enabled in this script. Please set --contrast_world_size=1."
        )

    rank, world_size, local_rank = 0, 1, 0
    is_main_process = True
    
    # Log GPU selection after initialization
    if gpu_ids is not None and is_main_process:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        logging.info(
            "Restricted GPUs to: %s (CUDA_VISIBLE_DEVICES=%s)",
            gpu_ids, cuda_visible
        )

    configure_determinism(
        enabled=bool(getattr(config, "deterministic", True)),
        warn_only=bool(getattr(config, "deterministic_warn_only", True)),
    )
    set_all_seeds(int(getattr(config, "seed_base", 1234)))
    if is_main_process:
        logging.info(
            "Determinism settings: enabled=%s, warn_only=%s, seed_base=%d, CUBLAS_WORKSPACE_CONFIG=%s",
            bool(getattr(config, "deterministic", True)),
            bool(getattr(config, "deterministic_warn_only", True)),
            int(getattr(config, "seed_base", 1234)),
            os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
        )
    
    # Set device based on local rank for distributed training
    # If GPU IDs were specified, the device index should be relative to
    # visible GPUs (CUDA_VISIBLE_DEVICES re-indexes them starting from 0).
    if torch.cuda.is_available() and "cuda" in config.device:
        # Extract device index from config.device (e.g., "cuda:1" -> 1)
        device_idx = int(config.device.split(":")[-1]) if ":" in config.device else 0
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(config.device)
    
    config.device = str(device)
    
    if is_main_process:
        if torch.cuda.is_available():
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
            logging.info(
                "Using device: %s (CUDA_VISIBLE_DEVICES=%s)",
                device, cuda_visible
            )
        else:
            logging.info("Using device: %s", device)
    
    # Load binary labels from aptamer list
    positive_set = load_positive_proteins(positive_list_file)
    if is_main_process:
        logging.info(
            "Loaded %d proteins with known high-affinity aptamers", len(positive_set)
        )

    # Load sequence dictionaries from FASTA files
    args_train_data = get_arg_train(
        fasta_file=train_fasta_file,
        prottrans_output_path=prottrans_output_path,
        esm2_output_path=esm2_output_path,
        pdb_dir=pdb_dir,
        proc_pdb_output_path=proc_pdb_output_path,
        dssp_output_path=dssp_output_path,
        num_workers=num_workers,
        disable_prottrans=disable_prottrans,
        disable_esm2=disable_esm2,
        disable_dssp=disable_dssp,
        disable_dihedrals=disable_dihedrals,
        disable_orientations=disable_orientations,
        disable_sidechains=disable_sidechains,
        disable_rbf=disable_rbf,
        disable_pos_embeddings=disable_pos_embeddings,
        disable_edge_direction=disable_edge_direction,
        plddt_geometry_downweight=plddt_geometry_downweight,
        plddt_dir=plddt_dir,
        plddt_weight_mode=plddt_weight_mode,
        plddt_thresh=plddt_thresh,
        plddt_soft_lo=plddt_soft_lo,
        plddt_soft_hi=plddt_soft_hi,
        plddt_edge_mode=plddt_edge_mode,
        plddt_pool_weight=plddt_pool_weight,
        plddt_knn_keep_min=plddt_knn_keep_min,
    )
    args_val_data = get_arg_val(
        fasta_file=val_fasta_file,
        prottrans_output_path=prottrans_output_path,
        esm2_output_path=esm2_output_path,
        pdb_dir=pdb_dir,
        proc_pdb_output_path=proc_pdb_output_path,
        dssp_output_path=dssp_output_path,
        num_workers=num_workers,
        disable_prottrans=disable_prottrans,
        disable_esm2=disable_esm2,
        disable_dssp=disable_dssp,
        disable_dihedrals=disable_dihedrals,
        disable_orientations=disable_orientations,
        disable_sidechains=disable_sidechains,
        disable_rbf=disable_rbf,
        disable_pos_embeddings=disable_pos_embeddings,
        disable_edge_direction=disable_edge_direction,
        plddt_geometry_downweight=plddt_geometry_downweight,
        plddt_dir=plddt_dir,
        plddt_weight_mode=plddt_weight_mode,
        plddt_thresh=plddt_thresh,
        plddt_soft_lo=plddt_soft_lo,
        plddt_soft_hi=plddt_soft_hi,
        plddt_edge_mode=plddt_edge_mode,
        plddt_pool_weight=plddt_pool_weight,
        plddt_knn_keep_min=plddt_knn_keep_min,
    )

    train_fasta_data = extract_dataset(args_train_data.fasta_file)
    if is_main_process:
        logging.info("train_fasta_data len: %d", len(train_fasta_data))
    
    # Handle validation set (only needed for traditional train/val split)
    if use_kfold:
        if is_main_process:
            logging.info("Using K-fold cross-validation with %d folds", k_folds)
        val_fasta_data = None
        val_data = None
    else:
        if is_main_process:
            logging.info("Using traditional train/val split")
        val_fasta_data = extract_dataset(args_val_data.fasta_file)
        val_data = ProteinGraphDataset(
            val_fasta_data,
            range(len(val_fasta_data)),
            args_val_data,
            task_list="bp",
        )

    # Build full training dataset (will be split into folds if using K-fold CV)
    full_train_data = ProteinGraphDataset(
        train_fasta_data,
        range(len(train_fasta_data)),
        args_train_data,
        task_list="bp",  # dummy task; labels are ignored and replaced by binary ones
    )
    
    # Filter out positive samples with multiple fragments to avoid label ambiguity
    # (if a UniProt ID has multiple fragments, some fragments should be positive
    # but others negative, which is contradictory). In addition to removing these
    # UniProt IDs from the positive set, we now also remove ALL of their fragments
    # from the training dataset itself so they are not treated as negatives.
    positive_set_filtered, multi_fragment_excluded = _filter_multi_fragment_positives(
        positive_set, full_train_data.IDs
    )
    positive_set = positive_set_filtered
    if multi_fragment_excluded:
        # Rebuild the underlying FASTA data and dataset without any sequences
        # belonging to multi-fragment positive UniProt IDs.
        filtered_ids = [
            protein_id
            for protein_id in full_train_data.IDs
            if _normalize_to_uniprot(protein_id) not in multi_fragment_excluded
        ]
        filtered_train_fasta_data = {
            protein_id: train_fasta_data[protein_id]
            for protein_id in filtered_ids
        }
        full_train_data = ProteinGraphDataset(
            filtered_train_fasta_data,
            range(len(filtered_train_fasta_data)),
            args_train_data,
            task_list="bp",
        )
    
    if is_main_process:
        if multi_fragment_excluded:
            excluded_list = sorted(list(multi_fragment_excluded))
            logging.info(
                "Excluded %d multi-fragment positive UniProt IDs "
                "and removed their fragments from training data:",
                len(multi_fragment_excluded)
            )
            for uniprot_id in excluded_list:
                logging.info("  - %s", uniprot_id)
        else:
            logging.info(
                "No multi-fragment positive samples found "
                "(all positives have single fragment or no fragment pattern)"
            )
        logging.info(
            "After filtering: %d proteins with known high-affinity aptamers",
            len(positive_set)
        )

    # Model now outputs a single logit per protein (binary high-affinity label)
    output_dim = 1

    # Calculate class weights for imbalanced dataset and store per-sample labels
    num_positives = 0
    num_negatives = 0
    all_labels = []
    for protein_id in full_train_data.IDs:
        normalized_id = _normalize_to_uniprot(protein_id)
        if normalized_id in positive_set:
            num_positives += 1
            all_labels.append(1)
        else:
            num_negatives += 1
            all_labels.append(0)
    all_labels = np.array(all_labels, dtype=np.int64)
    
    # Calculate pos_weight for BCELoss: weight for positive class
    # pos_weight = num_negatives / num_positives
    # Allow manual override via config
    if hasattr(config, 'pos_weight') and config.pos_weight is not None:
        pos_weight = config.pos_weight
        if is_main_process:
            logging.info("Using manual pos_weight=%.2f (FocalLoss will handle class imbalance)", pos_weight)
    else:
        if num_positives > 0:
            pos_weight = num_negatives / num_positives
        else:
            pos_weight = 1.0  # Fallback if no positives found
        
        if is_main_process:
            logging.info(
                "Class distribution: %d positives, %d negatives (ratio: %.2f:1)",
                num_positives, num_negatives, pos_weight
            )
            logging.info("Using auto-calculated pos_weight=%.2f (FocalLoss will handle class imbalance)", pos_weight)
    
    # Classification loss (focal/asymmetric_focal/focal_tversky).
    focal_loss = build_classification_loss(config)
    if is_main_process:
        logging.info("Using loss_type=%s", str(getattr(config, "loss_type", "focal")))
        if getattr(config, "contrast_multi_view", False):
            logging.info(
                "Multi-view contrast enabled: feature_mask_prob=%.4f, feature_mask_value=%.4f, augment_eps_view2=%s",
                float(getattr(config, "feature_mask_prob", 0.0)),
                float(getattr(config, "feature_mask_value", 0.0)),
                str(getattr(config, "augment_eps_view2", None)),
            )
        if getattr(config, "ema_enabled", False) or getattr(config, "swa_enabled", False):
            logging.info(
                "Weight averaging: EMA(enabled=%s, decay=%.6f, eval=%s), SWA(enabled=%s, start_epoch=%d, eval=%s)",
                str(getattr(config, "ema_enabled", False)),
                float(getattr(config, "ema_decay", 0.999)),
                str(getattr(config, "eval_use_ema", True)),
                str(getattr(config, "swa_enabled", False)),
                int(getattr(config, "swa_start_epoch", 70)),
                str(getattr(config, "eval_use_swa", False)),
            )
    # Keep pos_weight for potential use in alpha if needed (currently using default alpha=1.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    if use_kfold:
        # K-fold cross-validation. Default is fragment-level StratifiedKFold.
        # If fold_membership_tsv is provided, use those precomputed splits
        # (cluster- or accession-grouped CV).
        fold_results = []
        train_indices = list(range(len(full_train_data)))
        if fold_membership_tsv:
            from grouped_cv_lib import load_kfold_index_splits

            membership_path = Path(fold_membership_tsv).expanduser()
            if not membership_path.is_absolute():
                membership_path = (Path.cwd() / membership_path).resolve()
            else:
                membership_path = membership_path.resolve()
            fold_splits = load_kfold_index_splits(
                membership_path,
                list(full_train_data.IDs),
                k_folds,
            )
            if is_main_process:
                logging.info(
                    "Using precomputed fold membership (%d folds): %s",
                    k_folds,
                    membership_path,
                )
        else:
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_splits = list(skf.split(train_indices, all_labels))
            if is_main_process:
                logging.info(
                    "Using fragment-level StratifiedKFold(n_splits=%d, shuffle=True, random_state=42)",
                    k_folds,
                )

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            if is_main_process:
                logging.info("=" * 80)
                logging.info("Fold %d/%d", fold_idx + 1, k_folds)
                logging.info("=" * 80)
            fold_seed = int(getattr(config, "seed_base", 1234)) + fold_idx
            set_all_seeds(fold_seed)
            train_generator = torch.Generator()
            train_generator.manual_seed(fold_seed)
            train_worker_init_fn = make_worker_init_fn(fold_seed)
            val_worker_init_fn = make_worker_init_fn(fold_seed + 100000)
            
            # Create fold-specific datasets
            train_subset = torch.utils.data.Subset(full_train_data, train_idx)
            
            # Compute train labels for this fold
            train_labels = all_labels[train_idx]
            num_positives_train = int(train_labels.sum())
            num_negatives_train = len(train_labels) - num_positives_train
            
            val_subset = torch.utils.data.Subset(full_train_data, val_idx)

            # Single-process: use custom stratified batch sampler so that each batch
            # has a fixed number of positives (k_pos) and negatives (B - k_pos),
            # with sampling done with or without replacement (controlled by
            # --stratified_without_replacement). Or use natural class distribution
            # if --use_stratified_sampling=False.
            train_sampler = None
            val_sampler = None
            batch_size = config.batch_size

            # Check if stratified sampling is enabled
            if not config.use_stratified_sampling:
                # Use natural class distribution (no balance enforcement)
                if is_main_process:
                    logging.info(
                        "Using natural class distribution sampling (no balance): fold %d, batch_size=%d, "
                        "%d positives, %d negatives",
                        fold_idx + 1,
                        batch_size,
                        num_positives_train,
                        num_negatives_train,
                    )
                train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args_train_data.num_workers,
                    prefetch_factor=1,
                    generator=train_generator,
                    worker_init_fn=train_worker_init_fn,
                )
            elif num_positives_train > 0 and num_negatives_train > 0:
                pos_indices_fold = [idx for idx in train_idx if all_labels[idx] == 1]
                neg_indices_fold = [idx for idx in train_idx if all_labels[idx] == 0]
                # Choose k_pos based on class ratio and batch size, but ensure >=1 and <= batch_size-1.
                # Allow manual override for stratified (non-balanced-epoch) sampling sweeps.
                if (not config.balanced_epoch_sampling) and (config.k_pos_override is not None):
                    k_pos = int(config.k_pos_override)
                else:
                    pos_frac = num_positives_train / float(len(train_labels))
                    k_pos = int(round(pos_frac * batch_size))
                    if k_pos <= 0:
                        k_pos = 1
                    if k_pos >= batch_size:
                        k_pos = batch_size - 1
                num_batches = int(math.ceil(len(train_idx) / float(batch_size)))
                if is_main_process:
                    if config.balanced_epoch_sampling:
                        logging.info(
                            "Using StratifiedPosNegBatchSampler with balanced epoch: fold %d, batch_size=%d, "
                            "balanced_epoch=True (50/50 per batch, equal counts per epoch), "
                            "%d positives, %d negatives, num_batches=%d",
                            fold_idx + 1,
                            batch_size,
                            num_positives_train,
                            num_negatives_train,
                            num_batches,
                        )
                    else:
                        logging.info(
                            "Using StratifiedPosNegBatchSampler: fold %d, batch_size=%d, k_pos=%d, replacement=%s, "
                            "%d positives, %d negatives, num_batches=%d",
                            fold_idx + 1,
                            batch_size,
                            k_pos,
                            "without" if config.stratified_without_replacement else "with",
                            num_positives_train,
                            num_negatives_train,
                            num_batches,
                        )
                        if config.k_pos_override is not None:
                            logging.info(
                                "k_pos override active: k_pos_override=%d (balanced_epoch_sampling=False required)",
                                int(config.k_pos_override),
                            )
                stratified_batch_sampler = StratifiedPosNegBatchSampler(
                    pos_indices_fold,
                    neg_indices_fold,
                    batch_size=batch_size,
                    k_pos=k_pos,
                    num_batches=num_batches,
                    shuffle=True,
                    with_replacement=not config.stratified_without_replacement,
                    stop_when_exhausted=config.stop_epoch_when_exhausted,
                    balanced_epoch=config.balanced_epoch_sampling,
                    negative_rounds=config.negative_rounds,
                    rotation_seed=config.negative_rotation_seed + fold_idx,
                )
                train_loader = DataLoader(
                    full_train_data,
                    batch_sampler=stratified_batch_sampler,
                    num_workers=args_train_data.num_workers,
                    prefetch_factor=1,
                    worker_init_fn=train_worker_init_fn,
                )
            else:
                # Fallback: regular shuffled DataLoader when stratification is impossible
                if is_main_process:
                    logging.warning(
                        "StratifiedPosNegBatchSampler not used for fold %d "
                        "(num_positives_train=%d, num_negatives_train=%d); "
                        "falling back to standard shuffled DataLoader.",
                        fold_idx + 1,
                        num_positives_train,
                        num_negatives_train,
                    )
                train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args_train_data.num_workers,
                    prefetch_factor=1,
                    generator=train_generator,
                    worker_init_fn=train_worker_init_fn,
                )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=val_sampler,
                drop_last=False,
                num_workers=args_train_data.num_workers,
                prefetch_factor=1,
                worker_init_fn=val_worker_init_fn,
            )

            # Initialize a fresh model for each fold (single-process)
            if is_main_process:
                logging.info("[rank0] Creating new model for fold %d...", fold_idx + 1)

            model = build_model(config, output_dim, device)

            optimizer = build_optimizer(config, model)
            
            # Initialize learning-rate scheduler.
            scheduler = None
            schedule_type = str(getattr(config, "lr_schedule_type", "plateau")).lower()
            scheduler_update_mode = "epoch"
            if schedule_type == "plateau":
                if config.scheduler_factor is None:
                    raise ValueError(
                        "lr_schedule_type='plateau' requires --scheduler_factor."
                    )
                if config.scheduler_factor > 1.0:
                    raise ValueError(
                        "scheduler_factor > 1 is not supported; use 1.0 for no decay."
                    )
                if config.scheduler_factor < 1.0:
                    scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode='max',  # maximize recall
                        factor=config.scheduler_factor,
                        patience=config.scheduler_patience,
                        min_lr=1e-6
                    )
                    if is_main_process:
                        logging.info(
                            f"Using ReduceLROnPlateau scheduler with factor={config.scheduler_factor}, "
                            f"patience={config.scheduler_patience}, mode='max' (monitoring recall)"
                        )
                else:
                    if is_main_process:
                        logging.info(
                            "Using constant LR (scheduler_factor=1.0, no decay)."
                        )
            elif schedule_type == "exponential":
                scheduler = ExponentialLR(optimizer, gamma=config.exp_gamma)
                if is_main_process:
                    logging.info(
                        "Using ExponentialLR scheduler with gamma=%.4f",
                        config.exp_gamma,
                    )
            elif schedule_type == "cosine":
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(config.max_epochs)),
                    eta_min=config.cosine_min_lr,
                )
                if is_main_process:
                    logging.info(
                        "Using CosineAnnealingLR scheduler with T_max=%d, eta_min=%.2e",
                        config.max_epochs,
                        config.cosine_min_lr,
                    )
            elif schedule_type == "onecycle":
                onecycle_max_lr = (
                    config.onecycle_max_lr
                    if config.onecycle_max_lr is not None
                    else config.optimizer["lr"]
                )
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=onecycle_max_lr,
                    epochs=max(1, int(config.max_epochs)),
                    steps_per_epoch=max(1, len(train_loader)),
                    pct_start=config.onecycle_pct_start,
                    anneal_strategy="cos",
                    div_factor=config.onecycle_div_factor,
                    final_div_factor=config.onecycle_final_div_factor,
                )
                scheduler_update_mode = "batch"
                if is_main_process:
                    logging.info(
                        "Using OneCycleLR scheduler with max_lr=%.2e, pct_start=%.3f, div_factor=%.1f, final_div_factor=%.1f",
                        onecycle_max_lr,
                        config.onecycle_pct_start,
                        config.onecycle_div_factor,
                        config.onecycle_final_div_factor,
                    )
            elif schedule_type == "constant":
                if is_main_process:
                    logging.info("Using constant LR (lr_schedule_type=constant).")
            else:
                raise ValueError(
                    f"Unsupported --lr_schedule_type={schedule_type}. "
                    "Choose from: plateau, exponential, cosine, onecycle, constant."
                )
            
            # Initialize GradScaler for mixed precision training (if enabled)
            scaler = build_scaler(config)
            if is_main_process and scaler is not None:
                logging.info("Using mixed precision training (torch.cuda.amp) to reduce memory usage")
            ema_state = None
            swa_state = None
            swa_count = 0

            # Pre-compute validation ground-truth labels once
            y_true_all = []
            for _, batch in enumerate(val_loader):
                labels = labels_from_batch_names(batch.name, positive_set, device)
                y_true_all.append(labels)
            y_true_all = torch.cat(y_true_all, dim=0).reshape(-1).to(device).float()  # Ensure float32

            # Initialize training tracking variables for this fold
            train_loss = []  # Per-batch total losses (for detailed tracking)
            train_loss_per_epoch = []  # Average total loss per epoch (for visualization)
            train_cls_loss_per_epoch = []  # Classification (Focal/BCE) loss per epoch
            train_cl_loss_per_epoch = []   # Contrastive loss per epoch (0 if not used)
            val_bce_per_epoch = []  # Validation classification loss per epoch (for visualization)
            val_aupr = []  # Validation AUPR per epoch
            val_precision_per_epoch = []  # Validation precision per epoch
            val_recall_per_epoch = []     # Validation recall per epoch
            val_f1_per_epoch = []         # Validation F1 per epoch
            val_auroc = []  # Validation AUROC per epoch
            es = 0
            best_eval_loss = None  # Initialize before epoch loop
            best_eval_aupr = None  # Initialize best AUPR for early stopping
            best_eval_recall = None  # Initialize best recall for early stopping
            best_eval_auroc = None  # Initialize best AUROC for early stopping
            saved_metrics = False  # Track whether we've already saved per-epoch metrics for this fold
            track_ef10 = bool(getattr(config, "track_ef10_checkpoint", False))
            val_ef10 = []
            val_ef10_hits = []
            ef_policy = HitPolicy(patience=int(config.early_stopping_patience))
            aupr_policy_frozen = False
            aupr_saved_epoch = None

            for ith_epoch in range(config.max_epochs):
                set_all_seeds(fold_seed + ith_epoch)
                # Synchronize all ranks at the start of each epoch (only log, no barrier to avoid deadlock)
                if is_main_process:
                    logging.info(f"[rank0] Fold {fold_idx+1}, Epoch {ith_epoch+1}/{config.max_epochs} starting")
                epoch_dropout = compute_dropout_for_epoch(config, ith_epoch)
                model.set_dropout_rate(epoch_dropout)
                if is_main_process and getattr(config, "dropout_schedule", "none") != "none":
                    logging.info(
                        "[rank0] Fold %d Epoch %d dropout schedule -> p=%.4f",
                        fold_idx + 1,
                        ith_epoch + 1,
                        epoch_dropout,
                    )
                
                # Track losses for this epoch
                epoch_total_losses = []
                epoch_cls_losses = []
                epoch_cl_losses = []
            
                # FIXED: Validate that train_loader has batches before processing
                # If train_loader is empty, this is a critical error that should be caught
                num_batches_processed = 0
                for idx_batch, batch in enumerate(train_loader):
                    batch = batch.to(device)
                    model.train()

                    # V: vertex/node features (scalar, vector)
                    h_V = (batch.node_s, batch.node_v)
                    # E: edge features (scalar, vector)
                    h_E = (batch.edge_s, batch.edge_v)
                    # Training-time edge dropout (augment_eps = keep probability)
                    edge_index_use = batch.edge_index
                    if config.augment_eps < 1.0:
                        edge_index_use, es_use, ev_use = apply_edge_dropout(
                            batch.edge_index, batch.edge_s, batch.edge_v, config.augment_eps, device
                        )
                        h_E = (es_use, ev_use)

                    # Build binary labels from protein IDs
                    y_true = labels_from_batch_names(batch.name, positive_set, device)

                    with get_autocast_context(scaler):
                        y_pred_raw = model(h_V, edge_index_use, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch))
                    if isinstance(y_pred_raw, tuple):
                        y_pred = y_pred_raw[0]
                    else:
                        y_pred = y_pred_raw
                    y_pred = y_pred.reshape(-1).float()
                    _loss = compute_supervised_loss(
                        focal_loss,
                        y_pred,
                        y_true,
                        pos_weight_tensor,
                        use_pos_weight_scale=bool(getattr(config, "loss_use_pos_weight_scale", True)),
                    )

                    if config.contrast:
                        if getattr(config, "contrast_multi_view", False):
                            # Explicit multi-view contrastive setup:
                            # view1 = edge-dropout view, view2 = edge-dropout + feature-masked view.
                            edge_keep_view2 = config.augment_eps
                            if getattr(config, "augment_eps_view2", None) is not None:
                                edge_keep_view2 = config.augment_eps_view2
                            edge_index_v2, es_v2, ev_v2 = apply_edge_dropout(
                                batch.edge_index,
                                batch.edge_s,
                                batch.edge_v,
                                edge_keep_view2,
                                device,
                            )
                            node_s_v2 = apply_feature_mask(
                                batch.node_s,
                                getattr(config, "feature_mask_prob", 0.0),
                                getattr(config, "feature_mask_value", 0.0),
                            )
                            h_V_v2 = (node_s_v2, batch.node_v)
                            h_E_v2 = (es_v2, ev_v2)
                            with get_autocast_context(scaler):
                                g_feat1 = model.encode_graph(
                                    h_V, edge_index_use, h_E, batch.seq, batch=batch.batch, perturbed=False, node_weight=_plddt_pool_weight(model, batch)
                                )
                                g_feat2 = model.encode_graph(
                                    h_V_v2, edge_index_v2, h_E_v2, batch.seq, batch=batch.batch, perturbed=False, node_weight=_plddt_pool_weight(model, batch)
                                )
                        else:
                            if isinstance(y_pred_raw, tuple):
                                g_feat1, g_feat2 = y_pred_raw[1], y_pred_raw[2]
                            else:
                                with get_autocast_context(scaler):
                                    g_feat1 = model.encode_graph(
                                        h_V, edge_index_use, h_E, batch.seq, batch=batch.batch, perturbed=False, node_weight=_plddt_pool_weight(model, batch)
                                    )
                                    g_feat2 = model.encode_graph(
                                        h_V, edge_index_use, h_E, batch.seq, batch=batch.batch, perturbed=True, node_weight=_plddt_pool_weight(model, batch)
                                    )

                        criterion = NT_Xent(
                            g_feat1.shape[0],
                            config.contrast_temp,
                            1,
                        )
                        cl_loss_raw = criterion(g_feat1, g_feat2)
                        cl_loss = config.contrast_weight * cl_loss_raw
                        loss = _loss + cl_loss
                    else:
                        cl_loss = torch.tensor(0.0, device=device)
                        cl_loss_raw = torch.tensor(0.0, device=device)
                        loss = _loss

                    if is_main_process:
                        if config.contrast:
                            cl_loss_val = cl_loss.detach().item()
                            cl_loss_raw_val = cl_loss_raw.detach().item()
                            logging.info(
                                "%d/%d train_epoch ||| Total: %.3f | Cls: %.3f | CL (raw): %.3f | CL (weighted, w=%.2f): %.3f",
                                idx_batch,                 # batch index within epoch
                                ith_epoch,                 # current epoch index
                                round(loss.detach().item(), 3),     # total loss
                                round(_loss.detach().item(), 3),    # classification (focal) loss
                                round(cl_loss_raw_val, 3),         # raw contrastive loss
                                config.contrast_weight,             # contrast weight
                                round(cl_loss_val, 3),              # weighted contrastive loss
                            )
                        else:
                            logging.info(
                                "%d/%d train_epoch ||| Total: %.3f | Cls: %.3f | CL: %.3f",
                                idx_batch,                 # batch index within epoch
                                ith_epoch,                 # current epoch index
                                round(loss.detach().item(), 3),     # total loss
                                round(_loss.detach().item(), 3),    # classification (focal) loss
                                round(cl_loss.detach().item(), 3),  # contrastive loss (0.0 if not used)
                            )
                    # Track loss values for plotting / checkpoint analysis
                    # (store detached CPU/NumPy copies so they don't keep graph references)
                    loss_val = loss.clone().detach().cpu().numpy()
                    cls_loss_val = _loss.clone().detach().cpu().numpy()
                    cl_loss_val = cl_loss.clone().detach().cpu().numpy()
                    train_loss.append(loss_val)
                    epoch_total_losses.append(loss_val)
                    epoch_cls_losses.append(cls_loss_val)
                    epoch_cl_losses.append(cl_loss_val)
                    num_batches_processed += 1

                    # Gradient-based parameter update step
                    optimizer.zero_grad()   # clear old gradients
                    
                    # Use scaler for mixed precision if enabled
                    try:
                        if scaler is not None:
                            scaler.scale(loss).backward()  # scale loss and backward
                            scaler.unscale_(optimizer)  # FIXED: Unscale before gradient clipping
                            # FIXED: Add gradient clipping to prevent gradient explosion
                            if config.gradient_clip_max_norm > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_norm=config.gradient_clip_max_norm,
                                )
                            scaler.step(optimizer)         # step optimizer
                            scaler.update()                # update scaler
                            if scheduler is not None and scheduler_update_mode == "batch":
                                scheduler.step()
                        else:
                            loss.backward()         # compute new gradients via backprop
                            # FIXED: Add gradient clipping to prevent gradient explosion
                            if config.gradient_clip_max_norm > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_norm=config.gradient_clip_max_norm,
                                )
                            optimizer.step()        # update model parameters
                            if scheduler is not None and scheduler_update_mode == "batch":
                                scheduler.step()
                    except Exception as e:
                        # Log error with rank information
                        logging.error(
                            "[rank%s] Error during backward pass: Fold %s, Epoch %s, "
                            "Batch %s: %s",
                            rank, fold_idx + 1, ith_epoch + 1, idx_batch, e,
                        )
                        raise
            
                # FIXED: Validate that batches were processed - if not, this is a critical error
                if num_batches_processed == 0:
                    error_msg = (
                        f"[rank{rank}] CRITICAL ERROR: No batches processed in Fold {fold_idx+1}, "
                        f"Epoch {ith_epoch+1}! train_loader appears to be empty. "
                        f"This indicates a serious problem with data loading or distributed training setup."
                    )
                    logging.error(error_msg)
                    raise RuntimeError(error_msg)
            
                # Calculate average training losses for this epoch (for visualization)
                # epoch_total_losses should never be empty at this point due to validation above
                if not epoch_total_losses:
                    error_msg = (
                        f"[rank{rank}] CRITICAL ERROR: epoch_total_losses is empty after processing "
                        f"{num_batches_processed} batches in Fold {fold_idx+1}, Epoch {ith_epoch+1}! "
                        f"This should never happen."
                    )
                    logging.error(error_msg)
                    raise RuntimeError(error_msg)
            
                epoch_avg_total = np.mean([l.item() if hasattr(l, 'item') else float(l) for l in epoch_total_losses])
                epoch_avg_cls = np.mean([l.item() if hasattr(l, 'item') else float(l) for l in epoch_cls_losses])
                epoch_avg_cl = np.mean([l.item() if hasattr(l, 'item') else float(l) for l in epoch_cl_losses])
                train_loss_per_epoch.append(epoch_avg_total)
                train_cls_loss_per_epoch.append(epoch_avg_cls)
                train_cl_loss_per_epoch.append(epoch_avg_cl)

                if getattr(config, "ema_enabled", False):
                    ema_state = _update_ema_state(ema_state, model, float(config.ema_decay))
                if getattr(config, "swa_enabled", False) and ith_epoch >= int(getattr(config, "swa_start_epoch", 0)):
                    swa_state, swa_count = _update_swa_state(swa_state, model, swa_count)
                eval_backup_state, eval_weight_mode = maybe_swap_eval_weights(
                    model, config, ema_state, swa_state, ith_epoch
                )
                if is_main_process and eval_weight_mode != "base":
                    logging.info(
                        "[rank0] Fold %d Epoch %d validation with %s weights",
                        fold_idx + 1,
                        ith_epoch + 1,
                        eval_weight_mode.upper(),
                    )

                # ------------------------------------------------------------------
                # Validation
                # ------------------------------------------------------------------
                logging.info(
                    "[rank0] Starting validation for Fold %s, Epoch %s/%s",
                    fold_idx + 1, ith_epoch + 1, config.max_epochs,
                )
                model.eval()
                y_pred_all = []
                num_val_batches_processed = 0
                with torch.no_grad():
                    # Iterate over validation batches (no gradient computation)
                    for idx_batch, batch in enumerate(val_loader):
                        time.sleep(0.03)  # small sleep to avoid overwhelming logs / I/O
                        batch = batch.to(device)

                        # Build node (V) and edge (E) features for the GNN
                        h_V = (batch.node_s, batch.node_v)
                        h_E = (batch.edge_s, batch.edge_v)

                        # Forward pass: prediction only (no contrastive outputs used here)
                        # Use autocast for mixed precision if enabled (validation doesn't need scaler)
                        if config.contrast:
                            if scaler is not None:
                                with autocast(device_type='cuda'):
                                    y_pred, _, _ = model(
                                        h_V, batch.edge_index, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch)
                                    )
                            else:
                                y_pred, _, _ = model(
                                    h_V, batch.edge_index, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch)
                                )
                        else:
                            if scaler is not None:
                                if AUTOCAST_DEVICE_TYPE:
                                    with autocast(device_type='cuda'):
                                        y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch))
                                else:
                                    with autocast(device_type='cuda'):
                                        y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch))
                            else:
                                y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch))

                        y_pred_all.append(y_pred)
                        num_val_batches_processed += 1
                        if is_main_process:
                            logging.info("%d/%d val_epoch", idx_batch, ith_epoch)

                    # Validate that validation batches were processed
                    if num_val_batches_processed == 0:
                        error_msg = (
                            f"CRITICAL ERROR: No validation batches processed! val_loader appears to be empty. "
                            f"This indicates a serious problem with validation data loading."
                        )
                        logging.error(error_msg)
                        raise RuntimeError(error_msg)

                    if not y_pred_all:
                        error_msg = (
                            f"CRITICAL ERROR: y_pred_all is empty after processing "
                            f"{num_val_batches_processed} validation batches! "
                            f"This should never happen."
                        )
                        logging.error(error_msg)
                        raise RuntimeError(error_msg)

                    # Concatenate predictions from all batches into a flat tensor
                    y_pred_all = torch.cat(y_pred_all, dim=0).reshape(-1).to(device)
            
                    # Convert to float32 if using mixed precision (autocast outputs FP16)
                    # Always ensure both are float32 to avoid dtype mismatches
                    y_pred_all = y_pred_all.float()
                    y_true_all = y_true_all.float()  # Ensure consistent dtype

                    # Validation BCE over the whole validation set
                    # Compute loss outside autocast context to ensure float32
                    eval_loss = compute_supervised_loss(
                        focal_loss,
                        y_pred_all,
                        y_true_all,
                        pos_weight_tensor,
                        use_pos_weight_scale=bool(getattr(config, "loss_use_pos_weight_scale", True)),
                    )
                    val_bce_per_epoch.append(
                        float(eval_loss.item())
                        if hasattr(eval_loss, "item")
                        else float(eval_loss)
                    )

                    # Binary metrics over proteins (using NumPy for sklearn metrics)
                    y_true_np = y_true_all.cpu().numpy()
                    y_scores_np = y_pred_all.cpu().numpy()
                    y_pred_labels = (y_scores_np >= 0.5).astype(int)

                    aupr = average_precision_score(y_true_np, y_scores_np)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true_np, y_pred_labels, average="binary", zero_division=0
                    )
                    try:
                        auroc = float(roc_auc_score(y_true_np, y_scores_np))
                    except ValueError:
                        auroc = float("nan")
                    val_aupr.append(aupr)
                    val_precision_per_epoch.append(float(precision))
                    val_recall_per_epoch.append(float(recall))
                    val_f1_per_epoch.append(float(f1))
                    val_auroc.append(auroc)
                    if track_ef10:
                        ef_hits, ef_value, ef_k = ef10_stats(y_true_np, y_scores_np)
                        val_ef10.append(float(ef_value))
                        val_ef10_hits.append(
                            float(ef_hits) if ef_hits == ef_hits else float("nan")
                        )

                    if is_main_process:
                        logging.info(
                            "%d VAL_epoch ||| loss: %.3f ||| aupr: %.3f ||| auroc: %.3f ||| precision: %.3f ||| recall: %.3f ||| f1: %.3f",
                            ith_epoch,
                            round(float(eval_loss), 3),
                            round(float(aupr), 3),
                            round(float(auroc), 3),
                            round(float(precision), 3),
                            round(float(recall), 3),
                            round(float(f1), 3),
                        )
                        if track_ef10:
                            hit_now = val_ef10_hits[-1]
                            ef_now = val_ef10[-1]
                            hit_text = "nan" if hit_now != hit_now else str(int(hit_now))
                            logging.info(
                                "%d VAL_ef10 ||| hits: %s ||| ef10: %.3f ||| k: %d",
                                ith_epoch,
                                hit_text,
                                ef_now if ef_now == ef_now else float("nan"),
                                ef_k,
                            )

                    # Early stopping logic based on selected metric
                    # Initialize best metrics on first epoch for all ranks
                    if best_eval_aupr is None:
                        best_eval_aupr = aupr
                    if best_eval_recall is None:
                        best_eval_recall = recall
                    if best_eval_loss is None:
                        best_eval_loss = eval_loss
                    prev_best_eval_aupr = best_eval_aupr
                    prev_best_eval_recall = best_eval_recall
                    prev_best_eval_loss = best_eval_loss
                    prev_best_eval_auroc = best_eval_auroc
                    
                    # Update early stopping counter based on selected metric.
                    # A frozen AUPR policy keeps its checkpoint and counter.
                    metric_improved = False
                    if not aupr_policy_frozen:
                        if config.early_stopping_metric == "aupr":
                            if aupr > best_eval_aupr + config.early_stopping_min_delta:
                                best_eval_aupr = aupr
                                metric_improved = True
                        elif config.early_stopping_metric == "recall":
                            if recall > best_eval_recall + config.early_stopping_min_delta:
                                best_eval_recall = recall
                                metric_improved = True
                        elif config.early_stopping_metric == "auroc":
                            if (not np.isnan(auroc) and best_eval_auroc is not None
                                    and auroc > best_eval_auroc + config.early_stopping_min_delta):
                                best_eval_auroc = auroc
                                metric_improved = True
                        elif config.early_stopping_metric == "loss":
                            if eval_loss < best_eval_loss - config.early_stopping_min_delta:
                                best_eval_loss = eval_loss
                                metric_improved = True

                        if metric_improved:
                            es = 0
                        else:
                            es += 1
                    
                    # Also update other metrics for logging (even if not used for early stopping)
                    if aupr > best_eval_aupr:
                        best_eval_aupr = aupr
                    if recall > best_eval_recall:
                        best_eval_recall = recall
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                    if not np.isnan(auroc):
                        if best_eval_auroc is None:
                            best_eval_auroc = auroc
                        elif auroc > best_eval_auroc:
                            best_eval_auroc = auroc
                    
                    # Step epoch-based learning-rate scheduler (if enabled)
                    if scheduler is not None and scheduler_update_mode == "epoch":
                        if schedule_type == "plateau":
                            scheduler.step(recall)  # Monitor validation recall
                            if is_main_process:
                                current_lr = optimizer.param_groups[0]['lr']
                                logging.info(f"LR Scheduler step: current_lr={current_lr:.2e}, validation_recall={recall:.4f}")
                        else:
                            scheduler.step()
                            if is_main_process:
                                current_lr = optimizer.param_groups[0]['lr']
                                logging.info(
                                    "LR Scheduler step (%s): current_lr=%.2e",
                                    schedule_type,
                                    current_lr,
                                )
            
                    if is_main_process:
                        # FIXED: Save model based on early_stopping_metric, not eval_loss
                        # This ensures saved model is best for the metric you care about (recall/AUPR)
                        # Save first epoch checkpoint, then only when selected metric improves.
                        should_save = aupr_checkpoint_should_save(
                            ith_epoch, metric_improved, aupr_policy_frozen
                        )
                        
                        if should_save:
                            aupr_saved_epoch = ith_epoch
                            model_path = (
                                config.model_save_path
                                + "affinity"
                                + f"{suffix}_fold{fold_idx+1}_es.pt"
                            )
                            save_model_state_dict(
                                model, model_path, config, rank, world_size
                            )
                            # Log what metric value triggered the save
                            if config.early_stopping_metric == "recall":
                                logging.info(
                                    "Saved model at epoch %d: RECALL=%.4f (improved from %.4f)",
                                    ith_epoch, recall, prev_best_eval_recall if ith_epoch > 0 else 0.0
                                )
                            elif config.early_stopping_metric == "aupr":
                                logging.info(
                                    "Saved model at epoch %d: AUPR=%.4f (improved from %.4f)",
                                    ith_epoch, aupr, prev_best_eval_aupr if ith_epoch > 0 else 0.0
                                )
                            elif config.early_stopping_metric == "auroc":
                                logging.info(
                                    "Saved model at epoch %d: AUROC=%.4f (improved from %.4f)",
                                    ith_epoch, auroc, prev_best_eval_auroc if ith_epoch > 0 and prev_best_eval_auroc is not None else 0.0
                                )
                            else:
                                logging.info(
                                    "Saved model at epoch %d: LOSS=%.4f (improved from %.4f)",
                                    ith_epoch, eval_loss, prev_best_eval_loss if ith_epoch > 0 else float('inf')
                                )
                            # Flush logging to ensure save messages appear
                            logging.getLogger().handlers[0].flush() if logging.getLogger().handlers else None
                        else:
                            # Metric did not improve this epoch; only log counter.
                            # Do not save current model here (it would overwrite with a worse model).
                            if config.early_stopping_metric == "aupr":
                                metric_name = "validation AUPR"
                            elif config.early_stopping_metric == "recall":
                                metric_name = "validation recall"
                            elif config.early_stopping_metric == "auroc":
                                metric_name = "validation AUROC"
                            else:
                                metric_name = "validation loss"
                            logging.info(
                                "Early stopping counter %d of %d (%s did not improve)",
                                es, config.early_stopping_patience, metric_name
                            )
                        if track_ef10:
                            was_ef_frozen = ef_policy.frozen
                            if ef_policy.observe(ith_epoch, val_ef10_hits[-1]):
                                ef_path = (
                                    config.model_save_path
                                    + "affinity"
                                    + f"{suffix}_fold{fold_idx+1}_es_ef10.pt"
                                )
                                save_model_state_dict(
                                    model, ef_path, config, rank, world_size
                                )
                                logging.info(
                                    "Saved EF@10%% checkpoint at epoch %d: hits=%s",
                                    ith_epoch,
                                    val_ef10_hits[-1],
                                )
                            elif ef_policy.frozen and not was_ef_frozen:
                                logging.info(
                                    "EF@10%% checkpoint policy frozen at epoch %d "
                                    "(saved epoch %s, hits=%s)",
                                    ith_epoch,
                                    ef_policy.saved_epoch,
                                    ef_policy.best_hits,
                                )

                if eval_backup_state is not None:
                    model.load_state_dict(eval_backup_state, strict=True)

                # Stop when the active policy or policies have used up patience.
                if (not aupr_policy_frozen) and es >= config.early_stopping_patience:
                    aupr_policy_frozen = True
                    if is_main_process and track_ef10:
                        logging.info(
                            "AUPR checkpoint policy frozen at epoch %d (saved epoch %s)",
                            ith_epoch,
                            aupr_saved_epoch,
                        )
                if trajectory_should_stop(
                    track_ef10,
                    es,
                    config.early_stopping_patience,
                    aupr_policy_frozen,
                    ef_policy.frozen,
                ):
                    if is_main_process:
                        # Format best metric value for logging
                        if config.early_stopping_metric == "aupr":
                            best_metric_value = best_eval_aupr
                            metric_name = "AUPR"
                        elif config.early_stopping_metric == "recall":
                            best_metric_value = best_eval_recall
                            metric_name = "Recall"
                        elif config.early_stopping_metric == "auroc":
                            best_metric_value = best_eval_auroc
                            metric_name = "AUROC"
                        else:  # loss
                            best_metric_value = best_eval_loss
                            metric_name = "Loss"
                        
                        logging.info(
                            "Early stopping triggered after %d epochs without improvement "
                            "(patience=%d, metric=%s, best %s=%.4f, best AUPR=%.4f, "
                            "best Recall=%.4f, best AUROC=%.4f, best Loss=%.4f)",
                            es,
                            config.early_stopping_patience,
                            config.early_stopping_metric.upper(),
                            metric_name,
                            best_metric_value,
                            best_eval_aupr,
                            best_eval_recall,
                            best_eval_auroc if best_eval_auroc is not None else float("nan"),
                            best_eval_loss,
                        )
                        # Create directory for loss file
                        loss_path = (
                            config.loss_save_path
                            + "affinity"
                            + f"{suffix}_fold{fold_idx+1}_val.pt"
                        )
                        Path(loss_path).parent.mkdir(parents=True, exist_ok=True)
                        # FIXED: Use len(val_aupr) or len(val_bce_per_epoch) for epochs list
                        # instead of train_loss_per_epoch.
                        num_epochs = max(
                            len(val_aupr),
                            len(val_bce_per_epoch),
                            len(train_loss_per_epoch),
                        )
                        if num_epochs == 0:
                            num_epochs = config.max_epochs  # Fallback to max_epochs if all are empty
                        metrics_payload = {
                                "train_total_loss_per_batch": train_loss,  # Per-batch total losses
                                "train_total_loss_per_epoch": train_loss_per_epoch,  # Per-epoch total losses
                                "train_cls_loss_per_epoch": train_cls_loss_per_epoch,  # Per-epoch classification losses
                                "train_cl_loss_per_epoch": train_cl_loss_per_epoch,    # Per-epoch contrastive losses
                                "val_loss_per_epoch": val_bce_per_epoch,  # Per-epoch validation classification losses
                                "val_aupr": val_aupr,  # Per-epoch validation AUPR
                                "val_auroc": val_auroc,
                                "val_precision": val_precision_per_epoch,
                                "val_recall": val_recall_per_epoch,
                                "val_f1": val_f1_per_epoch,
                                "epochs": list(range(num_epochs)),  # FIXED: Use actual number of epochs from validation data
                            }
                        attach_ef10_curves(
                            metrics_payload,
                            config,
                            val_ef10,
                            val_ef10_hits,
                            ef_policy,
                            aupr_policy_frozen,
                            aupr_saved_epoch,
                        )
                        torch.save(metrics_payload, loss_path)
                        saved_metrics = True
                    break

            # Store fold results
            if is_main_process:
                best_fold_aupr = max(val_aupr) if val_aupr else 0.0
                best_fold_recall = max(val_recall_per_epoch) if val_recall_per_epoch else 0.0
                if val_auroc and not np.all(np.isnan(val_auroc)):
                    best_fold_auroc = float(np.nanmax(val_auroc))
                else:
                    best_fold_auroc = 0.0
                fold_results.append({
                    "fold": fold_idx + 1,
                    "best_aupr": best_fold_aupr,
                    "best_recall": best_fold_recall,
                    "best_auroc": best_fold_auroc,
                    "final_aupr": val_aupr[-1] if val_aupr else 0.0,
                    "best_loss": (
                        min([l.item() if hasattr(l, 'item') else l
                             for l in train_loss])
                        if train_loss else float('inf')
                    ),
                })
                logging.info(
                    "Fold %d/%d completed. Best AUPR: %.4f, Best Recall: %.4f, Best AUROC: %.4f",
                    fold_idx + 1, k_folds, best_fold_aupr, best_fold_recall, best_fold_auroc
                )
                
                # Save fold model
                model_path = (
                    config.model_save_path
                    + "affinity"
                    + f"{suffix}_fold{fold_idx+1}.pt"
                )
                save_model_state_dict(
                    model, model_path, config, rank, world_size
                )

                # If early stopping did not trigger (i.e., we finished all epochs),
                # ensure that per-epoch metrics are still saved.
                # FIXED: Save even if train_loss_per_epoch is empty (use validation data length)
                if not saved_metrics:
                    loss_path = config.loss_save_path + "affinity" + f"{suffix}_fold{fold_idx+1}_val.pt"
                    Path(loss_path).parent.mkdir(parents=True, exist_ok=True)
                    # FIXED: Use len(val_aupr) or len(val_bce_per_epoch) for epochs list instead of train_loss_per_epoch
                    num_epochs = max(len(val_aupr), len(val_bce_per_epoch), len(train_loss_per_epoch))
                    if num_epochs == 0:
                        num_epochs = config.max_epochs  # Fallback to max_epochs if all are empty
                    metrics_payload = {
                            "train_total_loss_per_batch": train_loss,  # Per-batch total losses
                            "train_total_loss_per_epoch": train_loss_per_epoch,  # Per-epoch total losses
                            "train_cls_loss_per_epoch": train_cls_loss_per_epoch,  # Per-epoch classification losses
                            "train_cl_loss_per_epoch": train_cl_loss_per_epoch,    # Per-epoch contrastive losses
                            "val_loss_per_epoch": val_bce_per_epoch,  # Per-epoch validation classification losses
                            "val_aupr": val_aupr,  # Per-epoch validation AUPR
                            "val_auroc": val_auroc,
                            "val_precision": val_precision_per_epoch,
                            "val_recall": val_recall_per_epoch,
                            "val_f1": val_f1_per_epoch,
                            "epochs": list(range(num_epochs)),  # FIXED: Use actual number of epochs from validation data
                        }
                    attach_ef10_curves(
                        metrics_payload,
                        config,
                        val_ef10,
                        val_ef10_hits,
                        ef_policy,
                        aupr_policy_frozen,
                        aupr_saved_epoch,
                    )
                    torch.save(metrics_payload, loss_path)
            
        # Aggregate results across folds
        if is_main_process:
            avg_aupr = sum([r["best_aupr"] for r in fold_results]) / len(fold_results)
            std_aupr = np.std([r["best_aupr"] for r in fold_results])
            avg_recall = sum([r["best_recall"] for r in fold_results]) / len(fold_results)
            std_recall = np.std([r["best_recall"] for r in fold_results])
            avg_auroc = sum([r["best_auroc"] for r in fold_results]) / len(fold_results)
            std_auroc = np.std([r["best_auroc"] for r in fold_results])
            logging.info("=" * 80)
            logging.info("K-fold CV Results:")
            logging.info("  Average Best AUPR: %.4f ± %.4f", avg_aupr, std_aupr)
            logging.info("  Average Best Recall: %.4f ± %.4f", avg_recall, std_recall)
            logging.info("  Average Best AUROC: %.4f ± %.4f", avg_auroc, std_auroc)
            for r in fold_results:
                logging.info(
                    "  Fold %d: Best AUPR = %.4f, Best Recall = %.4f, Best AUROC = %.4f",
                    r["fold"], r["best_aupr"], r["best_recall"], r["best_auroc"],
                )
            logging.info("=" * 80)
            
            # Save aggregated results
            # Create directory for loss file
            loss_path = config.loss_save_path + "affinity" + f"{suffix}_kfold_results.pt"
            Path(loss_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "fold_results": fold_results,
                    "avg_aupr": avg_aupr,
                    "std_aupr": std_aupr,
                    "avg_recall": avg_recall,
                    "std_recall": std_recall,
                    "avg_auroc": avg_auroc,
                    "std_auroc": std_auroc,
                },
                loss_path,
            )
    else:
        # Traditional train/val split (single-process)
        run_seed = int(getattr(config, "seed_base", 1234))
        set_all_seeds(run_seed)
        train_generator = torch.Generator()
        train_generator.manual_seed(run_seed)
        train_worker_init_fn = make_worker_init_fn(run_seed)
        val_worker_init_fn = make_worker_init_fn(run_seed + 100000)
        train_sampler = None
        val_sampler = None
        
        train_loader = DataLoader(
            full_train_data,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args_train_data.num_workers,
            prefetch_factor=1,
            generator=train_generator,
            worker_init_fn=train_worker_init_fn,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args_val_data.num_workers,
            prefetch_factor=1,
            worker_init_fn=val_worker_init_fn,
        )
        
        model = build_model(config, output_dim, device)

        optimizer = build_optimizer(config, model)
        
        # Initialize GradScaler for mixed precision training (if enabled)
        scaler = build_scaler(config)
        if is_main_process and scaler is not None:
            logging.info(
                "Using mixed precision training (torch.cuda.amp) to reduce memory usage"
            )
        ema_state = None
        swa_state = None
        swa_count = 0

        # Pre-compute validation ground-truth labels once
        y_true_all = []
        for _, batch in enumerate(val_loader):
            labels = labels_from_batch_names(batch.name, positive_set, device)
            y_true_all.append(labels)
        y_true_all = torch.cat(y_true_all, dim=0).reshape(-1).to(device)
        
        train_loss = []  # Per-batch total losses (for detailed tracking)
        train_loss_per_epoch = []  # Average total loss per epoch (for visualization)
        train_cls_loss_per_epoch = []  # Classification (Focal/BCE) loss per epoch
        train_cl_loss_per_epoch = []   # Contrastive loss per epoch (0 if not used)
        val_bce_per_epoch = []  # Validation classification loss per epoch (for visualization)
        val_aupr = []  # Validation AUPR per epoch
        val_precision_per_epoch = []  # Validation precision per epoch
        val_recall_per_epoch = []     # Validation recall per epoch
        val_f1_per_epoch = []         # Validation F1 per epoch
        val_auroc = []  # Validation AUROC per epoch
        es = 0
        best_eval_loss = None  # Initialize before epoch loop
        best_eval_aupr = None  # Initialize best AUPR for early stopping
        best_eval_recall = None  # Initialize best recall for early stopping
        best_eval_auroc = None  # Initialize best AUROC for early stopping
        saved_metrics = False  # Track whether we've already saved per-epoch metrics
        track_ef10 = bool(getattr(config, "track_ef10_checkpoint", False))
        val_ef10 = []
        val_ef10_hits = []
        ef_policy = HitPolicy(patience=int(config.early_stopping_patience))
        aupr_policy_frozen = False
        aupr_saved_epoch = None
        
        for ith_epoch in range(config.max_epochs):
            set_all_seeds(run_seed + ith_epoch)
            epoch_dropout = compute_dropout_for_epoch(config, ith_epoch)
            model.set_dropout_rate(epoch_dropout)
            if is_main_process and getattr(config, "dropout_schedule", "none") != "none":
                logging.info(
                    "Epoch %d/%d dropout schedule -> p=%.4f",
                    ith_epoch + 1,
                    config.max_epochs,
                    epoch_dropout,
                )
            # Track losses for this epoch
            epoch_total_losses = []
            epoch_cls_losses = []
            epoch_cl_losses = []
            
            # FIXED: Validate that train_loader has batches before processing
            # If train_loader is empty, this is a critical error that should be caught
            num_batches_processed = 0
            for idx_batch, batch in enumerate(train_loader):
                batch = batch.to(device)
                model.train()

                # V: vertex/node features (scalar, vector)
                h_V = (batch.node_s, batch.node_v)
                # E: edge features (scalar, vector)
                h_E = (batch.edge_s, batch.edge_v)
                # Training-time edge dropout (augment_eps = keep probability)
                edge_index_use = batch.edge_index
                if config.augment_eps < 1.0:
                    edge_index_use, es_use, ev_use = apply_edge_dropout(
                        batch.edge_index, batch.edge_s, batch.edge_v, config.augment_eps, device
                    )
                    h_E = (es_use, ev_use)

                # Build binary labels from protein IDs
                y_true = labels_from_batch_names(batch.name, positive_set, device)

                with get_autocast_context(scaler):
                    y_pred_raw = model(h_V, edge_index_use, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch))
                if isinstance(y_pred_raw, tuple):
                    y_pred = y_pred_raw[0]
                else:
                    y_pred = y_pred_raw
                y_pred = y_pred.reshape(-1).float()
                _loss = compute_supervised_loss(
                    focal_loss,
                    y_pred,
                    y_true,
                    pos_weight_tensor,
                    use_pos_weight_scale=bool(getattr(config, "loss_use_pos_weight_scale", True)),
                )

                if config.contrast:
                    if getattr(config, "contrast_multi_view", False):
                        edge_keep_view2 = config.augment_eps
                        if getattr(config, "augment_eps_view2", None) is not None:
                            edge_keep_view2 = config.augment_eps_view2
                        edge_index_v2, es_v2, ev_v2 = apply_edge_dropout(
                            batch.edge_index,
                            batch.edge_s,
                            batch.edge_v,
                            edge_keep_view2,
                            device,
                        )
                        node_s_v2 = apply_feature_mask(
                            batch.node_s,
                            getattr(config, "feature_mask_prob", 0.0),
                            getattr(config, "feature_mask_value", 0.0),
                        )
                        h_V_v2 = (node_s_v2, batch.node_v)
                        h_E_v2 = (es_v2, ev_v2)
                        with get_autocast_context(scaler):
                            g_feat1 = model.encode_graph(
                                h_V, edge_index_use, h_E, batch.seq, batch=batch.batch, perturbed=False, node_weight=_plddt_pool_weight(model, batch)
                            )
                            g_feat2 = model.encode_graph(
                                h_V_v2, edge_index_v2, h_E_v2, batch.seq, batch=batch.batch, perturbed=False, node_weight=_plddt_pool_weight(model, batch)
                            )
                    else:
                        if isinstance(y_pred_raw, tuple):
                            g_feat1, g_feat2 = y_pred_raw[1], y_pred_raw[2]
                        else:
                            with get_autocast_context(scaler):
                                g_feat1 = model.encode_graph(
                                    h_V, edge_index_use, h_E, batch.seq, batch=batch.batch, perturbed=False, node_weight=_plddt_pool_weight(model, batch)
                                )
                                g_feat2 = model.encode_graph(
                                    h_V, edge_index_use, h_E, batch.seq, batch=batch.batch, perturbed=True, node_weight=_plddt_pool_weight(model, batch)
                                )

                    criterion = NT_Xent(
                        g_feat1.shape[0],
                        config.contrast_temp,
                        1,
                    )
                    cl_loss_raw = criterion(g_feat1, g_feat2)
                    cl_loss = config.contrast_weight * cl_loss_raw
                    loss = _loss + cl_loss
                else:
                    cl_loss = torch.tensor(0.0, device=device)
                    cl_loss_raw = torch.tensor(0.0, device=device)
                    loss = _loss

                if is_main_process:
                    if config.contrast:
                        cl_loss_val = cl_loss.detach().item()
                        cl_loss_raw_val = cl_loss_raw.detach().item()
                        logging.info(
                            "%d/%d train_epoch ||| Total: %.3f | Cls: %.3f | CL (raw): %.3f | CL (weighted, w=%.2f): %.3f",
                            idx_batch,                 # batch index within epoch
                            ith_epoch,                 # current epoch index
                            round(loss.detach().item(), 3),     # total loss
                            round(_loss.detach().item(), 3),    # classification (focal) loss
                            round(cl_loss_raw_val, 3),         # raw contrastive loss
                            config.contrast_weight,             # contrast weight
                            round(cl_loss_val, 3),              # weighted contrastive loss
                        )
                    else:
                        logging.info(
                            "%d/%d train_epoch ||| Total: %.3f | Cls: %.3f | CL: %.3f",
                            idx_batch,                 # batch index within epoch
                            ith_epoch,                 # current epoch index
                            round(loss.detach().item(), 3),     # total loss
                            round(_loss.detach().item(), 3),    # classification (focal) loss
                            round(cl_loss.detach().item(), 3),  # contrastive loss (0.0 if not used)
                        )
                # Track loss values for plotting / checkpoint analysis
                # (store detached CPU/NumPy copies so they don't keep graph references)
                loss_val = loss.clone().detach().cpu().numpy()
                cls_loss_val = _loss.clone().detach().cpu().numpy()
                cl_loss_val = cl_loss.clone().detach().cpu().numpy()
                train_loss.append(loss_val)
                epoch_total_losses.append(loss_val)
                epoch_cls_losses.append(cls_loss_val)
                epoch_cl_losses.append(cl_loss_val)
                num_batches_processed += 1

                # Gradient-based parameter update step
                optimizer.zero_grad()   # clear old gradients
                # Use scaler for mixed precision if enabled
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if config.gradient_clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.gradient_clip_max_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None and scheduler_update_mode == "batch":
                        scheduler.step()
                else:
                    loss.backward()
                    if config.gradient_clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.gradient_clip_max_norm,
                        )
                    optimizer.step()
                    if scheduler is not None and scheduler_update_mode == "batch":
                        scheduler.step()

            # Validate that batches were processed
            if num_batches_processed == 0:
                error_msg = (
                    f"[rank{rank}] CRITICAL ERROR: No batches processed in Epoch {ith_epoch+1}! "
                    f"train_loader appears to be empty. "
                    f"This indicates a serious problem with data loading or distributed training setup."
                )
                logging.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Calculate average training losses for this epoch (after all batches)
            # epoch_total_losses should never be empty at this point due to validation above
            if not epoch_total_losses:
                error_msg = (
                    f"[rank{rank}] CRITICAL ERROR: epoch_total_losses is empty after processing "
                    f"{num_batches_processed} batches in Epoch {ith_epoch+1}! "
                    f"This should never happen."
                )
                logging.error(error_msg)
                raise RuntimeError(error_msg)
            
            epoch_avg_total = np.mean([l.item() if hasattr(l, 'item') else float(l) for l in epoch_total_losses])
            epoch_avg_cls = np.mean([l.item() if hasattr(l, 'item') else float(l) for l in epoch_cls_losses])
            epoch_avg_cl = np.mean([l.item() if hasattr(l, 'item') else float(l) for l in epoch_cl_losses])
            train_loss_per_epoch.append(epoch_avg_total)
            train_cls_loss_per_epoch.append(epoch_avg_cls)
            train_cl_loss_per_epoch.append(epoch_avg_cl)

            if getattr(config, "ema_enabled", False):
                ema_state = _update_ema_state(ema_state, model, float(config.ema_decay))
            if getattr(config, "swa_enabled", False) and ith_epoch >= int(getattr(config, "swa_start_epoch", 0)):
                swa_state, swa_count = _update_swa_state(swa_state, model, swa_count)
            eval_backup_state, eval_weight_mode = maybe_swap_eval_weights(
                model, config, ema_state, swa_state, ith_epoch
            )
            if is_main_process and eval_weight_mode != "base":
                logging.info(
                    "Epoch %d/%d validation with %s weights",
                    ith_epoch + 1,
                    config.max_epochs,
                    eval_weight_mode.upper(),
                )

            # ------------------------------------------------------------------
            # Validation
            # ------------------------------------------------------------------
            model.eval()
            y_pred_all = []
            num_val_batches_processed = 0
            with torch.no_grad():
                # Iterate over validation batches (no gradient computation)
                for idx_batch, batch in enumerate(val_loader):
                    time.sleep(0.03)  # small sleep to avoid overwhelming logs / I/O
                    batch = batch.to(device)

                    # Build node (V) and edge (E) features for the GNN
                    h_V = (batch.node_s, batch.node_v)
                    h_E = (batch.edge_s, batch.edge_v)

                    # Forward pass: prediction only (no contrastive outputs used here)
                    # Use autocast for mixed precision if enabled (validation doesn't need scaler)
                    if config.contrast:
                        if scaler is not None:
                            with autocast(device_type='cuda'):
                                y_pred, _, _ = model(
                                    h_V, batch.edge_index, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch)
                                )
                        else:
                            y_pred, _, _ = model(
                                h_V, batch.edge_index, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch)
                            )
                    else:
                        if scaler is not None:
                            if AUTOCAST_DEVICE_TYPE:
                                with autocast(device_type='cuda'):
                                    y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch))
                            else:
                                with autocast(device_type='cuda'):
                                    y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch))
                        else:
                            y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch, node_weight=_plddt_pool_weight(model, batch))

                    y_pred_all.append(y_pred)
                    num_val_batches_processed += 1
                    if is_main_process:
                        logging.info("%d/%d val_epoch", idx_batch, ith_epoch)

                # FIXED: Validate that validation batches were processed
                if num_val_batches_processed == 0:
                    error_msg = (
                        f"CRITICAL ERROR: No validation batches processed! val_loader appears to be empty. "
                        f"This indicates a serious problem with validation data loading."
                    )
                    logging.error(error_msg)
                    raise RuntimeError(error_msg)
                
                if not y_pred_all:
                    error_msg = (
                        f"CRITICAL ERROR: y_pred_all is empty after processing "
                        f"{num_val_batches_processed} validation batches! "
                        f"This should never happen."
                    )
                    logging.error(error_msg)
                    raise RuntimeError(error_msg)

                # Concatenate predictions from all batches into a flat tensor
                y_pred_all = torch.cat(y_pred_all, dim=0).reshape(-1).to(device)
                
                # Convert to float32 if using mixed precision (autocast outputs FP16)
                # Always ensure both are float32 to avoid dtype mismatches
                y_pred_all = y_pred_all.float()
                y_true_all = y_true_all.float()  # Ensure consistent dtype

                # Validation BCE over the whole validation set
                # Compute loss outside autocast context to ensure float32
                eval_loss = compute_supervised_loss(
                    focal_loss,
                    y_pred_all,
                    y_true_all,
                    pos_weight_tensor,
                    use_pos_weight_scale=bool(getattr(config, "loss_use_pos_weight_scale", True)),
                )
                val_bce_per_epoch.append(
                    float(eval_loss.item())
                    if hasattr(eval_loss, "item")
                    else float(eval_loss)
                )

                # Binary metrics over proteins (using NumPy for sklearn metrics)
                y_true_np = y_true_all.cpu().numpy()
                y_scores_np = y_pred_all.cpu().numpy()
                y_pred_labels = (y_scores_np >= 0.5).astype(int)

                aupr = average_precision_score(y_true_np, y_scores_np)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_np, y_pred_labels, average="binary", zero_division=0
                )
                try:
                    auroc = float(roc_auc_score(y_true_np, y_scores_np))
                except ValueError:
                    auroc = float("nan")

                val_aupr.append(aupr)
                val_precision_per_epoch.append(float(precision))
                val_recall_per_epoch.append(float(recall))
                val_f1_per_epoch.append(float(f1))
                val_auroc.append(auroc)
                if track_ef10:
                    ef_hits, ef_value, ef_k = ef10_stats(y_true_np, y_scores_np)
                    val_ef10.append(float(ef_value))
                    val_ef10_hits.append(
                        float(ef_hits) if ef_hits == ef_hits else float("nan")
                    )

                if is_main_process:
                    logging.info(
                        "%d VAL_epoch ||| loss: %.3f ||| aupr: %.3f ||| auroc: %.3f ||| precision: %.3f ||| recall: %.3f ||| f1: %.3f",
                        ith_epoch,
                        round(float(eval_loss), 3),
                        round(float(aupr), 3),
                        round(float(auroc), 3),
                        round(float(precision), 3),
                        round(float(recall), 3),
                        round(float(f1), 3),
                    )
                    if track_ef10:
                        hit_now = val_ef10_hits[-1]
                        ef_now = val_ef10[-1]
                        hit_text = "nan" if hit_now != hit_now else str(int(hit_now))
                        logging.info(
                            "%d VAL_ef10 ||| hits: %s ||| ef10: %.3f ||| k: %d",
                            ith_epoch,
                            hit_text,
                            ef_now if ef_now == ef_now else float("nan"),
                            ef_k,
                        )

                # Early stopping logic based on selected metric
                # Initialize best metrics on first epoch for all ranks
                if best_eval_aupr is None:
                    best_eval_aupr = aupr
                if best_eval_recall is None:
                    best_eval_recall = recall
                if best_eval_loss is None:
                    best_eval_loss = eval_loss
                
                # Update early stopping counter based on selected metric.
                # A frozen policy keeps its checkpoint and counter.
                metric_improved = False
                if not aupr_policy_frozen:
                    if config.early_stopping_metric == "aupr":
                        if aupr > best_eval_aupr:
                            best_eval_aupr = aupr
                            metric_improved = True
                    elif config.early_stopping_metric == "recall":
                        if recall > best_eval_recall:
                            best_eval_recall = recall
                            metric_improved = True
                    elif config.early_stopping_metric == "auroc":
                        if not np.isnan(auroc) and best_eval_auroc is not None and auroc > best_eval_auroc:
                            best_eval_auroc = auroc
                            metric_improved = True
                    elif config.early_stopping_metric == "loss":
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            metric_improved = True

                    if metric_improved:
                        es = 0
                    else:
                        es += 1
                
                # Also update other metrics for logging (even if not used for early stopping)
                if aupr > best_eval_aupr:
                    best_eval_aupr = aupr
                if recall > best_eval_recall:
                    best_eval_recall = recall
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                if not np.isnan(auroc):
                    if best_eval_auroc is None:
                        best_eval_auroc = auroc
                    elif auroc > best_eval_auroc:
                        best_eval_auroc = auroc
                
                if is_main_process:
                    if ith_epoch == 0 or eval_loss < best_eval_loss:
                        model_path = (
                            config.model_save_path + "affinity" + f"{suffix}.pt"
                        )
                        save_model_state_dict(
                            model, model_path, config, rank, world_size
                        )
                        # Flush logging to ensure save messages appear
                        logging.getLogger().handlers[0].flush() if logging.getLogger().handlers else None
                    elif not (track_ef10 and aupr_policy_frozen):
                        # Determine metric name for logging
                        if config.early_stopping_metric == "aupr":
                            metric_name = "validation AUPR"
                        elif config.early_stopping_metric == "recall":
                            metric_name = "validation recall"
                        elif config.early_stopping_metric == "auroc":
                            metric_name = "validation AUROC"
                        else:  # loss
                            metric_name = "validation loss"
                        logging.info(
                            "Early stopping counter %d of %d (%s did not improve)",
                            es, config.early_stopping_patience, metric_name
                        )
                        model_path = (
                            config.model_save_path
                            + "affinity"
                            + f"{suffix}_es.pt"
                        )
                        save_model_state_dict(
                            model, model_path, config, rank, world_size
                        )
                        aupr_saved_epoch = ith_epoch
                        # Flush logging
                        logging.getLogger().handlers[0].flush() if logging.getLogger().handlers else None
                    if track_ef10:
                        was_ef_frozen = ef_policy.frozen
                        if ef_policy.observe(ith_epoch, val_ef10_hits[-1]):
                            ef_path = (
                                config.model_save_path
                                + "affinity"
                                + f"{suffix}_es_ef10.pt"
                            )
                            save_model_state_dict(
                                model, ef_path, config, rank, world_size
                            )
                            logging.info(
                                "Saved EF@10%% checkpoint at epoch %d: hits=%s",
                                ith_epoch,
                                val_ef10_hits[-1],
                            )
                        elif ef_policy.frozen and not was_ef_frozen:
                            logging.info(
                                "EF@10%% checkpoint policy frozen at epoch %d "
                                "(saved epoch %s, hits=%s)",
                                ith_epoch,
                                ef_policy.saved_epoch,
                                ef_policy.best_hits,
                            )

                if eval_backup_state is not None:
                    model.load_state_dict(eval_backup_state, strict=True)

                if (not aupr_policy_frozen) and es >= config.early_stopping_patience:
                    aupr_policy_frozen = True
                    if is_main_process and track_ef10:
                        logging.info(
                            "AUPR checkpoint policy frozen at epoch %d (saved epoch %s)",
                            ith_epoch,
                            aupr_saved_epoch,
                        )
                if trajectory_should_stop(
                    track_ef10,
                    es,
                    config.early_stopping_patience,
                    aupr_policy_frozen,
                    ef_policy.frozen,
                ):
                    if is_main_process:
                        # Format best metric value for logging
                        if config.early_stopping_metric == "aupr":
                            best_metric_value = best_eval_aupr
                            metric_name = "AUPR"
                        elif config.early_stopping_metric == "recall":
                            best_metric_value = best_eval_recall
                            metric_name = "Recall"
                        elif config.early_stopping_metric == "auroc":
                            best_metric_value = best_eval_auroc
                            metric_name = "AUROC"
                        else:  # loss
                            best_metric_value = best_eval_loss
                            metric_name = "Loss"
                        
                        logging.info(
                            "Early stopping triggered after %d epochs without improvement "
                            "(patience=%d, metric=%s, best %s=%.4f, best AUPR=%.4f, "
                            "best Recall=%.4f, best AUROC=%.4f, best Loss=%.4f)",
                            es,
                            config.early_stopping_patience,
                            config.early_stopping_metric.upper(),
                            metric_name,
                            best_metric_value,
                            best_eval_aupr,
                            best_eval_recall,
                            best_eval_auroc if best_eval_auroc is not None else float("nan"),
                            best_eval_loss,
                        )
                        # Create directory for loss file
                        loss_path = config.loss_save_path + "affinity" + f"{suffix}_val.pt"
                        Path(loss_path).parent.mkdir(parents=True, exist_ok=True)
                        # FIXED: Use len(val_aupr) or len(val_bce_per_epoch) for epochs list instead of train_loss_per_epoch
                        num_epochs = max(len(val_aupr), len(val_bce_per_epoch), len(train_loss_per_epoch))
                        if num_epochs == 0:
                            num_epochs = config.max_epochs  # Fallback to max_epochs if all are empty
                        metrics_payload = {
                                "train_total_loss_per_batch": train_loss,  # Per-batch total losses
                                "train_total_loss_per_epoch": train_loss_per_epoch,  # Per-epoch total losses
                                "train_cls_loss_per_epoch": train_cls_loss_per_epoch,  # Per-epoch classification losses
                                "train_cl_loss_per_epoch": train_cl_loss_per_epoch,    # Per-epoch contrastive losses
                                "val_loss_per_epoch": val_bce_per_epoch,  # Per-epoch validation classification losses
                                "val_aupr": val_aupr,  # Per-epoch validation AUPR
                                "val_auroc": val_auroc,
                                "val_precision": val_precision_per_epoch,
                                "val_recall": val_recall_per_epoch,
                                "val_f1": val_f1_per_epoch,
                                "epochs": list(range(num_epochs)),  # FIXED: Use actual number of epochs from validation data
                            }
                        attach_ef10_curves(
                            metrics_payload,
                            config,
                            val_ef10,
                            val_ef10_hits,
                            ef_policy,
                            aupr_policy_frozen,
                            aupr_saved_epoch,
                        )
                        torch.save(metrics_payload, loss_path)
                        saved_metrics = True
                    break

        # If early stopping did not trigger (i.e., we finished all epochs),
        # ensure that per-epoch metrics are still saved.
        # FIXED: Save even if train_loss_per_epoch is empty (use validation data length)
        if is_main_process and (not saved_metrics):
            loss_path = config.loss_save_path + "affinity" + f"{suffix}_val.pt"
            Path(loss_path).parent.mkdir(parents=True, exist_ok=True)
            # FIXED: Use len(val_aupr) or len(val_bce_per_epoch) for epochs list instead of train_loss_per_epoch
            num_epochs = max(len(val_aupr), len(val_bce_per_epoch), len(train_loss_per_epoch))
            if num_epochs == 0:
                num_epochs = config.max_epochs  # Fallback to max_epochs if all are empty
            metrics_payload = {
                    "train_total_loss_per_batch": train_loss,  # Per-batch total losses
                    "train_total_loss_per_epoch": train_loss_per_epoch,  # Per-epoch total losses
                    "train_cls_loss_per_epoch": train_cls_loss_per_epoch,  # Per-epoch classification losses
                    "train_cl_loss_per_epoch": train_cl_loss_per_epoch,    # Per-epoch contrastive losses
                    "val_loss_per_epoch": val_bce_per_epoch,  # Per-epoch validation classification losses
                    "val_aupr": val_aupr,  # Per-epoch validation AUPR
                    "val_auroc": val_auroc,
                    "val_precision": val_precision_per_epoch,
                    "val_recall": val_recall_per_epoch,
                    "val_f1": val_f1_per_epoch,
                    "epochs": list(range(num_epochs)),  # FIXED: Use actual number of epochs from validation data
                }
            attach_ef10_curves(
                metrics_payload,
                config,
                val_ef10,
                val_ef10_hits,
                ef_policy,
                aupr_policy_frozen,
                aupr_saved_epoch,
            )
            torch.save(metrics_payload, loss_path)


def str2bool(v):
    """Convert a string to a boolean value."""
    if isinstance(v, bool):
        return v
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", type=str, default="", help="suffix for saved models")
    p.add_argument("--output_dir", type=str, default=None, help="Base directory for model/loss/test outputs (overrides config.*_save_path when set)")
    p.add_argument(
        "--device",
        type=str,
        default="1",
        help="CUDA device index",
    )
    p.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated list of GPU IDs to use (e.g., '0,1,2,3'). "
            "Sets CUDA_VISIBLE_DEVICES. If not specified, all available GPUs "
            "are used."
        ),
    )
    p.add_argument(
        "--pooling",
        default="MTP",
        type=str,
        choices=["MTP", "GMP", "GAP"],
        help="Pooling type: MTP, GMP (max), or GAP (mean)",
    )
    p.add_argument(
        "--contrast",
        default=True,
        type=str2bool,
        help="whether to do contrastive learning",
    )
    p.add_argument(
        "--contrast_temp",
        type=float,
        default=0.1,
        help="Temperature for contrastive learning similarity scaling",
    )
    p.add_argument(
        "--contrast_world_size",
        type=int,
        default=1,
        help="Number of distributed workers for contrastive learning",
    )
    p.add_argument(
        "--contrast_weight",
        type=float,
        default=0.05,
        help="Weight of contrastive loss term relative to BCE loss",
    )
    p.add_argument(
        "--contrast_multi_view",
        default=False,
        type=str2bool,
        help="Enable explicit multi-view contrastive learning (edge dropout + feature masking view).",
    )
    p.add_argument(
        "--feature_mask_prob",
        type=float,
        default=0.0,
        help="Feature masking probability for the second contrastive view.",
    )
    p.add_argument(
        "--feature_mask_value",
        type=float,
        default=0.0,
        help="Scalar value used when masking node features for contrastive view-2.",
    )
    p.add_argument(
        "--augment_eps_view2",
        type=float,
        default=None,
        help="Optional edge keep probability for contrastive view-2. Defaults to --augment_eps.",
    )
    p.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Manual override for positive class weight in BCELoss. "
             "If None, automatically calculated from class distribution. "
             "pos_weight = num_negatives / num_positives",
    )
    p.add_argument(
        "--focal_alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for Focal Loss (weighting factor for rare class, default: 1.0)",
    )
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help=(
            "Gamma parameter for Focal Loss (focusing parameter, default: 2.0). "
            "Higher gamma focuses more on hard examples."
        ),
    )
    p.add_argument(
        "--loss_type",
        type=str,
        default="focal",
        choices=["focal", "asymmetric_focal", "focal_tversky", "pairwise_rank_bce"],
        help="Classification loss type.",
    )
    p.add_argument(
        "--loss_use_pos_weight_scale",
        default=True,
        type=str2bool,
        help="Apply pos_weight scaling on top of samplewise losses.",
    )
    p.add_argument("--asym_alpha_pos", type=float, default=1.0, help="Asymmetric focal alpha for positives.")
    p.add_argument("--asym_alpha_neg", type=float, default=1.0, help="Asymmetric focal alpha for negatives.")
    p.add_argument("--asym_gamma_pos", type=float, default=0.0, help="Asymmetric focal gamma for positives.")
    p.add_argument("--asym_gamma_neg", type=float, default=2.0, help="Asymmetric focal gamma for negatives.")
    p.add_argument("--tversky_alpha", type=float, default=0.3, help="Focal-Tversky alpha (FP penalty).")
    p.add_argument("--tversky_beta", type=float, default=0.7, help="Focal-Tversky beta (FN penalty).")
    p.add_argument("--tversky_gamma", type=float, default=1.33, help="Focal-Tversky gamma.")
    p.add_argument(
        "--pairwise_margin",
        type=float,
        default=0.0,
        help="Margin in logit space for pairwise_rank_bce.",
    )
    p.add_argument(
        "--pairwise_lambda_cal",
        type=float,
        default=0.25,
        help="BCE calibration weight for pairwise_rank_bce.",
    )
    p.add_argument(
        "--log_file",
        type=str,
        default="logs/train_affinity.log",
        help="Path to the log file where training logs will be written",
    )
    p.add_argument("--batch_size", type=int, default=32, help="batch size")
    p.add_argument(
        "--k_pos_override",
        type=int,
        default=None,
        help=(
            "Optional override for number of positives per batch (k_pos) when using "
            "stratified sampling with balanced_epoch_sampling=False. "
            "Must satisfy 1 <= k_pos_override <= batch_size-1."
        ),
    )
    p.add_argument(
        "--stratified_without_replacement",
        nargs="?",
        const=True,
        default=True,
        type=lambda x: False if x and x.lower() == "false" else True,
        help=(
            "Sample without replacement in StratifiedPosNegBatchSampler (default: True). "
            "Set to False to use replacement sampling. When True, samples are cycled "
            "through without reuse within an epoch, which may reduce overfitting."
        ),
    )
    p.add_argument(
        "--stop_epoch_when_exhausted",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: False if x and x.lower() == "false" else True,
        help=(
            "When using stratified_without_replacement, stop the epoch once either "
            "the positive or negative pool is exhausted, instead of cycling (default: False)."
        ),
    )
    p.add_argument(
        "--negative_rounds",
        type=int,
        default=1,
        help=(
            "Disjoint rotating negative bags per positive each epoch. Values >1 require "
            "stratified sampling without replacement and stop_epoch_when_exhausted."
        ),
    )
    p.add_argument(
        "--negative_rotation_seed",
        type=int,
        default=20260718,
        help="Seed for deterministic rotating negative bags.",
    )
    p.add_argument(
        "--use_stratified_sampling",
        nargs="?",
        const=True,
        default=True,
        type=lambda x: False if x and x.lower() == "false" else True,
        help=(
            "Use StratifiedPosNegBatchSampler to enforce balanced batches (default: True). "
            "Set to False to use natural class distribution sampling without balance."
        ),
    )
    p.add_argument(
        "--balanced_epoch_sampling",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: False if x and x.lower() == "false" else True,
        help=(
            "When using stratified sampling, ensure each epoch sees equal numbers "
            "of positives and negatives by oversampling the minority class (default: False)."
        ),
    )
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of epochs to wait without improvement before early stopping (default: 5)",
    )
    p.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.01,
        help=(
            "Minimum improvement required to reset patience counter "
            "(default: 0.01 for recall; use ~0.001 for aupr/auroc)."
        ),
    )
    p.add_argument(
        "--early_stopping_metric",
        type=str,
        default="recall",
        choices=["aupr", "recall", "auroc", "loss"],
        help=(
            "Metric to use for early stopping: 'aupr' (PR AUC), "
            "'recall' (binary recall at 0.5), 'auroc' (ROC AUC), "
            "or 'loss' (validation loss). "
            "Default: 'recall' (recommended for datasets with noisy negative labels)."
        ),
    )
    p.add_argument(
        "--track_ef10_checkpoint",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Opt-in validation EF@10% tracking and an independent "
            "*_es_ef10.pt checkpoint. Default false. Requires "
            "early_stopping_metric=aupr so *_es.pt stays the AUPR policy."
        ),
    )
    p.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs (default: 100)",
    )
    p.add_argument(
        "--aptamer_list_file",
        type=str,
        required=True,
        help="Text file listing proteins with known high-affinity aptamers (one ID per line)",
    )
    # Data path arguments (optional, uses defaults if not provided)
    p.add_argument(
        "--train_fasta_file",
        type=str,
        default=None,
        help="Path to training FASTA file (default: ../data/nrPDB-GO_train_sequences.fasta)",
    )
    p.add_argument(
        "--val_fasta_file",
        type=str,
        default=None,
        help=(
            "Path to validation FASTA file "
            "(default: ../data/nrPDB-GO_val_sequences.fasta). "
            "Ignored if --use_kfold is True."
        ),
    )
    p.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for K-fold cross-validation (default: 5)",
    )
    p.add_argument(
        "--use_kfold",
        default=True,
        type=str2bool,
        help=(
            "Whether to use K-fold cross-validation (default: True). "
            "If False, uses traditional train/val split."
        ),
    )
    p.add_argument(
        "--fold_membership_tsv",
        type=str,
        default=None,
        help=(
            "Optional long-format train80 fold membership TSV "
            "(protein_id, fold, split, dataset_index). When set, K-fold "
            "splits are read from this file instead of fragment-level "
            "StratifiedKFold. Use grouped cluster/accession membership "
            "from export_grouped_train_fold_membership.py."
        ),
    )
    p.add_argument(
        "--prottrans_output_path",
        type=str,
        default=None,
        help="Path to ProtTrans embeddings directory (default: ../data/bpDataSet/prottrans/)",
    )
    p.add_argument(
        "--esm2_output_path",
        type=str,
        default=None,
        help="Path to ESM2 embeddings directory (default: ../data/bpDataSet/esm2/)",
    )
    p.add_argument(
        "--pdb_dir",
        type=str,
        default=None,
        help="Path to PDB directory (default: ../data/pdb_dir/)",
    )
    p.add_argument(
        "--proc_pdb_output_path",
        type=str,
        default=None,
        help="Path to processed PDB data directory (default: ../data/bpDataSet/structure_data/)",
    )
    p.add_argument(
        "--dssp_output_path",
        type=str,
        default=None,
        help="Path to DSSP features directory (default: ../data/bpDataSet/dssp/)",
    )
    p.add_argument(
        "--disable_prottrans",
        default=False,
        type=str2bool,
        help="Use zero ProtTrans features instead of loading embeddings (default: False)",
    )
    p.add_argument(
        "--disable_esm2",
        default=False,
        type=str2bool,
        help="Use zero ESM2 features instead of loading embeddings (default: False)",
    )
    p.add_argument(
        "--disable_dssp",
        default=False,
        type=str2bool,
        help="Use zero DSSP features instead of loading DSSP arrays (default: False)",
    )
    p.add_argument(
        "--disable_dihedrals",
        default=False,
        type=str2bool,
        help="Zero backbone dihedral node_s features (6 dims) while keeping tensor shape (default: False)",
    )
    p.add_argument(
        "--disable_orientations",
        default=False,
        type=str2bool,
        help="Zero backbone orientation node_v vectors (2 channels) (default: False)",
    )
    p.add_argument(
        "--disable_sidechains",
        default=False,
        type=str2bool,
        help="Zero sidechain proxy node_v vector (1 channel) (default: False)",
    )
    p.add_argument(
        "--disable_rbf",
        default=False,
        type=str2bool,
        help="Zero RBF distance edge_s features (16 dims) (default: False)",
    )
    p.add_argument(
        "--disable_pos_embeddings",
        default=False,
        type=str2bool,
        help="Zero sinusoidal positional edge_s embeddings (16 dims) (default: False)",
    )
    p.add_argument(
        "--disable_edge_direction",
        default=False,
        type=str2bool,
        help="Zero unit Cα–Cα edge_v direction vectors (default: False)",
    )
    p.add_argument(
        "--plddt_geometry_downweight",
        default=False,
        type=str2bool,
        help=(
            "Downweight DSSP/dihedrals/orientations/sidechains/edges on low-pLDDT "
            "residues. Sequence and LM embeddings are kept. Does not drop nodes."
        ),
    )
    p.add_argument(
        "--plddt_dir",
        type=str,
        default=None,
        help="Directory of per-residue pLDDT TSVs (residue_index, plddt).",
    )
    p.add_argument(
        "--plddt_weight_mode",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="hard: weight=1 if pLDDT>=thresh else 0. soft: linear ramp soft_lo–soft_hi.",
    )
    p.add_argument(
        "--plddt_thresh",
        type=float,
        default=50.0,
        help="Hard-mode pLDDT threshold (default: 50).",
    )
    p.add_argument(
        "--plddt_soft_lo",
        type=float,
        default=40.0,
        help="Soft-mode ramp start (weight 0).",
    )
    p.add_argument(
        "--plddt_soft_hi",
        type=float,
        default=60.0,
        help="Soft-mode ramp end (weight 1).",
    )
    p.add_argument(
        "--plddt_edge_mode",
        type=str,
        default="scale",
        choices=["scale", "knn_highconf", "none"],
        help=(
            "scale: multiply edge features by min(w_i,w_j). "
            "knn_highconf: rebuild k-NN on high-pLDDT CAs then scale. "
            "none: leave edges unchanged."
        ),
    )
    p.add_argument(
        "--plddt_pool_weight",
        default=False,
        type=str2bool,
        help="Use pLDDT weights for GAP pooling. Off by default.",
    )
    p.add_argument(
        "--plddt_knn_keep_min",
        type=float,
        default=0.5,
        help="Minimum residue weight to keep a CA in knn_highconf (default: 0.5).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of data loader workers (default: 8)",
    )
    p.add_argument(
        "--use_mixed_precision",
        default=False,
        type=str2bool,
        help=(
            "Whether to use mixed precision training (default: False). "
            "Uses torch.cuda.amp to reduce memory usage by ~50%. "
            "Recommended when experiencing OOM errors."
        ),
    )
    p.add_argument(
        "--seed_base",
        type=int,
        default=1234,
        help="Base seed for reproducibility (default: 1234). Fold/epoch seeds derive from this.",
    )
    p.add_argument(
        "--deterministic",
        default=True,
        type=str2bool,
        help="Enable deterministic CUDA/cuDNN/PyTorch behavior (default: True).",
    )
    p.add_argument(
        "--deterministic_warn_only",
        default=True,
        type=str2bool,
        help=(
            "When deterministic mode is enabled, warn instead of raising for nondeterministic ops "
            "(default: True). Set False to fail fast."
        ),
    )

    p.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer (default: 5e-5). Reduced from 1e-4 for more stable learning.",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization) for optimizer (default: 1e-5).",
    )
    p.add_argument(
        "--scheduler_factor",
        type=float,
        default=None,
        help=(
            "Factor for ReduceLROnPlateau (e.g., 0.2, 0.5, 0.8). Use 1.0 for no LR decay "
            "(constant LR). If None, no scheduler is used. Values > 1 are not supported."
        ),
    )
    p.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        help="Patience (in epochs) for ReduceLROnPlateau scheduler (default: 5).",
    )
    p.add_argument(
        "--lr_schedule_type",
        type=str,
        default="plateau",
        choices=["plateau", "exponential", "cosine", "onecycle", "constant"],
        help=(
            "Learning-rate schedule type. "
            "'plateau' uses --scheduler_factor/--scheduler_patience, "
            "'exponential' uses --exp_gamma, "
            "'cosine' uses --cosine_min_lr, "
            "'onecycle' uses OneCycleLR, "
            "'constant' disables scheduling."
        ),
    )
    p.add_argument(
        "--exp_gamma",
        type=float,
        default=0.95,
        help="Gamma for ExponentialLR when --lr_schedule_type=exponential (default: 0.95).",
    )
    p.add_argument(
        "--cosine_min_lr",
        type=float,
        default=1e-6,
        help="eta_min for CosineAnnealingLR when --lr_schedule_type=cosine (default: 1e-6).",
    )
    p.add_argument(
        "--onecycle_max_lr",
        type=float,
        default=None,
        help=(
            "Max LR for OneCycleLR when --lr_schedule_type=onecycle. "
            "If not set, --learning_rate is used as max_lr."
        ),
    )
    p.add_argument(
        "--onecycle_pct_start",
        type=float,
        default=0.1,
        help="OneCycleLR pct_start (default: 0.1).",
    )
    p.add_argument(
        "--onecycle_div_factor",
        type=float,
        default=25.0,
        help="OneCycleLR div_factor (default: 25.0).",
    )
    p.add_argument(
        "--onecycle_final_div_factor",
        type=float,
        default=1e4,
        help="OneCycleLR final_div_factor (default: 1e4).",
    )
    p.add_argument(
        "--gradient_clip_max_norm",
        type=float,
        default=1.0,
        help="Maximum norm for gradient clipping (default: 1.0). Set to 0 to disable gradient clipping.",
    )

    p.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden embedding dimension for GNN layers (default: 256)",
    )
    p.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of message-passing layers in GNN (default: 3)",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate in GNN layers (default: 0.1)",
    )
    p.add_argument(
        "--dropout_schedule",
        type=str,
        default="none",
        choices=["none", "linear"],
        help="Dropout schedule type across epochs.",
    )
    p.add_argument(
        "--dropout_final",
        type=float,
        default=0.1,
        help="Final dropout for linear dropout schedule.",
    )
    p.add_argument(
        "--dropout_schedule_start_epoch",
        type=int,
        default=0,
        help="Epoch index to start applying dropout schedule.",
    )
    p.add_argument(
        "--use_residual",
        default=False,
        type=str2bool,
        help="Enable residual connections across message-passing layers.",
    )
    p.add_argument(
        "--residual_alpha",
        type=float,
        default=1.0,
        help="Residual branch scaling factor.",
    )
    p.add_argument(
        "--augment_eps",
        type=float,
        default=0.95,
        help="Edge dropout / keep probability for augmentation (default: 0.95)",
    )
    p.add_argument(
        "--ema_enabled",
        default=False,
        type=str2bool,
        help="Enable exponential moving average of model weights.",
    )
    p.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay factor.",
    )
    p.add_argument(
        "--swa_enabled",
        default=False,
        type=str2bool,
        help="Enable stochastic weight averaging (running average of weights).",
    )
    p.add_argument(
        "--swa_start_epoch",
        type=int,
        default=70,
        help="Epoch index to start SWA updates.",
    )
    p.add_argument(
        "--eval_use_ema",
        default=True,
        type=str2bool,
        help="Use EMA weights for validation/checkpointing when available.",
    )
    p.add_argument(
        "--eval_use_swa",
        default=False,
        type=str2bool,
        help="Use SWA weights for validation/checkpointing when available.",
    )

    args = p.parse_args()

    # ------------------------------------------------------------------
    # Logging configuration
    # ------------------------------------------------------------------
    # Use SLURM job ID in log file name if available
    if args.log_file is None:
        # Default log file name
        log_file_base = "logs/train_affinity.log"
    else:
        log_file_base = args.log_file
    
    # Replace %j placeholder with job ID if present (SLURM-style placeholder)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        # Replace %j with actual job ID if present in the log file name
        if "%j" in log_file_base:
            log_file_base = log_file_base.replace("%j", slurm_job_id)
            log_file = Path(log_file_base)
        else:
            # If no %j placeholder, insert job ID before file extension
            log_file_path = Path(log_file_base)
            log_file = (
                log_file_path.parent
                / f"{log_file_path.stem}_{slurm_job_id}{log_file_path.suffix}"
            )
    else:
        # If no job ID and %j is present, remove it (fallback)
        if "%j" in log_file_base:
            log_file_base = log_file_base.replace("%j", "")
        log_file = Path(log_file_base)
    
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,  # Override any existing configuration
    )
    # Ensure unbuffered logging for real-time updates
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.INFO)

    logging.info("Parsed arguments: %s", args)

    config = get_config()
    # Override default output directories if requested
    if args.output_dir is not None:
        base_out = Path(args.output_dir)
        config.model_save_path = str(base_out / "model_")
        config.loss_save_path = str(base_out / "loss_")
        config.test_result_path = str(base_out / "test_")
    config.batch_size = args.batch_size
    config.k_pos_override = args.k_pos_override
    if config.k_pos_override is not None:
        if config.k_pos_override < 1 or config.k_pos_override >= config.batch_size:
            raise ValueError(
                f"Invalid --k_pos_override={config.k_pos_override}. "
                f"Must satisfy 1 <= k_pos_override <= batch_size-1 (batch_size={config.batch_size})."
            )
    config.use_stratified_sampling = args.use_stratified_sampling
    config.balanced_epoch_sampling = args.balanced_epoch_sampling
    config.stratified_without_replacement = args.stratified_without_replacement
    config.stop_epoch_when_exhausted = args.stop_epoch_when_exhausted
    config.negative_rounds = args.negative_rounds
    config.negative_rotation_seed = args.negative_rotation_seed
    if config.negative_rounds < 1:
        raise ValueError("--negative_rounds must be >= 1")
    if config.negative_rounds > 1 and (
        not config.use_stratified_sampling
        or not config.stratified_without_replacement
        or not config.stop_epoch_when_exhausted
        or config.balanced_epoch_sampling
    ):
        raise ValueError(
            "--negative_rounds > 1 requires stratified sampling, sampling without "
            "replacement, stop_epoch_when_exhausted=true, and balanced_epoch_sampling=false."
        )
    config.max_epochs = args.max_epochs
    config.early_stopping_patience = args.early_stopping_patience
    config.early_stopping_min_delta = args.early_stopping_min_delta
    config.early_stopping_metric = args.early_stopping_metric.lower()  # Normalize to lowercase
    config.track_ef10_checkpoint = bool(args.track_ef10_checkpoint)
    if config.track_ef10_checkpoint and config.early_stopping_metric != "aupr":
        raise ValueError(
            "--track_ef10_checkpoint requires --early_stopping_metric aupr "
            "so *_es.pt remains the AUPR checkpoint."
        )

    
    config.optimizer["lr"] = args.learning_rate
    config.optimizer["weight_decay"] = args.weight_decay
    config.scheduler_factor = args.scheduler_factor
    config.scheduler_patience = args.scheduler_patience
    config.lr_schedule_type = args.lr_schedule_type
    config.exp_gamma = args.exp_gamma
    config.cosine_min_lr = args.cosine_min_lr
    config.onecycle_max_lr = args.onecycle_max_lr
    config.onecycle_pct_start = args.onecycle_pct_start
    config.onecycle_div_factor = args.onecycle_div_factor
    config.onecycle_final_div_factor = args.onecycle_final_div_factor
    config.gradient_clip_max_norm = args.gradient_clip_max_norm
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.num_layers
    config.dropout = args.dropout
    config.dropout_schedule = args.dropout_schedule
    config.dropout_final = args.dropout_final
    config.dropout_schedule_start_epoch = args.dropout_schedule_start_epoch
    config.use_residual = args.use_residual
    config.residual_alpha = args.residual_alpha
    config.augment_eps = args.augment_eps
    config.augment_eps_view2 = args.augment_eps_view2
    if torch.cuda.is_available() and args.device != "":
        config.device = "cuda:" + args.device
        logging.info("Using device: %s", config.device)
    else:
        config.device = "cpu"
        logging.warning("CUDA unavailable or no device specified, using CPU")
    config.pooling = args.pooling
    config.contrast = args.contrast
    config.contrast_temp = args.contrast_temp
    config.contrast_world_size = args.contrast_world_size
    config.contrast_weight = args.contrast_weight
    config.contrast_multi_view = args.contrast_multi_view
    config.feature_mask_prob = args.feature_mask_prob
    config.feature_mask_value = args.feature_mask_value
    config.pos_weight = args.pos_weight
    config.focal_alpha = args.focal_alpha
    config.focal_gamma = args.focal_gamma
    config.loss_type = args.loss_type
    config.loss_use_pos_weight_scale = args.loss_use_pos_weight_scale
    config.asym_alpha_pos = args.asym_alpha_pos
    config.asym_alpha_neg = args.asym_alpha_neg
    config.asym_gamma_pos = args.asym_gamma_pos
    config.asym_gamma_neg = args.asym_gamma_neg
    config.tversky_alpha = args.tversky_alpha
    config.tversky_beta = args.tversky_beta
    config.tversky_gamma = args.tversky_gamma
    config.pairwise_margin = args.pairwise_margin
    config.pairwise_lambda_cal = args.pairwise_lambda_cal
    config.ema_enabled = args.ema_enabled
    config.ema_decay = args.ema_decay
    config.swa_enabled = args.swa_enabled
    config.swa_start_epoch = args.swa_start_epoch
    config.eval_use_ema = args.eval_use_ema
    config.eval_use_swa = args.eval_use_swa
    config.use_mixed_precision = args.use_mixed_precision
    config.seed_base = args.seed_base
    config.deterministic = args.deterministic
    config.deterministic_warn_only = args.deterministic_warn_only
    config.plddt_pool_weight = args.plddt_pool_weight
    if args.plddt_geometry_downweight or args.plddt_pool_weight:
        if not args.plddt_dir:
            raise ValueError(
                "--plddt_dir is required when --plddt_geometry_downweight or "
                "--plddt_pool_weight is true"
            )
        logging.info(
            "pLDDT weighting: geometry=%s pool=%s mode=%s thresh=%s "
            "soft_lo=%s soft_hi=%s edge=%s knn_keep=%s dir=%s",
            args.plddt_geometry_downweight,
            args.plddt_pool_weight,
            args.plddt_weight_mode,
            args.plddt_thresh,
            args.plddt_soft_lo,
            args.plddt_soft_hi,
            args.plddt_edge_mode,
            args.plddt_knn_keep_min,
            args.plddt_dir,
        )

    train(
        config,
        args.suffix,
        args.aptamer_list_file,
        train_fasta_file=args.train_fasta_file,
        val_fasta_file=args.val_fasta_file,
        prottrans_output_path=args.prottrans_output_path,
        esm2_output_path=args.esm2_output_path,
        pdb_dir=args.pdb_dir,
        proc_pdb_output_path=args.proc_pdb_output_path,
        dssp_output_path=args.dssp_output_path,
        num_workers=args.num_workers,
        disable_prottrans=args.disable_prottrans,
        disable_esm2=args.disable_esm2,
        disable_dssp=args.disable_dssp,
        disable_dihedrals=args.disable_dihedrals,
        disable_orientations=args.disable_orientations,
        disable_sidechains=args.disable_sidechains,
        disable_rbf=args.disable_rbf,
        disable_pos_embeddings=args.disable_pos_embeddings,
        disable_edge_direction=args.disable_edge_direction,
        plddt_geometry_downweight=args.plddt_geometry_downweight,
        plddt_dir=args.plddt_dir,
        plddt_weight_mode=args.plddt_weight_mode,
        plddt_thresh=args.plddt_thresh,
        plddt_soft_lo=args.plddt_soft_lo,
        plddt_soft_hi=args.plddt_soft_hi,
        plddt_edge_mode=args.plddt_edge_mode,
        plddt_pool_weight=args.plddt_pool_weight,
        plddt_knn_keep_min=args.plddt_knn_keep_min,
        k_folds=args.k_folds,
        use_kfold=args.use_kfold,
        gpu_ids=args.gpu_ids,
        fold_membership_tsv=args.fold_membership_tsv,
    )
