#!/usr/bin/env python3
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
from pathlib import Path
import logging
import warnings
import ml_collections
import copy
from Bio import SeqIO
import numpy as np
import torch
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
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold

# Note: The following classes were previously imported from model.py and nt_xent.py,
# but have been incorporated directly into this script.
# Dependencies: These classes require the 'gvp' and 'pool' modules to be available.
# If those modules are not available, you may need to keep the imports or install
# the required dependencies.

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
# Try to import new state dict API - can pass None for optimizers if not needed
try:
    from torch.distributed.checkpoint.state_dict import get_state_dict
    NEW_STATE_DICT_API_AVAILABLE = True
except ImportError:
    NEW_STATE_DICT_API_AVAILABLE = False
try:
    from torch.distributed.fsdp import apply_activation_checkpointing
    ACTIVATION_CHECKPOINTING_AVAILABLE = True
except ImportError:
    # apply_activation_checkpointing not available in PyTorch 2.9.1+
    # It was removed/deprecated in newer PyTorch versions
    ACTIVATION_CHECKPOINTING_AVAILABLE = False

try:
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    SIZE_BASED_WRAP_AVAILABLE = True
except ImportError:
    SIZE_BASED_WRAP_AVAILABLE = False


# -----------------------------------------------------------------------------
# NT-Xent Contrastive Loss (previously from nt_xent.py)
# -----------------------------------------------------------------------------

class GatherLayer(torch.autograd.Function):
    """All-gather autograd layer for distributed training."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


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
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)
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
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
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
        
        # Compute BCE loss (both probs and targets are now float32)
        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Compute p_t: probability of true class
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting for positive class
        alpha_t = torch.where(targets == 1, self.alpha, 1.0)
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()


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
    ):
        super(GVP_encoder, self).__init__()
        activations = (F.relu, None)
        self.seq_in = seq_in
        
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
            if pertubed:
                h_V = layer(h_V, edge_index, h_E, pertubed=True)
            else:
                h_V = layer(h_V, edge_index, h_E)

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

    def forward(self, h_V, edge_index, h_E, seq, batch=None, node_weight=None):
        """Forward pass through the complete network."""
        h_V_ = self.GVP_encoder(h_V, edge_index, h_E, seq)

        if self.pooling == "MTP":
            x = self.pool(h_V_, batch, edge_index.long())
        elif (
            self.pooling == "GAP"
            and getattr(self, "pool_use_plddt_weight", False)
            and node_weight is not None
        ):
            x = plddt_geo.weighted_global_mean_pool(h_V_, batch, node_weight)
        else:
            x = self.pool(h_V_, batch)

        if self.pertub:
            h_V_pertub = self.GVP_encoder(h_V, edge_index, h_E, seq, pertubed=True)
            if self.pooling == "MTP":
                x_pertub = self.pool(h_V_pertub, batch, edge_index.long())
            elif (
                self.pooling == "GAP"
                and getattr(self, "pool_use_plddt_weight", False)
                and node_weight is not None
            ):
                x_pertub = plddt_geo.weighted_global_mean_pool(
                    h_V_pertub, batch, node_weight
                )
            else:
                x_pertub = self.pool(h_V_pertub, batch)
            y_pred = self.readout(x)
            return y_pred, x, x_pertub
        else:
            y_pred = self.readout(x)
            return y_pred


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
        # Note: "device" is ignored when using FSDP with multiple GPUs.
        # In FSDP mode, each rank uses its own GPU based on local_rank.
        # This value is only used when FSDP is disabled.
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
        "augment_eps": 0.95,  # Will be updated from args.augment_eps
        "scheduler": {
            "step_size": 30,
            "gamma": 0.75,
        },
        "model_save_path": "Model/model_",
        "loss_save_path": "Model/loss_",
        "test_result_path": "Model/test_",
        "use_fsdp": False,
        # Options: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_mixed_precision": False,
        # Set to False to avoid gathering all parameters on rank 0 during init
        # (which causes OOM). Model is initialized identically on all ranks.
        "fsdp_sync_module_states": False,
        # Use DDP instead of FSDP for simpler 2-GPU training
        # DDP is recommended for 2 GPUs unless model is too large for single GPU
        "use_ddp": False,
        # Mixed precision for DDP (uses torch.cuda.amp - can reduce memory usage by ~50%)
        "use_mixed_precision": False,
    }
)


def get_config():
    """Return a deep copy of the default configuration."""
    config = copy.deepcopy(CONFIG)
    return config


# -----------------------------------------------------------------------------
# Distributed Training and FSDP2 Setup
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


def init_distributed():
    """Initialize distributed training environment."""
    # Migrate deprecated NCCL environment variables to new names
    # This fixes the deprecation warnings by using the new variable names
    if "NCCL_BLOCKING_WAIT" in os.environ:
        if "TORCH_NCCL_BLOCKING_WAIT" not in os.environ:
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = os.environ["NCCL_BLOCKING_WAIT"]
        # Optionally remove old variable to avoid confusion
        # del os.environ["NCCL_BLOCKING_WAIT"]
    
    if "NCCL_ASYNC_ERROR_HANDLING" in os.environ:
        if "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = os.environ["NCCL_ASYNC_ERROR_HANDLING"]
        # Optionally remove old variable to avoid confusion
        # del os.environ["NCCL_ASYNC_ERROR_HANDLING"]
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        # Set up PyTorch distributed environment from SLURM variables
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        # SLURM_LOCALID might not be set by srun on single node
        # If not set, calculate from SLURM_PROCID (assuming single node)
        if "SLURM_LOCALID" in os.environ:
            local_rank = int(os.environ["SLURM_LOCALID"])
        else:
            # On a single node, local_rank equals rank
            # This ensures each rank uses a different GPU
            local_rank = rank
        
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
        # Set master address and port for distributed initialization
        if "MASTER_ADDR" not in os.environ:
            # Use the first node in the SLURM job
            node_list = os.environ.get("SLURM_NODELIST", "")
            if node_list:
                # Extract first node name (simple approach)
                # For more complex cases, you might need to parse SLURM_NODELIST
                first_node = node_list.split(",")[0].split("[")[0]
                os.environ["MASTER_ADDR"] = first_node
            else:
                # Fallback to localhost for single-node setups
                os.environ["MASTER_ADDR"] = "127.0.0.1"
        
        if "MASTER_PORT" not in os.environ:
            # Use a default port or get from SLURM if available
            # Try to get a unique port based on job ID to avoid conflicts
            job_id = os.environ.get("SLURM_JOB_ID", "")
            if job_id:
                # Use job ID to generate a unique port (base 29500 + job_id % 1000)
                port = 29500 + (int(job_id) % 1000)
                os.environ["MASTER_PORT"] = str(port)
            else:
                os.environ["MASTER_PORT"] = "29500"
    else:
        # Single GPU or CPU mode
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        # Initialize process group
        if not dist.is_initialized():
            # The socket connection warning may occur if:
            # 1. MASTER_ADDR/MASTER_PORT are incorrect
            # 2. Network connectivity issues between nodes
            # 3. Firewall blocking the port
            # Check MASTER_ADDR and MASTER_PORT are set correctly above
            
            # Set device_id to avoid barrier() warnings
            # This specifies which device to use for distributed operations
            device_id = None
            if torch.cuda.is_available():
                # Set the current CUDA device to local_rank before init
                torch.cuda.set_device(local_rank)
                # device_id should be an integer (device index), not a torch.device object
                device_id = local_rank
            
            try:
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method="env://",
                    device_id=device_id,  # Specify device index to avoid barrier() warnings
                )
            except RuntimeError as e:
                if "socket" in str(e).lower() or "connect" in str(e).lower():
                    logging.error(
                        f"Failed to initialize distributed training: {e}\n"
                        f"Check that MASTER_ADDR={os.environ.get('MASTER_ADDR', 'NOT SET')} "
                        f"and MASTER_PORT={os.environ.get('MASTER_PORT', 'NOT SET')} are correct, "
                        f"and that all nodes can reach each other on the network."
                    )
                    raise
                else:
                    raise
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def get_fsdp_sharding_strategy(strategy_str):
    """Convert string to FSDP sharding strategy."""
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    return strategy_map.get(strategy_str.upper(), ShardingStrategy.FULL_SHARD)


def wrap_model_with_ddp(model, device, rank, world_size):
    """Wrap model with DDP for distributed training.
    
    DDP is simpler than FSDP and recommended for 2 GPUs when model fits on each GPU.
    DDP replicates the model on each GPU and synchronizes gradients.
    """
    if world_size == 1:
        return model
    
    # Ensure process group is healthy before wrapping
    if not dist.is_initialized():
        raise RuntimeError("Distributed training not initialized")
    
    # Synchronize all ranks before wrapping to ensure they're ready
    # Multiple synchronization points to ensure all ranks are truly ready
    for sync_round in range(2):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()
        if sync_round == 0:
            time.sleep(0.5)  # Small delay between sync rounds
    
    # Move model to device before wrapping with DDP
    model = model.to(device)
    
    # Ensure model is on the correct device and synchronized
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Wrap with DDP
    # Set find_unused_parameters=True because the model may have unused parameters
    # (e.g., from contrastive learning branch when contrast=True)
    try:
        model = DDP(
            model,
            device_ids=[rank] if torch.cuda.is_available() else None,
            output_device=device if torch.cuda.is_available() else None,
            find_unused_parameters=True,  # Required when model has unused parameters
        )
    except Exception as e:
        if rank == 0:
            logging.error(f"[rank0] Failed to wrap model with DDP: {e}")
        raise
    
    # Synchronize after wrapping to ensure all ranks finished initialization
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()
    
    if rank == 0:
        logging.info(
            f"[rank0] Wrapped model with DDP: rank={rank}, "
            f"device={device}, world_size={world_size}"
        )
    
    return model


def wrap_model_with_fsdp(model, config, rank, world_size, local_rank=None):
    """Wrap model with FSDP for distributed training.
    
    FSDP shards model parameters across GPUs, useful for very large models.
    For 2 GPUs, DDP is usually simpler and sufficient.
    """
    if not config.use_fsdp or world_size == 1:
        return model
    
    # Get local_rank if not provided
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    
    # Define auto wrap policy (wrap transformer-like modules)
    # For GVP models, we can wrap at the GVPConvLayer level
    def auto_wrap_policy_fn(
        module,
        recurse=True,
        unwrapped_params=0,
        nonwrapped_numel=None,
        **kwargs
    ):
        """Auto wrap policy for FSDP.
        
        Args:
            module: The module to potentially wrap
            recurse: Whether to recurse into child modules (default: True)
            unwrapped_params: Number of unwrapped parameters (default: 0)
            nonwrapped_numel: Alternative parameter name used by FSDP2
                (default: None)
            **kwargs: Additional arguments for FSDP2 compatibility
        """
        # Wrap key modules for FSDP sharding
        # Only wrap leaf modules with parameters, not container modules
        # This avoids wrapping empty containers or modules with no parameters
        
        # First check: module must have parameters (nonwrapped_numel > 0)
        # nonwrapped_numel is provided by FSDP and represents the number of
        # elements in unwrapped parameters
        if nonwrapped_numel is not None and nonwrapped_numel == 0:
            return False
        
        # Second check: verify module actually has parameters
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        if param_count == 0:
            return False
        
        # Third check: only wrap specific module types
        if isinstance(module, (
            GVPConvLayer,  # Core GVP convolution layers
            GraphMultisetTransformer,  # Pooling module
            nn.Linear,  # Linear layers
            GVP,  # GVP transformation modules
        )):
            return True
        
        return False
    
    # Set up mixed precision if enabled
    # Note: For FSDP2, we use reduce_dtype for gradient communication
    # while keeping parameters in float32 to avoid dtype mismatches
    # with input data. For true mixed precision, use torch.autocast
    # in the forward pass.
    mixed_precision_policy = None
    if config.fsdp_mixed_precision:
        # Use float16 only for gradient reduction, keep params in float32
        # This avoids dtype mismatch errors while still saving communication
        mixed_precision_policy = MixedPrecision(
            param_dtype=None,  # Keep parameters in float32
            reduce_dtype=torch.float16,  # Use float16 for gradient reduction
            buffer_dtype=torch.float32,  # Keep buffers in float32
        )
    
    # Wrap model with FSDP
    sharding_strategy = get_fsdp_sharding_strategy(config.fsdp_sharding_strategy)
    
    # Explicitly ensure sync_module_states is False (as boolean, not string)
    # This must be done before any logging that uses it
    sync_module_states = bool(getattr(config, 'fsdp_sync_module_states', False))
    # Force to False to avoid OOM - model is on CPU and sync requires GPU
    if sync_module_states:
        if rank == 0:
            logging.warning(
                "[rank0] sync_module_states was True, forcing to False "
                "to avoid OOM with CPU model"
            )
        sync_module_states = False
    
    # Set device before wrapping with FSDP
    # Each rank must use its own GPU (local_rank)
    if torch.cuda.is_available():
        # Ensure this rank uses the correct GPU as the default CUDA device
        torch.cuda.set_device(local_rank)
        # Set device_id to local_rank so FSDP knows which GPU to use
        # The model is on CPU, so FSDP will shard first, then move shards
        device_id = local_rank
        if rank == 0:
            logging.info(
                f"[rank0] Wrapping model with FSDP: rank={rank}, "
                f"local_rank={local_rank}, device_id={device_id}, "
                f"sync_module_states={sync_module_states}"
            )
    else:
        device_id = None
    
    
    # Ensure model is on CPU before wrapping (FSDP will handle device placement)
    # This prevents loading the entire model on one GPU before sharding
    if torch.cuda.is_available():
        # Move model to CPU explicitly to ensure it's not on GPU
        model = model.cpu()
        if rank == 0:
            logging.info(
                "[rank0] Model moved to CPU before FSDP wrapping "
                "(to prevent OOM during initialization)"
            )
    
    if rank == 0:
        logging.info(
            f"[rank0] FSDP config: sync_module_states={sync_module_states}, "
            f"device_id={device_id}, sharding_strategy={sharding_strategy}"
        )
    
    # Wrap model with FSDP
    # With sync_module_states=False and model on CPU, FSDP should:
    # 1. Shard parameters on CPU first
    # 2. Then move each shard to the specified device_id
    # The model is on CPU, so FSDP will shard first, avoiding OOM
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=auto_wrap_policy_fn,
        sync_module_states=sync_module_states,  # Explicitly False to avoid OOM
        device_id=device_id,  # Set to local_rank so FSDP knows which GPU to use
        limit_all_gathers=True,  # Limit all-gather operations to reduce memory
        use_orig_params=True,  # Use original parameters for better memory efficiency
    )
    
    # Log memory usage after FSDP wrapping to verify sharding
    if rank == 0 and torch.cuda.is_available():
        for i in range(world_size):
            if i < torch.cuda.device_count():
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logging.info(
                    f"[rank0] After FSDP wrapping - GPU {i}: "
                    f"{mem_allocated:.2f} GB allocated, "
                    f"{mem_reserved:.2f} GB reserved"
                )
    
    # Log device assignment after FSDP wrapping
    if rank == 0:
        if torch.cuda.is_available():
            for i in range(world_size):
                if i < torch.cuda.device_count():
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    logging.info(
                        f"[rank0] GPU {i}: {mem_allocated:.2f} GB allocated, "
                        f"{mem_reserved:.2f} GB reserved"
                    )
    
    # Apply activation checkpointing to FSDP-wrapped model
    # This trades compute for memory by recomputing activations during backward
    # Note: apply_activation_checkpointing was removed in PyTorch 2.1+ and is not
    # available in PyTorch 2.9.1. The warning below is expected.
    if ACTIVATION_CHECKPOINTING_AVAILABLE:
        try:
            check_fn = lambda submodule: isinstance(submodule, GVPConvLayer)
            apply_activation_checkpointing(model, check_fn=check_fn)
            if rank == 0:
                logging.info(
                    "[rank0] Applied activation checkpointing to GVPConvLayer "
                    "modules using FSDP API"
                )
        except Exception as e:
            if rank == 0:
                logging.warning(
                    f"[rank0] Failed to apply activation checkpointing: {e}"
                )
    else:
        # In PyTorch 2.9.1+, apply_activation_checkpointing is not available
        # This is expected and training will work, just with higher memory usage
        # Only log once to avoid spam (this function may be called multiple times for k-folds)
        if rank == 0 and not hasattr(wrap_model_with_fsdp, '_checkpoint_warning_logged'):
            logging.warning(
                "[rank0] Activation checkpointing not available in PyTorch 2.9.1+. "
                "apply_activation_checkpointing was removed in PyTorch 2.1+. "
                "Memory usage may be high. This is expected and training will continue."
            )
            wrap_model_with_fsdp._checkpoint_warning_logged = True
    
    return model


def save_model_state_dict(model, save_path, config, rank, world_size):
    """Save model state dict with DDP/FSDP support."""
    # Create directory if it doesn't exist
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle DDP-wrapped models
    if isinstance(model, DDP):
        if rank == 0:
            logging.info(f"[rank0] Saving DDP model to {save_path}...")
            # DDP wraps the model, so we need to access .module
            torch.save(model.module.state_dict(), save_path)
            logging.info(f"[rank0] Model saved successfully to {save_path}")
        # Synchronize all ranks after save
        if world_size > 1:
            dist.barrier()
        return
    
    # Handle FSDP-wrapped models
    if config.use_fsdp and world_size > 1 and isinstance(model, FSDP):
        # Use rank-0-only saving as primary method (more reliable on network filesystems)
        # This gathers the full model state to rank 0 and saves as a single file
        # More reliable than sharded checkpoints which can hang on network mounts
        if rank == 0:
            logging.info(f"[rank0] Saving model to {save_path}...")
            logging.info("[rank0] Using rank-0-only method with CPU offloading (gathering full state dict)")
            for handler in logging.getLogger().handlers:
                handler.flush()
        
        # Ensure all ranks are synchronized before starting save
        if world_size > 1:
            dist.barrier()
        
        # Primary method: Save on rank 0 only with CPU offloading
        try:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    # Only rank 0 will have the full state dict, others get None
                    state_dict = model.state_dict()
            
            # Only rank 0 saves the model
            if rank == 0:
                if state_dict is None:
                    raise RuntimeError("Rank 0 did not receive state dict - this should not happen")
                torch.save(state_dict, save_path)
                logging.info(f"[rank0] Model saved successfully to {save_path}")
                logging.info(f"[rank0] Model size: {sum(p.numel() for p in state_dict.values())} parameters")
                for handler in logging.getLogger().handlers:
                    handler.flush()
            
            # Synchronize after save to ensure all ranks complete
            if world_size > 1:
                dist.barrier()
                
        except Exception as e:
            if rank == 0:
                logging.error(f"[rank0] Model save failed: {e}")
                logging.error(f"[rank0] Error type: {type(e).__name__}")
                import traceback
                logging.error(f"[rank0] Traceback:\n{traceback.format_exc()}")
                for handler in logging.getLogger().handlers:
                    handler.flush()
            # Re-raise to let caller handle the error
            raise
    else:
        # Regular saving for non-FSDP models
        if rank == 0:
            logging.info(f"[rank0] Saving model to {save_path}...")
            for handler in logging.getLogger().handlers:
                handler.flush()
        torch.save(model.state_dict(), save_path)
        if rank == 0:
            logging.info(f"[rank0] Model saved successfully to {save_path}")
            for handler in logging.getLogger().handlers:
                handler.flush()


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

    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
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

    def __len__(self):
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


# -----------------------------------------------------------------------------
# Utilities for loading binary labels from an aptamer list
# -----------------------------------------------------------------------------
# The training script expects a **plain text file** that lists one UniProt
# accession per line, e.g.:
#     P12345
#     Q8ABC0
#     ...
#
# In this project, such a list can be generated from the AptaDB table
# `Tables/AptaDB/aptamer_target_proteins_human_2024-08-01.tsv` by taking the
# first column (TargetID), which is a UniProt ID. For example, from the
# project root:
#
#     cat Tables/AptaDB/aptamer_target_proteins_human_2024-08-01.tsv | cut -f1 > aptamer_uniprot_ids.txt
#
# You can then pass this file to the script via:
#     --aptamer_list_file aptamer_uniprot_ids.txt

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
    test_fasta_file=None,
    prottrans_output_path=None,
    esm2_output_path=None,
    pdb_dir=None,
    proc_pdb_output_path=None,
    dssp_output_path=None,
    num_workers=None,
    k_folds=5,
    use_kfold=True,
    gpu_ids=None,
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
        test_fasta_file: Path to test FASTA file (optional, for final
            evaluation after K-fold CV)
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
    """
    # Set GPU visibility before initializing distributed training
    # This must be done before PyTorch initializes CUDA
    if gpu_ids is not None:
        set_gpu_ids(gpu_ids)
    
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()
    is_main_process = rank == 0
    
    # Log rank and local_rank information for debugging
    logging.info(
        "[Rank %d] Initialized distributed training: world_size=%d, local_rank=%d",
        rank, world_size, local_rank
    )
    
    # Log device assignment for each rank
    if torch.cuda.is_available():
        assigned_device = f"cuda:{local_rank}"
        logging.info(
            "[Rank %d] Will use device: %s (local_rank=%d, "
            "CUDA_VISIBLE_DEVICES=%s)",
            rank, assigned_device, local_rank,
            os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        )
    
    # Log GPU selection after initialization
    if gpu_ids is not None and is_main_process:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        logging.info(
            "Restricted GPUs to: %s (CUDA_VISIBLE_DEVICES=%s)",
            gpu_ids, cuda_visible
        )
    
    # Set device based on local rank for distributed training
    if (config.use_fsdp or config.use_ddp) and world_size > 1:
        if torch.cuda.is_available():
            # When using FSDP or DDP with multiple GPUs, use local_rank
            # CUDA_VISIBLE_DEVICES has already been set,
            # so local_rank maps to visible GPUs
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    else:
        # When not using FSDP2, use the device from config
        # If GPU IDs were specified, the device index should be
        # relative to visible GPUs
        if torch.cuda.is_available() and "cuda" in config.device:
            # Extract device index from config.device (e.g., "cuda:1" -> 1)
            device_idx = (
                int(config.device.split(":")[-1])
                if ":" in config.device else 0
            )
            # If CUDA_VISIBLE_DEVICES is set,
            # device_idx is relative to visible GPUs
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
        if config.use_fsdp and world_size > 1:
            logging.info("Using FSDP with %d processes", world_size)
        elif config.use_ddp and world_size > 1:
            logging.info("Using DDP with %d processes (recommended for 2 GPUs)", world_size)

    # Load sequence dictionaries from FASTA files
    args_train_data = get_arg_train(
        fasta_file=train_fasta_file,
        prottrans_output_path=prottrans_output_path,
        esm2_output_path=esm2_output_path,
        pdb_dir=pdb_dir,
        proc_pdb_output_path=proc_pdb_output_path,
        dssp_output_path=dssp_output_path,
        num_workers=num_workers,
    )
    args_val_data = get_arg_val(
        fasta_file=val_fasta_file,
        prottrans_output_path=prottrans_output_path,
        esm2_output_path=esm2_output_path,
        pdb_dir=pdb_dir,
        proc_pdb_output_path=proc_pdb_output_path,
        dssp_output_path=dssp_output_path,
        num_workers=num_workers,
    )

    train_fasta_data = extract_dataset(args_train_data.fasta_file)
    if is_main_process:
        logging.info("train_fasta_data len: %d", len(train_fasta_data))
    
    # Handle test set (same logic for both K-fold CV and traditional split)
    if test_fasta_file:
        test_fasta_data = extract_dataset(test_fasta_file)
        if is_main_process:
            logging.info("Using separate test set with %d sequences", len(test_fasta_data))
    else:
        test_fasta_data = None
        if is_main_process:
            logging.info("No separate test set provided")
    
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
    # but others negative, which is contradictory)
    positive_set_filtered, multi_fragment_excluded = _filter_multi_fragment_positives(
        positive_set, full_train_data.IDs
    )
    positive_set = positive_set_filtered
    if is_main_process:
        if multi_fragment_excluded:
            excluded_list = sorted(list(multi_fragment_excluded))
            logging.info(
                "Excluded %d multi-fragment positive UniProt IDs:",
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

    # Build test dataset if provided
    if test_fasta_data:
        test_args = get_data_args(
            split="train",  # doesn't matter for test
            fasta_file=test_fasta_file,
            prottrans_output_path=prottrans_output_path,
            esm2_output_path=esm2_output_path,
            pdb_dir=pdb_dir,
            proc_pdb_output_path=proc_pdb_output_path,
            dssp_output_path=dssp_output_path,
            num_workers=num_workers,
        )
        test_data = ProteinGraphDataset(
            test_fasta_data,
            range(len(test_fasta_data)),
            test_args,
            task_list="bp",
        )
        test_loader = DataLoader(
            test_data,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=test_args.num_workers,
            prefetch_factor=1,
        )
    else:
        test_loader = None

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
    
    # Focal Loss for handling extreme class imbalance
    # Focal Loss automatically down-weights easy negatives and focuses on hard examples
    # alpha can be set based on pos_weight if needed, but default alpha=1.0 works well
    # gamma=2.0 is the standard focusing parameter
    focal_loss = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    # Keep pos_weight for potential use in alpha if needed (currently using default alpha=1.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    if use_kfold:
        # K-fold cross-validation with stratified splits to preserve
        # positive/negative ratio in each fold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []
        
        # Get indices for K-fold splitting
        train_indices = list(range(len(full_train_data)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_indices, all_labels)):
            if is_main_process:
                logging.info("=" * 80)
                logging.info("Fold %d/%d", fold_idx + 1, k_folds)
                logging.info("=" * 80)
            
            # Create fold-specific datasets
            train_subset = torch.utils.data.Subset(full_train_data, train_idx)
            
            # Compute train labels for this fold
            train_labels = all_labels[train_idx]
            num_positives_train = int(train_labels.sum())
            num_negatives_train = len(train_labels) - num_positives_train
            
            val_subset = torch.utils.data.Subset(full_train_data, val_idx)

            # Create samplers / batch samplers for distributed vs non-distributed training
            if config.use_fsdp or config.use_ddp:
                # Distributed: use DistributedSampler; per-batch class balance is not enforced,
                # but folds are stratified and class imbalance is handled by FocalLoss.
                train_sampler = DistributedSampler(
                    train_subset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,  # Ensure all ranks process same number of batches
                )
                val_sampler = DistributedSampler(
                    val_subset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=True,  # Ensure all ranks process same number of batches
                )
                train_loader = DataLoader(
                    train_subset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    sampler=train_sampler,
                    drop_last=True,
                    num_workers=args_train_data.num_workers,
                    prefetch_factor=1,
                )
            else:
                # Non-distributed: use custom stratified batch sampler so that each batch
                # has a fixed number of positives (k_pos) and negatives (B - k_pos),
                # with sampling done with or without replacement (controlled by --stratified_without_replacement flag).
                # Or use natural class distribution if --use_stratified_sampling=False.
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
                    )
                elif num_positives_train > 0 and num_negatives_train > 0:
                    pos_indices_fold = [idx for idx in train_idx if all_labels[idx] == 1]
                    neg_indices_fold = [idx for idx in train_idx if all_labels[idx] == 0]
                    # Choose k_pos based on class ratio and batch size, but ensure >=1 and <= batch_size-1
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
                    )
                    train_loader = DataLoader(
                        full_train_data,
                        batch_sampler=stratified_batch_sampler,
                        num_workers=args_train_data.num_workers,
                        prefetch_factor=1,
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
                    )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=val_sampler,
                drop_last=(val_sampler is not None),  # True when using DistributedSampler
                num_workers=args_train_data.num_workers,
                prefetch_factor=1,
            )

            # Initialize model for this fold
            # Strategy: Keep DDP wrapper across folds to avoid NCCL timeout issues
            # For fold 0: Create model and wrap with DDP
            # For fold > 0: Create new model but reuse existing DDP wrapper
            
            if fold_idx == 0:
                # First fold: Create model and wrap with DDP normally
                if is_main_process:
                    logging.info("[rank0] Creating new model for fold 0...")
                
                # Reseed to ensure identical model initialization on all ranks
                seed_base = 1234 + fold_idx
                torch.manual_seed(seed_base)
                np.random.seed(seed_base)
                random.seed(seed_base)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_base)

                # Create model on CPU first
                model = GGN_GO(
                    2324,              # input node feature dimension
                    config.hidden_dim,     # hidden embedding size
                    config.num_layers,      # number of message-passing layers
                    config.augment_eps,    # edge dropout / keep probability
                    config.dropout,         # dropout rate in GNN layers
                    output_dim,        # output dimension (1 logit per protein)
                    config.pooling,    # graph-level pooling type ("MTP" or "GMP")
                    pertub=config.contrast,  # enable contrastive perturbation branch
                )
                
                # Wrap model with DDP or FSDP if enabled
                if config.use_ddp and world_size > 1:
                    model = wrap_model_with_ddp(model, device, rank, world_size)
                elif (config.use_fsdp or config.use_ddp) and world_size > 1:
                    model = wrap_model_with_fsdp(model, config, rank, world_size, local_rank)
                else:
                    model = model.to(device)
            else:
                # Subsequent folds: Replace underlying model but keep DDP wrapper
                if is_main_process:
                    logging.info(f"[rank0] Replacing model for fold {fold_idx} (keeping DDP wrapper)...")
                
                # Clean up optimizer and scaler from previous fold
                try:
                    if 'optimizer' in locals():
                        del optimizer
                    if 'scaler' in locals():
                        del scaler
                except:
                    pass
                
                # Synchronize before creating new model
                if (config.use_fsdp or config.use_ddp) and world_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dist.barrier()
                
                # Reseed to ensure identical model initialization on all ranks
                seed_base = 1234 + fold_idx
                torch.manual_seed(seed_base)
                np.random.seed(seed_base)
                random.seed(seed_base)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_base)

                # Strategy for subsequent folds:
                # - In DDP/FSDP mode, keep the existing wrapper and reinitialize the
                #   underlying model parameters in-place to avoid re-creating the
                #   process group (which has been a source of NCCL timeouts).
                # - In single-GPU / non-distributed mode, create a fresh model.

                if config.use_ddp and world_size > 1 and isinstance(model, DDP):
                    # In-place parameter reinitialization for the wrapped model
                    if is_main_process:
                        logging.info(
                            "[rank0] Reinitializing DDP model parameters in-place "
                            "for fold %d (keeping existing DDP wrapper)" % fold_idx
                        )
                    base_model = model.module

                    # Reinitialize all submodules that implement reset_parameters
                    with torch.no_grad():
                        for submodule in base_model.modules():
                            if hasattr(submodule, "reset_parameters"):
                                submodule.reset_parameters()

                    # Ensure model remains on the correct device
                    base_model.to(device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    if is_main_process:
                        logging.info(
                            "[rank0] In-place parameter reinitialization complete "
                            "for fold %d" % fold_idx
                        )

                elif (config.use_fsdp or config.use_ddp) and world_size > 1 and isinstance(model, FSDP):
                    # For FSDP, we also avoid destroying the wrapper; instead, we
                    # reinitialize parameters of the wrapped module in-place.
                    if is_main_process:
                        logging.info(
                            "[rank0] Reinitializing FSDP model parameters in-place "
                            "for fold %d" % fold_idx
                        )
                    base_model = model.module if hasattr(model, "module") else model

                    with torch.no_grad():
                        for submodule in base_model.modules():
                            if hasattr(submodule, "reset_parameters"):
                                submodule.reset_parameters()

                    base_model.to(device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    if is_main_process:
                        logging.info(
                            "[rank0] In-place FSDP parameter reinitialization "
                            "complete for fold %d" % fold_idx
                        )

                else:
                    # Single GPU or no DDP/FSDP: create a brand-new model instance
                    new_model = GGN_GO(
                        2324,              # input node feature dimension
                    config.hidden_dim,     # hidden embedding size
                    config.num_layers,      # number of message-passing layers
                    config.augment_eps,    # edge dropout / keep probability
                    config.dropout,         # dropout rate in GNN layers
                        output_dim,        # output dimension (1 logit per protein)
                        config.pooling,    # graph-level pooling type ("MTP" or "GMP")
                        pertub=config.contrast,  # enable contrastive perturbation branch
                    )
                    model = new_model.to(device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if is_main_process:
                        logging.info("[rank0] Model replacement complete, all ranks synchronized")

            optimizer = torch.optim.Adam(
                params=model.parameters(),
                **config.optimizer,
            )
            
            # Initialize GradScaler for mixed precision training (if enabled)
            scaler = GradScaler() if config.use_mixed_precision and torch.cuda.is_available() else None
            if is_main_process and scaler is not None:
                logging.info("Using mixed precision training (torch.cuda.amp) to reduce memory usage")

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
            es = 0
            best_eval_loss = None  # Initialize before epoch loop
            best_eval_aupr = None  # Initialize best AUPR for early stopping
            best_eval_recall = None  # Initialize best recall for early stopping
            saved_metrics = False  # Track whether we've already saved per-epoch metrics for this fold

            for ith_epoch in range(config.max_epochs):
                # Set epoch for DistributedSampler to ensure proper shuffling
                if (config.use_fsdp or config.use_ddp) and train_sampler is not None:
                    train_sampler.set_epoch(ith_epoch)
                
                # Synchronize all ranks at the start of each epoch (only log, no barrier to avoid deadlock)
                if is_main_process:
                    logging.info(f"[rank0] Fold {fold_idx+1}, Epoch {ith_epoch+1}/{config.max_epochs} starting")
                
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

                    # Build binary labels from protein IDs
                    y_true = labels_from_batch_names(batch.name, positive_set, device)

                    if config.contrast:
                        # Model returns: prediction + two graph embeddings
                        # (for contrastive views)
                        # Use autocast for mixed precision if enabled
                        if scaler is not None:
                            if AUTOCAST_DEVICE_TYPE:
                                with autocast(device_type='cuda'):
                                    y_pred, g_feat1, g_feat2 = model(
                                        h_V, batch.edge_index, h_E, batch.seq, batch.batch
                                    )
                            else:
                                with autocast(device_type='cuda'):
                                    y_pred, g_feat1, g_feat2 = model(
                                        h_V, batch.edge_index, h_E, batch.seq, batch.batch
                                    )
                        else:
                            y_pred, g_feat1, g_feat2 = model(
                                h_V, batch.edge_index, h_E, batch.seq, batch.batch
                            )

                        # Focal Loss for classification (handles class imbalance automatically)
                        y_pred = y_pred.reshape(-1).float()
                        # Focal Loss handles class imbalance automatically
                        loss_per_sample = focal_loss(y_pred, y_true)
                        loss_per_sample = loss_per_sample * (1.0 + (pos_weight_tensor - 1.0) * y_true)
                        _loss = loss_per_sample.mean()

                        # Contrastive loss between paired graph embeddings (g_feat1, g_feat2)
                        # CRITICAL: Use actual world_size, not config.contrast_world_size
                        # This ensures GatherLayer uses the correct number of ranks
                        actual_world_size = world_size if (config.use_fsdp or config.use_ddp) else 1
                        criterion = NT_Xent(
                            g_feat1.shape[0],        # batch size
                            config.contrast_temp,    # temperature for similarity scaling
                            actual_world_size,      # number of distributed workers (use actual world_size)
                        )
                        # Weight of contrastive term vs BCE
                        cl_loss_raw = criterion(g_feat1, g_feat2)
                        cl_loss = config.contrast_weight * cl_loss_raw

                        # Total loss = supervised classification + (weighted) contrastive regularization
                        loss = _loss + cl_loss
                    else:
                        # Purely supervised setting: model returns only predictions
                        # Use autocast for mixed precision if enabled
                        if scaler is not None:
                            if AUTOCAST_DEVICE_TYPE:
                                with autocast(device_type='cuda'):
                                    y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                            else:
                                with autocast(device_type='cuda'):
                                    y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                        else:
                            y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                        y_pred = y_pred.reshape(-1).float()

                        # Focal Loss handles class imbalance automatically
                        loss_per_sample = focal_loss(y_pred, y_true)
                        loss_per_sample = loss_per_sample * (1.0 + (pos_weight_tensor - 1.0) * y_true)
                        _loss = loss_per_sample.mean()
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
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_max_norm)
                            scaler.step(optimizer)         # step optimizer
                            scaler.update()                # update scaler
                        else:
                            loss.backward()         # compute new gradients via backprop
                            # FIXED: Add gradient clipping to prevent gradient explosion
                            if config.gradient_clip_max_norm > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_max_norm)
                            optimizer.step()        # update model parameters
                    except Exception as e:
                        # Log error with rank information
                        logging.error(f"[rank{rank}] Error during backward pass: Fold {fold_idx+1}, Epoch {ith_epoch+1}, Batch {idx_batch}: {e}")
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

                # ------------------------------------------------------------------
                # Validation
                # ------------------------------------------------------------------
                # AGGRESSIVE DEBUG LOGGING
                logging.info(f"[rank0] Starting validation for Fold {fold_idx+1}, Epoch {ith_epoch+1}/{config.max_epochs}")
                print(f"[DEBUG] Validation block reached: Fold {fold_idx+1}, Epoch {ith_epoch+1}")
                import sys
                sys.stdout.flush()
                if is_main_process:

                    print(f"[DEBUG] is_main_process=True, logging validation start")
                    sys.stdout.flush()
                else:
                    print(f"[DEBUG] is_main_process=False, skipping validation log")
                    sys.stdout.flush()
            
                # Synchronize all ranks before validation
                if (config.use_fsdp or config.use_ddp) and world_size > 1:
                    # barrier() will use the device from init_process_group, no warning needed
                    print(f"[DEBUG] Synchronizing ranks before validation")
                    sys.stdout.flush()
                    dist.barrier()
        
                print(f"[DEBUG] Setting model.eval()")
                sys.stdout.flush()
                model.eval()
                y_pred_all = []
                num_val_batches_processed = 0
                print(f"[DEBUG] Initialized y_pred_all=[], num_val_batches_processed=0")
                sys.stdout.flush()
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
                                        h_V, batch.edge_index, h_E, batch.seq, batch.batch
                                    )
                            else:
                                y_pred, _, _ = model(
                                    h_V, batch.edge_index, h_E, batch.seq, batch.batch
                                )
                        else:
                            if scaler is not None:
                                if AUTOCAST_DEVICE_TYPE:
                                    with autocast(device_type='cuda'):
                                        y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                                else:
                                    with autocast(device_type='cuda'):
                                        y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                            else:
                                y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)

                        y_pred_all.append(y_pred)
                        num_val_batches_processed += 1
                        if idx_batch == 0 or idx_batch % 10 == 0:
                            print(f"[DEBUG] Processed {num_val_batches_processed} validation batches")
                            sys.stdout.flush()
                        if is_main_process:
                            logging.info("%d/%d val_epoch", idx_batch, ith_epoch)

                    # FIXED: Validate that validation batches were processed
                    print(f"[DEBUG] Validation loop completed. num_val_batches_processed={num_val_batches_processed}, len(y_pred_all)={len(y_pred_all)}")
                    sys.stdout.flush()
                    if num_val_batches_processed == 0:
                        error_msg = (
                            f"CRITICAL ERROR: No validation batches processed! val_loader appears to be empty. "
                            f"This indicates a serious problem with validation data loading."
                        )
                        print(f"[DEBUG] ERROR: {error_msg}")
                        sys.stdout.flush()
                        logging.error(error_msg)
                        raise RuntimeError(error_msg)
            
                    if not y_pred_all:
                        error_msg = (
                            f"CRITICAL ERROR: y_pred_all is empty after processing "
                            f"{num_val_batches_processed} validation batches! "
                            f"This should never happen."
                        )
                        print(f"[DEBUG] ERROR: {error_msg}")
                        sys.stdout.flush()
                        logging.error(error_msg)
                        raise RuntimeError(error_msg)
                
                    print(f"[DEBUG] Validation checks passed, proceeding to metrics calculation")
                    sys.stdout.flush()

                    # Concatenate predictions from all batches into a flat tensor
                    y_pred_all = torch.cat(y_pred_all, dim=0).reshape(-1).to(device)
            
                    # Convert to float32 if using mixed precision (autocast outputs FP16)
                    # Always ensure both are float32 to avoid dtype mismatches
                    y_pred_all = y_pred_all.float()
                    y_true_all = y_true_all.float()  # Ensure consistent dtype

                    # Validation BCE over the whole validation set
                    # Compute loss outside autocast context to ensure float32
                    eval_loss = focal_loss(y_pred_all, y_true_all).mean()
                    val_bce_per_epoch.append(float(eval_loss.item()) if hasattr(eval_loss, 'item') else float(eval_loss))

                    # Binary metrics over proteins (using NumPy for sklearn metrics)
                    y_true_np = y_true_all.cpu().numpy()
                    y_scores_np = y_pred_all.cpu().numpy()
                    y_pred_labels = (y_scores_np >= 0.5).astype(int)

                    aupr = average_precision_score(y_true_np, y_scores_np)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true_np, y_pred_labels, average="binary", zero_division=0
                    )

                    print(f"[DEBUG] Calculating metrics: aupr={aupr}, precision={precision}, recall={recall}, f1={f1}")
                    sys.stdout.flush()
                    val_aupr.append(aupr)
                    val_precision_per_epoch.append(float(precision))
                    val_recall_per_epoch.append(float(recall))
                    val_f1_per_epoch.append(float(f1))
                    print(f"[DEBUG] Metrics appended! len(val_aupr)={len(val_aupr)}, val_aupr={val_aupr}")
                    sys.stdout.flush()

                    if is_main_process:
                        logging.info(
                            "%d VAL_epoch ||| loss: %.3f ||| aupr: %.3f ||| precision: %.3f ||| recall: %.3f ||| f1: %.3f",
                            ith_epoch,
                            round(float(eval_loss), 3),
                            round(float(aupr), 3),
                            round(float(precision), 3),
                            round(float(recall), 3),
                            round(float(f1), 3),
                        )

                    # Early stopping logic based on selected metric
                    # Initialize best metrics on first epoch for all ranks
                    if best_eval_aupr is None:
                        best_eval_aupr = aupr
                    if best_eval_recall is None:
                        best_eval_recall = recall
                    if best_eval_loss is None:
                        best_eval_loss = eval_loss
                    
                    # Update early stopping counter based on selected metric
                    # Only reset patience if improvement is at least min_delta
                    metric_improved = False
                    if config.early_stopping_metric == "aupr":
                        if aupr > best_eval_aupr + config.early_stopping_min_delta:
                            best_eval_aupr = aupr
                            metric_improved = True
                    elif config.early_stopping_metric == "recall":
                        if recall > best_eval_recall + config.early_stopping_min_delta:
                            best_eval_recall = recall
                            metric_improved = True
                    elif config.early_stopping_metric == "loss":
                        # For loss, improvement means decrease, so subtract min_delta
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
            
                    # Synchronize early stopping counter across all ranks for distributed training
                    if (config.use_fsdp or config.use_ddp) and world_size > 1:
                        # Convert es to tensor for synchronization
                        es_tensor = torch.tensor(es, dtype=torch.int32, device=device)
                        dist.all_reduce(es_tensor, op=dist.ReduceOp.MAX)
                        es = int(es_tensor.item())
                        # Synchronize best metrics (use MAX for AUPR/recall, MIN for loss)
                        if isinstance(best_eval_aupr, torch.Tensor):
                            best_eval_aupr_tensor = best_eval_aupr.detach().clone().to(device)
                        else:
                            best_eval_aupr_tensor = torch.tensor(best_eval_aupr, dtype=torch.float32, device=device)
                        dist.all_reduce(best_eval_aupr_tensor, op=dist.ReduceOp.MAX)
                        best_eval_aupr = float(best_eval_aupr_tensor.item())
                        
                        if isinstance(best_eval_recall, torch.Tensor):
                            best_eval_recall_tensor = best_eval_recall.detach().clone().to(device)
                        else:
                            best_eval_recall_tensor = torch.tensor(best_eval_recall, dtype=torch.float32, device=device)
                        dist.all_reduce(best_eval_recall_tensor, op=dist.ReduceOp.MAX)
                        best_eval_recall = float(best_eval_recall_tensor.item())
                        
                        if isinstance(best_eval_loss, torch.Tensor):
                            best_eval_loss_tensor = best_eval_loss.detach().clone().to(device)
                        else:
                            best_eval_loss_tensor = torch.tensor(best_eval_loss, dtype=torch.float32, device=device)
                        dist.all_reduce(best_eval_loss_tensor, op=dist.ReduceOp.MIN)
                        best_eval_loss = float(best_eval_loss_tensor.item())
            
                    # Synchronize all ranks before model saving to avoid deadlocks
                    if (config.use_fsdp or config.use_ddp) and world_size > 1:
                        # barrier() will use the device from init_process_group, no warning needed
                        dist.barrier()
            
                    if is_main_process:
                        # FIXED: Save model based on early_stopping_metric, not eval_loss
                        # This ensures saved model is best for the metric you care about (recall/AUPR)
                        should_save = False
                        if ith_epoch == 0:
                            # Always save first epoch
                            should_save = True
                        elif config.early_stopping_metric == "recall":
                            # Save when recall improves
                            if recall > best_eval_recall:
                                should_save = True
                        elif config.early_stopping_metric == "aupr":
                            # Save when AUPR improves
                            if aupr > best_eval_aupr:
                                should_save = True
                        else:  # early_stopping_metric == "loss"
                            # Save when loss improves
                            if eval_loss < best_eval_loss:
                                should_save = True
                        
                        if should_save:
                            model_path = (
                                config.model_save_path
                                + "affinity"
                                + f"{suffix}_fold{fold_idx+1}.pt"
                            )
                            save_model_state_dict(
                                model, model_path, config, rank, world_size
                            )
                            # Log what metric value triggered the save
                            if config.early_stopping_metric == "recall":
                                logging.info(
                                    "Saved model at epoch %d: RECALL=%.4f (improved from %.4f)",
                                    ith_epoch, recall, best_eval_recall if ith_epoch > 0 else 0.0
                                )
                            elif config.early_stopping_metric == "aupr":
                                logging.info(
                                    "Saved model at epoch %d: AUPR=%.4f (improved from %.4f)",
                                    ith_epoch, aupr, best_eval_aupr if ith_epoch > 0 else 0.0
                                )
                            else:
                                logging.info(
                                    "Saved model at epoch %d: LOSS=%.4f (improved from %.4f)",
                                    ith_epoch, eval_loss, best_eval_loss if ith_epoch > 0 else float('inf')
                                )
                            # Flush logging to ensure save messages appear
                            logging.getLogger().handlers[0].flush() if logging.getLogger().handlers else None
                        else:
                            # Determine metric name for logging
                            if config.early_stopping_metric == "aupr":
                                metric_name = "validation AUPR"
                            elif config.early_stopping_metric == "recall":
                                metric_name = "validation recall"
                            else:  # loss
                                metric_name = "validation loss"
                            logging.info(
                                "Early stopping counter %d of %d (%s did not improve)",
                                es, config.early_stopping_patience, metric_name
                            )
                            
                            # FIXED: Also save model at early stopping, but use best metric, not current
                            # Save the best model we've seen so far (which was already saved above)
                            # This is a backup save when early stopping triggers
                            model_path = (
                                config.model_save_path
                                + "affinity"
                                + f"{suffix}_fold{fold_idx+1}_es.pt"
                            )
                            # Load the best model that was already saved (if it exists)
                            # Actually, we should just copy the best model file, or save current best
                            # For now, save current model but log that it's not the best
                            save_model_state_dict(
                                model, model_path, config, rank, world_size
                            )
                            logging.info(
                                "Saved early-stopping model at epoch %d (best %s was at earlier epoch)",
                                ith_epoch, config.early_stopping_metric.upper()
                            )
                            # Flush logging
                            logging.getLogger().handlers[0].flush() if logging.getLogger().handlers else None
            
                    # Synchronize after model saving
                    if (config.use_fsdp or config.use_ddp) and world_size > 1:
                        # barrier() will use the device from init_process_group, no warning needed
                        dist.barrier()

                # Check for early stopping (stop after patience epochs without improvement)
                if es >= config.early_stopping_patience:
                    if is_main_process:
                        # Format best metric value for logging
                        if config.early_stopping_metric == "aupr":
                            best_metric_value = best_eval_aupr
                            metric_name = "AUPR"
                        elif config.early_stopping_metric == "recall":
                            best_metric_value = best_eval_recall
                            metric_name = "Recall"
                        else:  # loss
                            best_metric_value = best_eval_loss
                            metric_name = "Loss"
                        
                        logging.info(
                            "Early stopping triggered after %d epochs without improvement "
                            "(patience=%d, metric=%s, best %s=%.4f, best AUPR=%.4f, best Recall=%.4f, best Loss=%.4f)",
                            es, config.early_stopping_patience, config.early_stopping_metric.upper(),
                            metric_name, best_metric_value, best_eval_aupr, best_eval_recall, best_eval_loss
                        )
                        # Create directory for loss file
                        loss_path = config.loss_save_path + "affinity" + f"{suffix}_fold{fold_idx+1}_val.pt"
                        Path(loss_path).parent.mkdir(parents=True, exist_ok=True)
                        # FIXED: Use len(val_aupr) or len(val_bce_per_epoch) for epochs list instead of train_loss_per_epoch
                        # This ensures epochs list matches validation data even if train_loss_per_epoch is empty
                        num_epochs = max(len(val_aupr), len(val_bce_per_epoch), len(train_loss_per_epoch))
                        if num_epochs == 0:
                            num_epochs = config.max_epochs  # Fallback to max_epochs if all are empty
                        torch.save(
                            {
                                "train_total_loss_per_batch": train_loss,  # Per-batch total losses
                                "train_total_loss_per_epoch": train_loss_per_epoch,  # Per-epoch total losses
                                "train_cls_loss_per_epoch": train_cls_loss_per_epoch,  # Per-epoch classification losses
                                "train_cl_loss_per_epoch": train_cl_loss_per_epoch,    # Per-epoch contrastive losses
                                "val_loss_per_epoch": val_bce_per_epoch,  # Per-epoch validation classification losses
                                "val_aupr": val_aupr,  # Per-epoch validation AUPR
                                "val_precision": val_precision_per_epoch,
                                "val_recall": val_recall_per_epoch,
                                "val_f1": val_f1_per_epoch,
                                "epochs": list(range(num_epochs)),  # FIXED: Use actual number of epochs from validation data
                            },
                            loss_path,
                        )
                        saved_metrics = True
                    break

            # Store fold results
            if is_main_process:
                best_fold_aupr = max(val_aupr) if val_aupr else 0.0
                best_fold_recall = max(val_recall_per_epoch) if val_recall_per_epoch else 0.0
                fold_results.append({
                    "fold": fold_idx + 1,
                    "best_aupr": best_fold_aupr,
                    "best_recall": best_fold_recall,
                    "final_aupr": val_aupr[-1] if val_aupr else 0.0,
                    "best_loss": (
                        min([l.item() if hasattr(l, 'item') else l
                             for l in train_loss])
                        if train_loss else float('inf')
                    ),
                })
                logging.info(
                    "Fold %d/%d completed. Best AUPR: %.4f, Best Recall: %.4f",
                    fold_idx + 1, k_folds, best_fold_aupr, best_fold_recall
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
                    torch.save(
                        {
                            "train_total_loss_per_batch": train_loss,  # Per-batch total losses
                            "train_total_loss_per_epoch": train_loss_per_epoch,  # Per-epoch total losses
                            "train_cls_loss_per_epoch": train_cls_loss_per_epoch,  # Per-epoch classification losses
                            "train_cl_loss_per_epoch": train_cl_loss_per_epoch,    # Per-epoch contrastive losses
                            "val_loss_per_epoch": val_bce_per_epoch,  # Per-epoch validation classification losses
                            "val_aupr": val_aupr,  # Per-epoch validation AUPR
                            "val_precision": val_precision_per_epoch,
                            "val_recall": val_recall_per_epoch,
                            "val_f1": val_f1_per_epoch,
                            "epochs": list(range(num_epochs)),  # FIXED: Use actual number of epochs from validation data
                        },
                        loss_path,
                    )
            
            # Synchronize all ranks after fold completion before starting next fold
            # This prevents NCCL timeout when creating DDP model for next fold
            if (config.use_fsdp or config.use_ddp) and world_size > 1:
                # Multiple barriers to ensure complete synchronization
                dist.barrier()
                # Small delay to ensure all operations complete
                time.sleep(0.5)
                dist.barrier()
                # Final synchronization before next fold
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dist.barrier()
        
        # Aggregate results across folds
        if is_main_process:
            avg_aupr = sum([r["best_aupr"] for r in fold_results]) / len(fold_results)
            std_aupr = np.std([r["best_aupr"] for r in fold_results])
            avg_recall = sum([r["best_recall"] for r in fold_results]) / len(fold_results)
            std_recall = np.std([r["best_recall"] for r in fold_results])
            logging.info("=" * 80)
            logging.info("K-fold CV Results:")
            logging.info("  Average Best AUPR: %.4f ± %.4f", avg_aupr, std_aupr)
            logging.info("  Average Best Recall: %.4f ± %.4f", avg_recall, std_recall)
            for r in fold_results:
                logging.info("  Fold %d: Best AUPR = %.4f, Best Recall = %.4f", r["fold"], r["best_aupr"], r["best_recall"])
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
                },
                loss_path,
            )
        
        # Final evaluation on test set if provided
        if test_loader and is_main_process:
            logging.info("Evaluating on test set...")
            # Train final model on all training data
            # Create model on CPU first - FSDP will handle device placement
            final_model = GGN_GO(
                    2324,              # input node feature dimension
                    config.hidden_dim,     # hidden embedding size
                    config.num_layers,      # number of message-passing layers
                    config.augment_eps,    # edge dropout / keep probability
                    config.dropout,         # dropout rate in GNN layers
                    output_dim,        # output dimension (1 logit per protein)
                config.pooling, pertub=config.contrast,
            )
            
            # Wrap model with DDP or FSDP if enabled
            if config.use_ddp and world_size > 1:
                final_model = wrap_model_with_ddp(final_model, device, rank, world_size)
            elif (config.use_fsdp or config.use_ddp) and world_size > 1:
                final_model = wrap_model_with_fsdp(final_model, config, rank, world_size, local_rank)
            else:
                final_model = final_model.to(device)
            
            final_optimizer = torch.optim.Adam(
                params=final_model.parameters(),
                **config.optimizer,
            )
            
            # Train on all training data
            # Create distributed sampler for distributed training (DDP or FSDP)
            if config.use_fsdp or config.use_ddp:
                all_train_sampler = DistributedSampler(
                    full_train_data,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,  # Ensure all ranks process same number of batches
                )
            else:
                all_train_sampler = None
            
            all_train_loader = DataLoader(
                full_train_data,
                batch_size=config.batch_size,
                shuffle=(all_train_sampler is None),
                sampler=all_train_sampler,
                drop_last=(all_train_sampler is not None),
                num_workers=args_train_data.num_workers,
                prefetch_factor=1,
            )
            
            for ith_epoch in range(config.max_epochs):
                # Set epoch for DistributedSampler to ensure proper shuffling
                if (config.use_fsdp or config.use_ddp) and all_train_sampler is not None:
                    all_train_sampler.set_epoch(ith_epoch)
                
                for idx_batch, batch in enumerate(all_train_loader):
                    batch = batch.to(device)
                    final_model.train()
                    h_V = (batch.node_s, batch.node_v)
                    h_E = (batch.edge_s, batch.edge_v)
                    y_true = labels_from_batch_names(batch.name, positive_set, device)
                    
                    if config.contrast:
                        y_pred, g_feat1, g_feat2 = final_model(
                            h_V, batch.edge_index, h_E, batch.seq, batch.batch
                        )
                        y_pred = y_pred.reshape(-1).float()
                        _loss = focal_loss(y_pred, y_true).mean()
                        # Use actual world size for contrastive loss as well
                        final_world_size = world_size if (config.use_fsdp or config.use_ddp) else 1
                        criterion = NT_Xent(
                            g_feat1.shape[0],
                            config.contrast_temp,
                            final_world_size,
                        )
                        cl_loss = config.contrast_weight * criterion(g_feat1, g_feat2)
                        loss = _loss + cl_loss
                    else:
                        y_pred = final_model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                        y_pred = y_pred.reshape(-1).float()
                        loss = focal_loss(y_pred, y_true).mean()
                    
                    final_optimizer.zero_grad()
                    loss.backward()
                    final_optimizer.step()
            
            # Evaluate on test set
            final_model.eval()
            y_pred_test = []
            y_true_test = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    h_V = (batch.node_s, batch.node_v)
                    h_E = (batch.edge_s, batch.edge_v)
                    labels = labels_from_batch_names(batch.name, positive_set, device)
                    
                    if config.contrast:
                        y_pred, _, _ = final_model(
                            h_V, batch.edge_index, h_E, batch.seq, batch.batch
                        )
                    else:
                        y_pred = final_model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                    
                    y_pred_test.append(y_pred.reshape(-1).cpu().numpy())
                    y_true_test.append(labels.cpu().numpy())
            
            y_pred_test = np.concatenate(y_pred_test)
            y_true_test = np.concatenate(y_true_test)
            test_aupr = average_precision_score(y_true_test, y_pred_test)
            logging.info("Test set AUPR: %.4f", test_aupr)
            
            model_path = (
                config.model_save_path + "affinity" + f"{suffix}_final.pt"
            )
            save_model_state_dict(
                final_model, model_path, config, rank, world_size
            )
    else:
        # Traditional train/val split (original code)
        # Create distributed samplers for distributed training (DDP or FSDP)
        if config.use_fsdp or config.use_ddp:
            train_sampler = DistributedSampler(
                full_train_data,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,  # Ensure all ranks process same number of batches
            )
            val_sampler = DistributedSampler(
                val_data,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=True,  # Ensure all ranks process same number of batches
            )
        else:
            train_sampler = None
            val_sampler = None
        
        train_loader = DataLoader(
            full_train_data,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=(train_sampler is not None),  # True when using DistributedSampler
            num_workers=args_train_data.num_workers,
            prefetch_factor=1,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            drop_last=(val_sampler is not None),  # True when using DistributedSampler
            num_workers=args_val_data.num_workers,
            prefetch_factor=1,
        )
        
        # Create model on CPU first - FSDP will handle device placement
        model = GGN_GO(
                    2324,              # input node feature dimension
                    config.hidden_dim,     # hidden embedding size
                    config.num_layers,      # number of message-passing layers
                    config.augment_eps,    # edge dropout / keep probability
                    config.dropout,         # dropout rate in GNN layers
                    output_dim,        # output dimension (1 logit per protein)
            config.pooling, pertub=config.contrast,
        )
        
        # Wrap model with DDP or FSDP if enabled
        if config.use_ddp and world_size > 1:
            model = wrap_model_with_ddp(model, device, rank, world_size)
        elif (config.use_fsdp or config.use_ddp) and world_size > 1:
            model = wrap_model_with_fsdp(model, config, rank, world_size, local_rank)
        else:
            model = model.to(device)
        
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            **config.optimizer,
        )
        
        # Initialize GradScaler for mixed precision training (if enabled)
        scaler = GradScaler() if config.use_mixed_precision and torch.cuda.is_available() else None
        if is_main_process and scaler is not None:
            logging.info("Using mixed precision training (torch.cuda.amp) to reduce memory usage")
        
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
        es = 0
        best_eval_loss = None  # Initialize before epoch loop
        best_eval_aupr = None  # Initialize best AUPR for early stopping
        best_eval_recall = None  # Initialize best recall for early stopping
        saved_metrics = False  # Track whether we've already saved per-epoch metrics
        
        for ith_epoch in range(config.max_epochs):
            # Set epoch for DistributedSampler to ensure proper shuffling
            if (config.use_fsdp or config.use_ddp) and train_sampler is not None:
                train_sampler.set_epoch(ith_epoch)
            
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

                # Build binary labels from protein IDs
                y_true = labels_from_batch_names(batch.name, positive_set, device)

                if config.contrast:
                    # Model returns: prediction + two graph embeddings
                    # (for contrastive views)
                    # Use autocast for mixed precision if enabled
                    if scaler is not None:
                        with autocast(device_type='cuda'):
                            y_pred, g_feat1, g_feat2 = model(
                                h_V, batch.edge_index, h_E, batch.seq, batch.batch
                            )
                    else:
                        y_pred, g_feat1, g_feat2 = model(
                            h_V, batch.edge_index, h_E, batch.seq, batch.batch
                        )

                    # Focal Loss for classification (handles class imbalance automatically)
                    y_pred = y_pred.reshape(-1).float()
                    # Focal Loss handles class imbalance automatically
                    loss_per_sample = focal_loss(y_pred, y_true)
                    loss_per_sample = loss_per_sample * (1.0 + (pos_weight_tensor - 1.0) * y_true)
                    _loss = loss_per_sample.mean()

                    # Contrastive loss between paired graph embeddings (g_feat1, g_feat2)
                    # Use actual world size here as well
                    actual_world_size_train_all = world_size if (config.use_fsdp or config.use_ddp) else 1
                    criterion = NT_Xent(
                        g_feat1.shape[0],        # batch size
                        config.contrast_temp,    # temperature for similarity scaling
                        actual_world_size_train_all,  # number of distributed workers
                    )
                    # Weight of contrastive term vs BCE
                    cl_loss_raw = criterion(g_feat1, g_feat2)
                    cl_loss = config.contrast_weight * cl_loss_raw

                    # Total loss = supervised classification + (weighted) contrastive regularization
                    loss = _loss + cl_loss
                else:
                    # Purely supervised setting: model returns only predictions
                    # Use autocast for mixed precision if enabled
                    if scaler is not None:
                        with autocast(device_type='cuda'):
                            y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                    else:
                        y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                    y_pred = y_pred.reshape(-1).float()

                    # Focal Loss handles class imbalance automatically
                    loss_per_sample = focal_loss(y_pred, y_true)
                    loss_per_sample = loss_per_sample * (1.0 + (pos_weight_tensor - 1.0) * y_true)
                    _loss = loss_per_sample.mean()
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
                    scaler.scale(loss).backward()  # scale loss and backward
                    scaler.step(optimizer)         # step optimizer
                    scaler.update()                # update scaler
                else:
                    loss.backward()         # compute new gradients via backprop
                    optimizer.step()        # update model parameters
            
            # FIXED: Validate that batches were processed - if not, this is a critical error
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

            # ------------------------------------------------------------------
            # Validation
            # ------------------------------------------------------------------
            # Synchronize all ranks before validation
            if (config.use_fsdp or config.use_ddp) and world_size > 1:
                # barrier() will use the device from init_process_group, no warning needed
                dist.barrier()
            
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
                                    h_V, batch.edge_index, h_E, batch.seq, batch.batch
                                )
                        else:
                            y_pred, _, _ = model(
                                h_V, batch.edge_index, h_E, batch.seq, batch.batch
                            )
                    else:
                        if scaler is not None:
                            if AUTOCAST_DEVICE_TYPE:
                                with autocast(device_type='cuda'):
                                    y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                            else:
                                with autocast(device_type='cuda'):
                                    y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                        else:
                            y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)

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
                eval_loss = focal_loss(y_pred_all, y_true_all).mean()
                val_bce_per_epoch.append(float(eval_loss.item()) if hasattr(eval_loss, 'item') else float(eval_loss))

                # Binary metrics over proteins (using NumPy for sklearn metrics)
                y_true_np = y_true_all.cpu().numpy()
                y_scores_np = y_pred_all.cpu().numpy()
                y_pred_labels = (y_scores_np >= 0.5).astype(int)

                aupr = average_precision_score(y_true_np, y_scores_np)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_np, y_pred_labels, average="binary", zero_division=0
                )

                val_aupr.append(aupr)
                val_precision_per_epoch.append(float(precision))
                val_recall_per_epoch.append(float(recall))
                val_f1_per_epoch.append(float(f1))

                if is_main_process:
                    logging.info(
                        "%d VAL_epoch ||| loss: %.3f ||| aupr: %.3f ||| precision: %.3f ||| recall: %.3f ||| f1: %.3f",
                        ith_epoch,
                        round(float(eval_loss), 3),
                        round(float(aupr), 3),
                        round(float(precision), 3),
                        round(float(recall), 3),
                        round(float(f1), 3),
                    )

                # Early stopping logic based on selected metric
                # Initialize best metrics on first epoch for all ranks
                if best_eval_aupr is None:
                    best_eval_aupr = aupr
                if best_eval_recall is None:
                    best_eval_recall = recall
                if best_eval_loss is None:
                    best_eval_loss = eval_loss
                
                # Update early stopping counter based on selected metric
                # Only reset patience if improvement is at least min_delta
                metric_improved = False
                if config.early_stopping_metric == "aupr":
                    if aupr > best_eval_aupr + config.early_stopping_min_delta:
                        best_eval_aupr = aupr
                        metric_improved = True
                elif config.early_stopping_metric == "recall":
                    if recall > best_eval_recall + config.early_stopping_min_delta:
                        best_eval_recall = recall
                        metric_improved = True
                elif config.early_stopping_metric == "loss":
                    # For loss, improvement means decrease, so subtract min_delta
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
                
                # Synchronize early stopping counter across all ranks for distributed training
                if (config.use_fsdp or config.use_ddp) and world_size > 1:
                    # Convert es to tensor for synchronization
                    es_tensor = torch.tensor(es, dtype=torch.int32, device=device)
                    dist.all_reduce(es_tensor, op=dist.ReduceOp.MAX)
                    es = int(es_tensor.item())
                    # Synchronize best metrics (use MAX for AUPR/recall, MIN for loss)
                    if isinstance(best_eval_aupr, torch.Tensor):
                        best_eval_aupr_tensor = best_eval_aupr.detach().clone().to(device)
                    else:
                        best_eval_aupr_tensor = torch.tensor(best_eval_aupr, dtype=torch.float32, device=device)
                    dist.all_reduce(best_eval_aupr_tensor, op=dist.ReduceOp.MAX)
                    best_eval_aupr = float(best_eval_aupr_tensor.item())
                    
                    if isinstance(best_eval_recall, torch.Tensor):
                        best_eval_recall_tensor = best_eval_recall.detach().clone().to(device)
                    else:
                        best_eval_recall_tensor = torch.tensor(best_eval_recall, dtype=torch.float32, device=device)
                    dist.all_reduce(best_eval_recall_tensor, op=dist.ReduceOp.MAX)
                    best_eval_recall = float(best_eval_recall_tensor.item())
                    
                    if isinstance(best_eval_loss, torch.Tensor):
                        best_eval_loss_tensor = best_eval_loss.detach().clone().to(device)
                    else:
                        best_eval_loss_tensor = torch.tensor(best_eval_loss, dtype=torch.float32, device=device)
                    dist.all_reduce(best_eval_loss_tensor, op=dist.ReduceOp.MIN)
                    best_eval_loss = float(best_eval_loss_tensor.item())
                
                # Synchronize all ranks before model saving to avoid deadlocks
                if (config.use_fsdp or config.use_ddp) and world_size > 1:
                    # barrier() will use the device from init_process_group, no warning needed
                    dist.barrier()
                
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
                    else:
                        # Determine metric name for logging
                        if config.early_stopping_metric == "aupr":
                            metric_name = "validation AUPR"
                        elif config.early_stopping_metric == "recall":
                            metric_name = "validation recall"
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
                        # Flush logging
                        logging.getLogger().handlers[0].flush() if logging.getLogger().handlers else None
                
                # Synchronize after model saving
                if (config.use_fsdp or config.use_ddp) and world_size > 1:
                    # barrier() will use the device from init_process_group, no warning needed
                    dist.barrier()

                # Check for early stopping (stop after patience epochs without improvement)
                if es >= config.early_stopping_patience:
                    if is_main_process:
                        # Format best metric value for logging
                        if config.early_stopping_metric == "aupr":
                            best_metric_value = best_eval_aupr
                            metric_name = "AUPR"
                        elif config.early_stopping_metric == "recall":
                            best_metric_value = best_eval_recall
                            metric_name = "Recall"
                        else:  # loss
                            best_metric_value = best_eval_loss
                            metric_name = "Loss"
                        
                        logging.info(
                            "Early stopping triggered after %d epochs without improvement "
                            "(patience=%d, metric=%s, best %s=%.4f, best AUPR=%.4f, best Recall=%.4f, best Loss=%.4f)",
                            es, config.early_stopping_patience, config.early_stopping_metric.upper(),
                            metric_name, best_metric_value, best_eval_aupr, best_eval_recall, best_eval_loss
                        )
                        # Create directory for loss file
                        loss_path = config.loss_save_path + "affinity" + f"{suffix}_val.pt"
                        Path(loss_path).parent.mkdir(parents=True, exist_ok=True)
                        # FIXED: Use len(val_aupr) or len(val_bce_per_epoch) for epochs list instead of train_loss_per_epoch
                        num_epochs = max(len(val_aupr), len(val_bce_per_epoch), len(train_loss_per_epoch))
                        if num_epochs == 0:
                            num_epochs = config.max_epochs  # Fallback to max_epochs if all are empty
                        torch.save(
                            {
                                "train_total_loss_per_batch": train_loss,  # Per-batch total losses
                                "train_total_loss_per_epoch": train_loss_per_epoch,  # Per-epoch total losses
                                "train_cls_loss_per_epoch": train_cls_loss_per_epoch,  # Per-epoch classification losses
                                "train_cl_loss_per_epoch": train_cl_loss_per_epoch,    # Per-epoch contrastive losses
                                "val_loss_per_epoch": val_bce_per_epoch,  # Per-epoch validation classification losses
                                "val_aupr": val_aupr,  # Per-epoch validation AUPR
                                "val_precision": val_precision_per_epoch,
                                "val_recall": val_recall_per_epoch,
                                "val_f1": val_f1_per_epoch,
                                "epochs": list(range(num_epochs)),  # FIXED: Use actual number of epochs from validation data
                            },
                            loss_path,
                        )
                        saved_metrics = True
                    break
        
        # Final evaluation on test set if provided
        if test_loader and is_main_process:
            logging.info("Evaluating on test set...")
            model.eval()
            y_pred_test = []
            y_true_test = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    h_V = (batch.node_s, batch.node_v)
                    h_E = (batch.edge_s, batch.edge_v)
                    labels = labels_from_batch_names(batch.name, positive_set, device)
                    
                    if config.contrast:
                        y_pred, _, _ = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                    else:
                        y_pred = model(h_V, batch.edge_index, h_E, batch.seq, batch.batch)
                    
                    y_pred_test.append(y_pred.reshape(-1).cpu().numpy())
                    y_true_test.append(labels.cpu().numpy())
            
            y_pred_test = np.concatenate(y_pred_test)
            y_true_test = np.concatenate(y_true_test)
            test_aupr = average_precision_score(y_true_test, y_pred_test)
            logging.info("Test set AUPR: %.4f", test_aupr)

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
            torch.save(
                {
                    "train_total_loss_per_batch": train_loss,  # Per-batch total losses
                    "train_total_loss_per_epoch": train_loss_per_epoch,  # Per-epoch total losses
                    "train_cls_loss_per_epoch": train_cls_loss_per_epoch,  # Per-epoch classification losses
                    "train_cl_loss_per_epoch": train_cl_loss_per_epoch,    # Per-epoch contrastive losses
                    "val_loss_per_epoch": val_bce_per_epoch,  # Per-epoch validation classification losses
                    "val_aupr": val_aupr,  # Per-epoch validation AUPR
                    "val_precision": val_precision_per_epoch,
                    "val_recall": val_recall_per_epoch,
                    "val_f1": val_f1_per_epoch,
                    "epochs": list(range(num_epochs)),  # FIXED: Use actual number of epochs from validation data
                },
                loss_path,
            )


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
        help="CUDA device index (used when not using FSDP2)",
    )
    p.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated list of GPU IDs to use (e.g., '0,1,2,3'). "
            "Sets CUDA_VISIBLE_DEVICES. If not specified, all available GPUs "
            "are used. When using FSDP2, this restricts which GPUs are "
            "visible to the process."
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
        help="Gamma parameter for Focal Loss (focusing parameter, default: 2.0). Higher gamma focuses more on hard examples.",
    )
    p.add_argument(
        "--log_file",
        type=str,
        default="logs/train_affinity.log",
        help="Path to the log file where training logs will be written",
    )
    p.add_argument("--batch_size", type=int, default=32, help="batch size")
    p.add_argument(
        "--stratified_without_replacement",
        nargs="?",
        const=True,
        default=True,
        type=lambda x: False if x and x.lower() == "false" else True,
        help="Sample without replacement in StratifiedPosNegBatchSampler (default: True, uses without replacement sampling). "
             "Set to False to use replacement sampling. When True, samples are cycled through without reuse within an epoch, which may reduce overfitting."
    )
    p.add_argument(
        "--stop_epoch_when_exhausted",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: False if x and x.lower() == "false" else True,
        help="When using stratified_without_replacement, stop the epoch once either the positive or negative pool is exhausted, instead of cycling (default: False)."
    )
    p.add_argument(
        "--use_stratified_sampling",
        nargs="?",
        const=True,
        default=True,
        type=lambda x: False if x and x.lower() == "false" else True,
        help="Use StratifiedPosNegBatchSampler to enforce balanced batches (default: True). Set to False to use natural class distribution sampling without balance."
    )
    p.add_argument(
        "--balanced_epoch_sampling",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: False if x and x.lower() == "false" else True,
        help="When using stratified sampling, ensure each epoch sees equal numbers of positives and negatives by oversampling the minority class (default: False)."
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
        help="Minimum improvement required to reset patience counter (default: 0.01 for recall, 0.001 for AUPR)",
    )
    p.add_argument(
        "--early_stopping_metric",
        type=str,
        default="recall",
        choices=["aupr", "recall", "loss"],
        help="Metric to use for early stopping: 'aupr' (Area Under PR Curve), 'recall' (True Positive Rate), or 'loss' (Validation Loss). Default: 'recall' (recommended for datasets with noisy negative labels).",
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
        "--test_fasta_file",
        type=str,
        default=None,
        help="Path to test FASTA file (for final evaluation after K-fold CV)",
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
        "--num_workers",
        type=int,
        default=None,
        help="Number of data loader workers (default: 8)",
    )
    p.add_argument(
        "--use_ddp",
        default=False,
        type=str2bool,
        help=(
            "Whether to use DDP (DistributedDataParallel) for distributed training "
            "(default: False). Recommended for 2 GPUs when model fits on each GPU. "
            "Simpler than FSDP."
        ),
    )
    p.add_argument(
        "--use_fsdp",
        default=False,
        type=str2bool,
        help=(
            "Whether to use FSDP (FullyShardedDataParallel) for distributed training "
            "(default: False). Use only if model is too large for single GPU. "
            "For 2 GPUs, DDP (--use_ddp) is usually simpler and sufficient."
        ),
    )
    p.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="FULL_SHARD",
        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"],
        help="FSDP sharding strategy (default: FULL_SHARD)",
    )
    p.add_argument(
        "--fsdp_mixed_precision",
        default=False,
        type=str2bool,
        help="Whether to use mixed precision with FSDP (default: False)",
    )
    p.add_argument(
        "--fsdp_sync_module_states",
        default=False,
        type=str2bool,
        help="Whether to sync module states across processes in FSDP (default: False)",
    )
    p.add_argument(
        "--use_mixed_precision",
        default=False,
        type=str2bool,
        help=(
            "Whether to use mixed precision training with DDP (default: False). "
            "Uses torch.cuda.amp to reduce memory usage by ~50%. "
            "Recommended when experiencing OOM errors."
        ),
    )

    p.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer (default: 5e-5). Reduced from 1e-4 for more stable learning.",
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
        "--augment_eps",
        type=float,
        default=0.95,
        help="Edge dropout / keep probability for augmentation (default: 0.95)",
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
    config.use_stratified_sampling = args.use_stratified_sampling
    config.balanced_epoch_sampling = args.balanced_epoch_sampling
    config.stratified_without_replacement = args.stratified_without_replacement
    config.stop_epoch_when_exhausted = args.stop_epoch_when_exhausted
    config.max_epochs = args.max_epochs
    config.early_stopping_patience = args.early_stopping_patience
    config.early_stopping_min_delta = args.early_stopping_min_delta
    config.early_stopping_metric = args.early_stopping_metric.lower()  # Normalize to lowercase

    
    config.optimizer["lr"] = args.learning_rate
    config.gradient_clip_max_norm = args.gradient_clip_max_norm
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.num_layers
    config.dropout = args.dropout
    config.augment_eps = args.augment_eps
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
    config.pos_weight = args.pos_weight
    config.focal_alpha = args.focal_alpha
    config.focal_gamma = args.focal_gamma
    config.use_ddp = args.use_ddp
    config.use_fsdp = args.use_fsdp
    config.fsdp_sharding_strategy = args.fsdp_sharding_strategy
    config.fsdp_mixed_precision = args.fsdp_mixed_precision
    config.fsdp_sync_module_states = args.fsdp_sync_module_states
    config.use_mixed_precision = args.use_mixed_precision
    
    # Validate that only one distributed method is selected
    if config.use_ddp and config.use_fsdp:
        raise ValueError("Cannot use both DDP and FSDP simultaneously. Choose one.")

    train(
        config,
        args.suffix,
        args.aptamer_list_file,
        train_fasta_file=args.train_fasta_file,
        val_fasta_file=args.val_fasta_file,
        test_fasta_file=args.test_fasta_file,
        prottrans_output_path=args.prottrans_output_path,
        esm2_output_path=args.esm2_output_path,
        pdb_dir=args.pdb_dir,
        proc_pdb_output_path=args.proc_pdb_output_path,
        dssp_output_path=args.dssp_output_path,
        num_workers=args.num_workers,
        k_folds=args.k_folds,
        use_kfold=args.use_kfold,
        gpu_ids=args.gpu_ids,
    )
