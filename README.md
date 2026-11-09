<h1 align="center">
  <img src="docs/ggn-apta-logo.png" alt="GGN-Apta" width="280">
</h1>

## Overview

GGN-Apta ranks human proteins by association with reported aptamer-target status. It combines sequence representations with AlphaFold2-predicted residue structure in a geometric graph neural network. The score runs from 0 to 1. It estimates association with the observed AptaDB annotation, and it is not a calibrated probability of aptamer binding or selection success.

Experimentally supported targets curated in AptaDB are observed positives. Proteins absent from the mapped AptaDB set are unlabeled background. For training, listed proteins are labeled 1 and unlisted proteins are labeled 0. Unlisted proteins are not confirmed negatives.

Each predicted structure is a residue graph. Backbone coordinates come from the AlphaFold2 model, and each residue is linked to its 50 nearest spatial neighbors. Node scalar features are amino-acid embeddings, backbone-dihedral encodings, DSSP descriptors, ProtTrans embeddings from ProtT5-XL-UniRef50, and ESM-2 embeddings from the 650M-parameter model. Node vector features encode local backbone direction and a backbone-derived proxy for side-chain direction. Edge scalar features encode inter-residue distance, and edge vector features encode the normalized direction between residues.

Four geometric vector perceptron layers update the graph. In each layer, vector norms are concatenated with the scalar inputs to update the scalar channel, and scalar gates modulate the vectors to update the vector channel. Global average pooling then feeds a multilayer perceptron that outputs the association score.

The starting set contained 23,391 structural fragments from 20,504 UniProt accessions. Two AptaDB-listed accessions represented by more than one fragment were excluded from model development and test evaluation. The remaining set has 23,374 fragments from 20,502 accessions, including 121 listed-positive fragments from 121 accessions.

## What you need to supply

This repository does not include sequences, structures, or precomputed embeddings. Pass these paths to `ggn-apta train`, `ggn-apta evaluate`, and `ggn-apta predict`.

`--aptamer-list` lists positive UniProt accessions, one per line. `--prottrans`, `--esm2`, `--pdb-dir`, `--coordinates`, and `--dssp` are ProtTrans embeddings, ESM-2 embeddings, structures, processed coordinates, and DSSP features.

## Setup

```bash
conda env create -f environment.yml
conda activate ggn-apta
pip install -e .
```

PyTorch and PyTorch Geometric need a CUDA build that matches the machine. The versions in `environment.yml` match the environment used to train the released model (Python 3.11, PyTorch 2.9.1, PyG 2.7.0, torch-scatter 2.1.2). ESM-2, ProtTrans, and AlphaFold DB stay under their own terms.

## Train

```bash
ggn-apta train \
  --aptamer-list /path/to/positives.txt \
  --train-fasta /path/to/train.fasta.gz \
  --output-dir /path/to/run \
  --prottrans /path/to/prottrans \
  --esm2 /path/to/esm2 \
  --pdb-dir /path/to/structures \
  --coordinates /path/to/coordinates \
  --dssp /path/to/dssp
```

Pass `--pos-weight` to skip the 0.5-times automatic positive weight. `ggn-apta train --help` lists the pinned released-model settings.

## Evaluate

```bash
ggn-apta evaluate \
  --model-dir /path/to/checkpoints \
  --test-fasta /path/to/test.fasta.gz \
  --output /path/to/metrics.tsv \
  --aptamer-list /path/to/positives.txt \
  --prottrans /path/to/prottrans \
  --esm2 /path/to/esm2 \
  --pdb-dir /path/to/structures \
  --coordinates /path/to/coordinates \
  --dssp /path/to/dssp
```

## Predict

```bash
ggn-apta predict \
  --model-dir /path/to/checkpoints \
  --fasta /path/to/proteins.fasta.gz \
  --output /path/to/predictions.tsv \
  --aptamer-list /path/to/positives.txt \
  --prottrans /path/to/prottrans \
  --esm2 /path/to/esm2 \
  --pdb-dir /path/to/structures \
  --coordinates /path/to/coordinates \
  --dssp /path/to/dssp
```

`--output` has one row per protein, a probability for each fold, and `ensemble_proba` as their mean.

## Checkpoints

Download `model_released_fold1_es.pt` through `model_released_fold5_es.pt` from GitHub Release `v0.1.0` into one directory and pass that directory to `--model-dir`. The files are not stored in git. Evaluation and prediction load them with the training module (`build_model`) and require every checkpoint key to match. The loader accepts names of the form `model_*_foldN_es.pt`.

## Checkpoint key check

```bash
ggn-apta check-checkpoint --checkpoint /path/to/model_released_fold1_es.pt
```

This builds `GGN_GO` from the training script and loads the file with `strict=True`.

## Released model

| Setting | Value |
| --- | --- |
| Focal alpha | 3.7 |
| Focal gamma | 1.5 |
| Positive weight | 0.5 times the training-set negative/positive ratio |
| Learning rate | 5e-5 |
| Weight decay | 1e-4 |
| Dropout | 0.60 |
| Batch size | 6 |
| k_pos | 1 |
| Hidden size / layers | 640 / 4 |
| Pooling | mean (GAP) |
| Contrastive weight / temperature | 1.0 / 0.1 |
| Edge keep probability | 0.60 |
| Early stopping | validation AUPRC, patience 5, min delta 0.001 |
| Schedule | exponential, gamma 0.950 |
| Folds | 5 |
| Positional embeddings | off |

## License

MIT. See `LICENSE` and `NOTICE`. Upstream copyrights for GGN-GO, the geometric vector perceptron, and PyTorch Geometric stay in `NOTICE`.
