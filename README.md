# GGN-Apta

GGN-Apta scores how likely a protein is to be an aptamer target. This repository trains the released model, evaluates it, and writes per-protein probabilities. The five released checkpoints are the 2026-09-07 baseline AUPRC early-stopping weights (homology identity 30%, seed 42, positional embeddings off).

Manuscript figures use an earlier `no_pos_emb` run from 2026-07-23. Those weights are not in this Release. An October EF@10% checkpoint experiment does not replace the released weights.

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

## Checkpoints

Download `model_released_fold1_es.pt` through `model_released_fold5_es.pt` from GitHub Release `v0.1.0` into one directory and pass that directory as `MODEL_DIR`. The files are not stored in git. Evaluation and prediction load them with the training module (`build_model`) and require every checkpoint key to match. The loader accepts names of the form `model_*_foldN_es.pt`.

## What you need to supply

This repository does not include sequences, structures, or precomputed embeddings. Set these paths before training, evaluation, or prediction:

- `APTAMER_LIST_FILE`: positive UniProt accessions, one per line
- `TRAIN_FASTA_FILE` or `TEST_FASTA_FILE` / `FASTA_FILE`
- `PROTTRANS_OUTPUT_PATH`: ProtTrans embeddings
- `ESM2_OUTPUT_PATH`: ESM-2 embeddings
- `PDB_DIR`: input structures
- `PROC_PDB_OUTPUT_PATH`: processed coordinates
- `DSSP_OUTPUT_PATH`: secondary-structure features

`PROJECT_ROOT` defaults to this repository. `DATA_ROOT` defaults to `PROJECT_ROOT`. `PYTHON_BIN` defaults to `python3`.

## Setup

```bash
conda env create -f environment.yml
conda activate ggn-apta
```

PyTorch and PyTorch Geometric need a CUDA build that matches the machine. The versions in `environment.yml` match the environment used to train the released model (Python 3.11, PyTorch 2.9.1, PyG 2.7.0, torch-scatter 2.1.2). ESM-2, ProtTrans, and AlphaFold DB stay under their own terms.

## Train

```bash
APTAMER_LIST_FILE=/path/to/positives.txt \
TRAIN_FASTA_FILE=/path/to/train.fasta.gz \
OUTPUT_DIR=/path/to/run \
PROTTRANS_OUTPUT_PATH=/path/to/prottrans \
ESM2_OUTPUT_PATH=/path/to/esm2 \
PDB_DIR=/path/to/structures \
PROC_PDB_OUTPUT_PATH=/path/to/coordinates \
DSSP_OUTPUT_PATH=/path/to/dssp \
bash scripts/train.sh
```

Set `POS_WEIGHT` to skip the 0.5-times automatic positive weight.

## Evaluate

```bash
MODEL_DIR=/path/to/checkpoints \
TEST_FASTA_FILE=/path/to/test.fasta.gz \
APTAMER_LIST_FILE=/path/to/positives.txt \
OUTPUT_FILE=/path/to/metrics.tsv \
PROTTRANS_OUTPUT_PATH=/path/to/prottrans \
ESM2_OUTPUT_PATH=/path/to/esm2 \
PDB_DIR=/path/to/structures \
PROC_PDB_OUTPUT_PATH=/path/to/coordinates \
DSSP_OUTPUT_PATH=/path/to/dssp \
bash scripts/evaluate.sh
```

## Predict

```bash
MODEL_DIR=/path/to/checkpoints \
FASTA_FILE=/path/to/proteins.fasta.gz \
APTAMER_LIST_FILE=/path/to/positives.txt \
OUTPUT_TSV=/path/to/predictions.tsv \
PROTTRANS_OUTPUT_PATH=/path/to/prottrans \
ESM2_OUTPUT_PATH=/path/to/esm2 \
PDB_DIR=/path/to/structures \
PROC_PDB_OUTPUT_PATH=/path/to/coordinates \
DSSP_OUTPUT_PATH=/path/to/dssp \
bash scripts/predict.sh
```

`OUTPUT_TSV` has one row per protein, a probability for each fold, and `ensemble_proba` as their mean.

## Checkpoint key check

```bash
python scripts/check_checkpoint_keys.py --checkpoint /path/to/model_released_fold1_es.pt
```

This builds `GGN_GO` from the training script and loads the file with `strict=True`.

## License

MIT. See `LICENSE` and `NOTICE`. Upstream copyrights for GGN-GO, the geometric vector perceptron, and PyTorch Geometric stay in `NOTICE`.
