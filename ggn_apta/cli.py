"""Argument parser for the released-model train, evaluate, and predict commands."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ggn_apta.pos_weight import CliError, positive_weight_text, require_positive_pw_mode
from ggn_apta.released import (
    CHECK_DESCRIPTION,
    EVALUATE_DESCRIPTION,
    PREDICT_DESCRIPTION,
    TRAIN_DESCRIPTION,
    evaluate_flags,
    predict_flags,
    train_flags,
)

ExecFn = Callable[[str, list[str]], None]
WeightFn = Callable[[str, str, float], str]

TRAIN_SCRIPT = Path(
    "Scripts/Sub/train_fixed_lr_scheduler_no_ddp_fsdp_allow_sched_factor_1_augment_eps_edge_dropout.py"
)
EVALUATE_SCRIPT = Path("Scripts/Sub/evaluate_test_set.py")
PREDICT_SCRIPT = Path("Scripts/Sub/export_protein_predictions.py")
CHECK_SCRIPT = Path("scripts/check_checkpoint_keys.py")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _missing(flag: str, path: str) -> CliError:
    shown = path if path else "<unset>"
    return CliError(f"ERROR: {flag} is missing: {shown}")


def _require_nonempty(path: str, flag: str) -> None:
    if path == "":
        raise _missing(flag, path)


def _require_file(path: str, flag: str) -> None:
    _require_nonempty(path, flag)
    if not Path(path).is_file():
        raise _missing(flag, path)


def _require_dir(path: str, flag: str) -> None:
    _require_nonempty(path, flag)
    if not Path(path).is_dir():
        raise _missing(flag, path)


def _require_feature_args(args: argparse.Namespace) -> None:
    _require_nonempty(args.prottrans, "--prottrans")
    _require_nonempty(args.esm2, "--esm2")
    _require_nonempty(args.pdb_dir, "--pdb-dir")
    _require_nonempty(args.coordinates, "--coordinates")
    _require_nonempty(args.dssp, "--dssp")


def _mkdir(path: Path, flag: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CliError(f"ERROR: {flag} could not be created: {path}: {exc}") from exc


def build_train_argv(args: argparse.Namespace, *, pos_weight: str, root: Path) -> list[str]:
    script = root / TRAIN_SCRIPT
    return [
        sys.executable,
        str(script),
        *train_flags(
            aptamer_list=args.aptamer_list,
            train_fasta=args.train_fasta,
            output_dir=args.output_dir,
            prottrans=args.prottrans,
            esm2=args.esm2,
            pdb_dir=args.pdb_dir,
            coordinates=args.coordinates,
            dssp=args.dssp,
            pos_weight=pos_weight,
            num_workers=args.num_workers,
            device=args.device,
        ),
    ]


def build_evaluate_argv(args: argparse.Namespace, *, root: Path) -> list[str]:
    script = root / EVALUATE_SCRIPT
    return [
        sys.executable,
        str(script),
        *evaluate_flags(
            model_dir=args.model_dir,
            test_fasta=args.test_fasta,
            aptamer_list=args.aptamer_list,
            output=args.output,
            checkpoint_mode=args.checkpoint_mode,
            prottrans=args.prottrans,
            esm2=args.esm2,
            pdb_dir=args.pdb_dir,
            coordinates=args.coordinates,
            dssp=args.dssp,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        ),
    ]


def build_predict_argv(args: argparse.Namespace, *, root: Path) -> list[str]:
    script = root / PREDICT_SCRIPT
    return [
        sys.executable,
        str(script),
        *predict_flags(
            model_dir=args.model_dir,
            fasta=args.fasta,
            aptamer_list=args.aptamer_list,
            output=args.output,
            checkpoint_mode=args.checkpoint_mode,
            prottrans=args.prottrans,
            esm2=args.esm2,
            pdb_dir=args.pdb_dir,
            coordinates=args.coordinates,
            dssp=args.dssp,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        ),
    ]


def build_check_argv(args: argparse.Namespace, *, root: Path) -> list[str]:
    script = root / CHECK_SCRIPT
    return [sys.executable, str(script), "--checkpoint", args.checkpoint]


def run_train(
    args: argparse.Namespace,
    *,
    root: Path | None = None,
    exec_fn: ExecFn = os.execv,
    weight_fn: WeightFn = positive_weight_text,
) -> None:
    root = repository_root() if root is None else root
    _require_file(args.aptamer_list, "--aptamer-list")
    _require_file(args.train_fasta, "--train-fasta")
    _require_nonempty(args.output_dir, "--output-dir")
    _require_dir(args.prottrans, "--prottrans")
    _require_dir(args.esm2, "--esm2")
    _require_dir(args.pdb_dir, "--pdb-dir")
    _require_dir(args.coordinates, "--coordinates")
    _require_dir(args.dssp, "--dssp")
    _mkdir(Path(args.output_dir), "--output-dir")
    if args.pos_weight is None or args.pos_weight == "":
        require_positive_pw_mode(args.pw_mode)
        pos_weight = weight_fn(args.train_fasta, args.aptamer_list, args.pw_mode)
    else:
        pos_weight = args.pos_weight
    argv = build_train_argv(args, pos_weight=pos_weight, root=root)
    exec_fn(sys.executable, argv)


def run_evaluate(
    args: argparse.Namespace,
    *,
    root: Path | None = None,
    exec_fn: ExecFn = os.execv,
) -> None:
    root = repository_root() if root is None else root
    _require_dir(args.model_dir, "--model-dir")
    _require_file(args.test_fasta, "--test-fasta")
    _require_file(args.aptamer_list, "--aptamer-list")
    _require_nonempty(args.output, "--output")
    _require_feature_args(args)
    argv = build_evaluate_argv(args, root=root)
    exec_fn(sys.executable, argv)


def run_predict(
    args: argparse.Namespace,
    *,
    root: Path | None = None,
    exec_fn: ExecFn = os.execv,
) -> None:
    root = repository_root() if root is None else root
    _require_dir(args.model_dir, "--model-dir")
    _require_file(args.fasta, "--fasta")
    _require_file(args.aptamer_list, "--aptamer-list")
    _require_nonempty(args.output, "--output")
    _require_feature_args(args)
    _mkdir(Path(args.output).parent, "--output")
    argv = build_predict_argv(args, root=root)
    exec_fn(sys.executable, argv)


def run_check(
    args: argparse.Namespace,
    *,
    root: Path | None = None,
    exec_fn: ExecFn = os.execv,
) -> None:
    root = repository_root() if root is None else root
    _require_file(args.checkpoint, "--checkpoint")
    argv = build_check_argv(args, root=root)
    exec_fn(sys.executable, argv)


def _add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--aptamer-list",
        required=True,
        help="Positive UniProt accessions, one per line.",
    )
    parser.add_argument("--prottrans", required=True, help="ProtTrans embeddings directory.")
    parser.add_argument("--esm2", required=True, help="ESM-2 embeddings directory.")
    parser.add_argument("--pdb-dir", required=True, help="Structure directory.")
    parser.add_argument("--coordinates", required=True, help="Processed coordinates directory.")
    parser.add_argument("--dssp", required=True, help="DSSP features directory.")


def _add_runtime_flags(parser: argparse.ArgumentParser, *, batch_size: bool) -> None:
    if batch_size:
        parser.add_argument("--batch-size", type=int, default=6, help="Batch size (default: 6).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Data loader workers (default: 4).",
    )
    parser.add_argument("--device", default="0", help="CUDA device index (default: 0).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ggn-apta",
        description="Train, evaluate, and run the released GGN-Apta model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    formatter = argparse.RawDescriptionHelpFormatter

    train = commands.add_parser(
        "train",
        description=TRAIN_DESCRIPTION,
        formatter_class=formatter,
        help="Train the released model.",
    )
    _add_shared_paths(train)
    train.add_argument("--train-fasta", required=True, help="Training FASTA, plain or gzipped.")
    train.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs.")
    train.add_argument(
        "--pos-weight",
        default=None,
        help="Skip the automatic 0.5-times positive weight and use this value.",
    )
    train.add_argument(
        "--pw-mode",
        type=float,
        default=0.5,
        help="Multiplier for the automatic negative/positive ratio (default: 0.5).",
    )
    _add_runtime_flags(train, batch_size=False)
    train.set_defaults(run=run_train)

    evaluate = commands.add_parser(
        "evaluate",
        description=EVALUATE_DESCRIPTION,
        formatter_class=formatter,
        help="Score a FASTA with the released five-fold ensemble.",
    )
    evaluate.add_argument("--model-dir", required=True, help="Directory of released checkpoints.")
    evaluate.add_argument("--test-fasta", required=True, help="Test FASTA, plain or gzipped.")
    evaluate.add_argument("--output", required=True, help="Metrics output file.")
    _add_shared_paths(evaluate)
    evaluate.add_argument(
        "--checkpoint-mode",
        choices=["es", "es_ef10", "final"],
        default="es",
        help="Checkpoint suffix to load (default: es).",
    )
    _add_runtime_flags(evaluate, batch_size=True)
    evaluate.set_defaults(run=run_evaluate)

    predict = commands.add_parser(
        "predict",
        description=PREDICT_DESCRIPTION,
        formatter_class=formatter,
        help="Write per-protein ensemble probabilities.",
    )
    predict.add_argument("--model-dir", required=True, help="Directory of released checkpoints.")
    predict.add_argument("--fasta", required=True, help="Protein FASTA, plain or gzipped.")
    predict.add_argument("--output", required=True, help="Per-protein prediction TSV.")
    _add_shared_paths(predict)
    predict.add_argument(
        "--checkpoint-mode",
        choices=["es", "es_ef10", "final"],
        default="es",
        help="Checkpoint suffix to load (default: es).",
    )
    _add_runtime_flags(predict, batch_size=True)
    predict.set_defaults(run=run_predict)

    check = commands.add_parser(
        "check-checkpoint",
        description=CHECK_DESCRIPTION,
        help="Check one checkpoint's keys against the training module.",
    )
    check.add_argument("--checkpoint", required=True, help="Path to a released .pt checkpoint.")
    check.set_defaults(run=run_check)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    exec_fn: ExecFn = os.execv,
    weight_fn: WeightFn = positive_weight_text,
) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            args.run(args, exec_fn=exec_fn, weight_fn=weight_fn)
        else:
            args.run(args, exec_fn=exec_fn)
    except CliError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
