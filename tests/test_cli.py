"""Argv assembly for the released-model commands. These lists match the former shell wrappers."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ggn_apta.cli import (  # noqa: E402
    TRAIN_SCRIPT,
    CliError,
    build_check_argv,
    build_evaluate_argv,
    build_parser,
    build_predict_argv,
    build_train_argv,
    repository_root,
    run_evaluate,
    run_predict,
    run_train,
)

APTAMER = "/data/positives.txt"
TRAIN_FASTA = "/data/train.fasta.gz"
OUTPUT_DIR = "/data/run"
PROTTRANS = "/data/prottrans"
ESM2 = "/data/esm2"
PDB_DIR = "/data/structures"
COORDINATES = "/data/coordinates"
DSSP = "/data/dssp"
MODEL_DIR = "/data/checkpoints"
TEST_FASTA = "/data/test.fasta.gz"
METRICS = "/data/metrics.tsv"
PREDICT_FASTA = "/data/proteins.fasta.gz"
PREDICTIONS = "/data/predictions.tsv"
POS_WEIGHT = "3.5"

# Frozen from scripts/train.sh before that wrapper was removed.
EXPECTED_TRAIN = [
    "--aptamer_list_file",
    APTAMER,
    "--train_fasta_file",
    TRAIN_FASTA,
    "--output_dir",
    OUTPUT_DIR,
    "--prottrans_output_path",
    PROTTRANS,
    "--esm2_output_path",
    ESM2,
    "--pdb_dir",
    PDB_DIR,
    "--proc_pdb_output_path",
    COORDINATES,
    "--dssp_output_path",
    DSSP,
    "--learning_rate",
    "5e-5",
    "--weight_decay",
    "1e-4",
    "--batch_size",
    "6",
    "--seed_base",
    "1234",
    "--k_folds",
    "5",
    "--max_epochs",
    "100",
    "--focal_alpha",
    "3.7",
    "--focal_gamma",
    "1.5",
    "--pos_weight",
    POS_WEIGHT,
    "--dropout",
    "0.60",
    "--k_pos_override",
    "1",
    "--lr_schedule_type",
    "exponential",
    "--exp_gamma",
    "0.950",
    "--scheduler_patience",
    "5",
    "--gradient_clip_max_norm",
    "5.0",
    "--hidden_dim",
    "640",
    "--num_layers",
    "4",
    "--pooling",
    "GAP",
    "--contrast",
    "true",
    "--contrast_weight",
    "1.0",
    "--contrast_temp",
    "0.1",
    "--augment_eps",
    "0.60",
    "--use_stratified_sampling",
    "true",
    "--balanced_epoch_sampling",
    "false",
    "--stratified_without_replacement",
    "true",
    "--stop_epoch_when_exhausted",
    "true",
    "--use_mixed_precision",
    "true",
    "--early_stopping_metric",
    "aupr",
    "--early_stopping_min_delta",
    "0.001",
    "--early_stopping_patience",
    "5",
    "--swa_enabled",
    "false",
    "--eval_use_swa",
    "false",
    "--ema_enabled",
    "false",
    "--ema_decay",
    "0.999",
    "--eval_use_ema",
    "true",
    "--loss_type",
    "focal",
    "--loss_use_pos_weight_scale",
    "true",
    "--negative_rounds",
    "1",
    "--negative_rotation_seed",
    "20260718",
    "--num_workers",
    "4",
    "--device",
    "0",
    "--gpu_ids",
    "0",
    "--disable_prottrans",
    "false",
    "--disable_esm2",
    "false",
    "--disable_dssp",
    "false",
    "--disable_dihedrals",
    "false",
    "--disable_orientations",
    "false",
    "--disable_sidechains",
    "false",
    "--disable_rbf",
    "false",
    "--disable_pos_embeddings",
    "true",
    "--disable_edge_direction",
    "false",
    "--deterministic",
    "true",
    "--deterministic_warn_only",
    "true",
    "--suffix",
    "_baseline_fa3p7_fg1p5_pw0p5x_lr5e-5_wd1e-4_drop0p60_bs6_kpos1_norep_"
    "homology_id30_seed42_strat_no_repl_stop",
]

# Frozen from scripts/evaluate.sh before that wrapper was removed.
EXPECTED_EVALUATE = [
    "--model_dir",
    MODEL_DIR,
    "--test_fasta_file",
    TEST_FASTA,
    "--aptamer_list_file",
    APTAMER,
    "--output_file",
    METRICS,
    "--checkpoint_mode",
    "es",
    "--hidden_dim",
    "640",
    "--num_layers",
    "4",
    "--dropout",
    "0.60",
    "--augment_eps",
    "0.60",
    "--contrast",
    "--contrast_weight",
    "1.0",
    "--contrast_temp",
    "0.1",
    "--pooling",
    "GAP",
    "--prottrans_output_path",
    PROTTRANS,
    "--esm2_output_path",
    ESM2,
    "--pdb_dir",
    PDB_DIR,
    "--proc_pdb_output_path",
    COORDINATES,
    "--dssp_output_path",
    DSSP,
    "--disable_pos_embeddings",
    "--batch_size",
    "6",
    "--num_workers",
    "4",
    "--device",
    "0",
]

# Frozen from scripts/predict.sh before that wrapper was removed.
EXPECTED_PREDICT = [
    "--model_dir",
    MODEL_DIR,
    "--fasta_file",
    PREDICT_FASTA,
    "--aptamer_list_file",
    APTAMER,
    "--output_predictions_tsv",
    PREDICTIONS,
    "--checkpoint_mode",
    "es",
    "--hidden_dim",
    "640",
    "--num_layers",
    "4",
    "--dropout",
    "0.60",
    "--augment_eps",
    "0.60",
    "--contrast",
    "--contrast_weight",
    "1.0",
    "--contrast_temp",
    "0.1",
    "--pooling",
    "GAP",
    "--prottrans_output_path",
    PROTTRANS,
    "--esm2_output_path",
    ESM2,
    "--pdb_dir",
    PDB_DIR,
    "--proc_pdb_output_path",
    COORDINATES,
    "--dssp_output_path",
    DSSP,
    "--disable_pos_embeddings",
    "--batch_size",
    "6",
    "--num_workers",
    "4",
    "--device",
    "0",
]


def _shared(prefix: list[str]) -> list[str]:
    return prefix + [
        "--aptamer-list",
        APTAMER,
        "--prottrans",
        PROTTRANS,
        "--esm2",
        ESM2,
        "--pdb-dir",
        PDB_DIR,
        "--coordinates",
        COORDINATES,
        "--dssp",
        DSSP,
    ]


class ArgvTests(unittest.TestCase):
    def test_import_does_not_load_the_trainer(self) -> None:
        name = (
            "Scripts.Sub.train_fixed_lr_scheduler_no_ddp_fsdp_"
            "allow_sched_factor_1_augment_eps_edge_dropout"
        )
        self.assertNotIn(name, sys.modules)

    def test_train_argv_matches_shell(self) -> None:
        args = build_parser().parse_args(
            _shared(
                [
                    "train",
                    "--train-fasta",
                    TRAIN_FASTA,
                    "--output-dir",
                    OUTPUT_DIR,
                    "--pos-weight",
                    POS_WEIGHT,
                ]
            )
        )
        argv = build_train_argv(args, pos_weight=POS_WEIGHT, root=repository_root())
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(Path(argv[1]), repository_root() / TRAIN_SCRIPT)
        self.assertEqual(argv[2:], EXPECTED_TRAIN)

    def test_train_device_is_also_gpu_ids(self) -> None:
        args = build_parser().parse_args(
            _shared(
                [
                    "train",
                    "--train-fasta",
                    TRAIN_FASTA,
                    "--output-dir",
                    OUTPUT_DIR,
                    "--num-workers",
                    "8",
                    "--device",
                    "1",
                ]
            )
        )
        flags = build_train_argv(args, pos_weight=POS_WEIGHT, root=repository_root())[2:]
        self.assertEqual(flags[flags.index("--num_workers") + 1], "8")
        self.assertEqual(flags[flags.index("--device") + 1], "1")
        self.assertEqual(flags[flags.index("--gpu_ids") + 1], "1")
        self.assertEqual(flags[flags.index("--batch_size") + 1], "6")

    def test_evaluate_argv_matches_shell(self) -> None:
        args = build_parser().parse_args(
            _shared(
                [
                    "evaluate",
                    "--model-dir",
                    MODEL_DIR,
                    "--test-fasta",
                    TEST_FASTA,
                    "--output",
                    METRICS,
                ]
            )
        )
        argv = build_evaluate_argv(args, root=repository_root())
        self.assertEqual(Path(argv[1]).name, "evaluate_test_set.py")
        self.assertEqual(argv[2:], EXPECTED_EVALUATE)
        self.assertNotIn("--gpu_ids", argv)

    def test_predict_argv_matches_shell(self) -> None:
        args = build_parser().parse_args(
            _shared(
                [
                    "predict",
                    "--model-dir",
                    MODEL_DIR,
                    "--fasta",
                    PREDICT_FASTA,
                    "--output",
                    PREDICTIONS,
                ]
            )
        )
        argv = build_predict_argv(args, root=repository_root())
        self.assertEqual(Path(argv[1]).name, "export_protein_predictions.py")
        self.assertEqual(argv[2:], EXPECTED_PREDICT)
        self.assertNotIn("--gpu_ids", argv)

    def test_help_lists_pinned_settings(self) -> None:
        train_help_text = _subcommand_help("train")
        evaluate_help_text = _subcommand_help("evaluate")
        self.assertIn("focal alpha 3.7", train_help_text)
        self.assertIn("batch size 6", train_help_text)
        self.assertNotIn("--batch-size", train_help_text)
        self.assertNotIn("--focal-alpha", train_help_text)
        self.assertIn("hidden size 640", evaluate_help_text)
        self.assertIn("--batch-size", evaluate_help_text)


def _subcommand_help(name: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            build_parser().parse_args([name, "--help"])
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    return buffer.getvalue()


class RunTests(unittest.TestCase):
    def test_missing_train_directory_skips_exec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aptamer = root / "positives.txt"
            fasta = root / "train.fasta.gz"
            aptamer.write_text("P12345\n")
            fasta.write_text(">p\nM\n")
            for name in ("esm2", "structures", "coordinates", "dssp"):
                (root / name).mkdir()
            out = root / "run"
            calls: list[tuple[object, ...]] = []
            args = build_parser().parse_args(
                [
                    "train",
                    "--aptamer-list",
                    str(aptamer),
                    "--train-fasta",
                    str(fasta),
                    "--output-dir",
                    str(out),
                    "--prottrans",
                    str(root / "missing-prottrans"),
                    "--esm2",
                    str(root / "esm2"),
                    "--pdb-dir",
                    str(root / "structures"),
                    "--coordinates",
                    str(root / "coordinates"),
                    "--dssp",
                    str(root / "dssp"),
                ]
            )
            with self.assertRaises(CliError) as caught:
                run_train(
                    args,
                    exec_fn=lambda *parts: calls.append(parts),
                    weight_fn=lambda *_parts: "1",
                )
            self.assertIn("--prottrans", str(caught.exception))
            self.assertEqual(calls, [])
            self.assertFalse(out.exists())

    def test_missing_evaluate_fasta_skips_exec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models"
            model.mkdir()
            aptamer = root / "positives.txt"
            aptamer.write_text("P12345\n")
            calls: list[tuple[object, ...]] = []
            args = build_parser().parse_args(
                [
                    "evaluate",
                    "--model-dir",
                    str(model),
                    "--test-fasta",
                    str(root / "missing.fasta.gz"),
                    "--output",
                    str(root / "nested" / "metrics.tsv"),
                    "--aptamer-list",
                    str(aptamer),
                    "--prottrans",
                    str(root / "not-a-directory"),
                    "--esm2",
                    str(root / "not-a-directory"),
                    "--pdb-dir",
                    str(root / "not-a-directory"),
                    "--coordinates",
                    str(root / "not-a-directory"),
                    "--dssp",
                    str(root / "not-a-directory"),
                ]
            )
            with self.assertRaises(CliError) as caught:
                run_evaluate(args, exec_fn=lambda *parts: calls.append(parts))
            self.assertIn("--test-fasta", str(caught.exception))
            self.assertEqual(calls, [])
            self.assertFalse((root / "nested").exists())

    def test_empty_feature_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models"
            model.mkdir()
            aptamer = root / "positives.txt"
            fasta = root / "test.fasta.gz"
            aptamer.write_text("P12345\n")
            fasta.write_text(">p\nM\n")
            args = build_parser().parse_args(
                [
                    "evaluate",
                    "--model-dir",
                    str(model),
                    "--test-fasta",
                    str(fasta),
                    "--output",
                    str(root / "metrics.tsv"),
                    "--aptamer-list",
                    str(aptamer),
                    "--prottrans",
                    "",
                    "--esm2",
                    str(root / "not-a-directory"),
                    "--pdb-dir",
                    str(root / "not-a-directory"),
                    "--coordinates",
                    str(root / "not-a-directory"),
                    "--dssp",
                    str(root / "not-a-directory"),
                ]
            )
            with self.assertRaises(CliError) as caught:
                run_evaluate(args, exec_fn=lambda *_parts: None)
            self.assertIn("--prottrans", str(caught.exception))
            self.assertIn("<unset>", str(caught.exception))

    def test_non_positive_pw_mode_skips_weight_and_exec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _train_tree(Path(tmp))
            calls: list[tuple[object, ...]] = []
            weight_calls: list[tuple[object, ...]] = []
            args = build_parser().parse_args(_train_command(paths, "--pw-mode", "0"))
            with self.assertRaises(CliError) as caught:
                run_train(
                    args,
                    exec_fn=lambda *parts: calls.append(parts),
                    weight_fn=lambda *parts: weight_calls.append(parts) or "1",
                )
            self.assertIn("--pw-mode", str(caught.exception))
            self.assertEqual(calls, [])
            self.assertEqual(weight_calls, [])

    def test_given_pos_weight_skips_automatic_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _train_tree(Path(tmp))
            recorded: dict[str, list[str]] = {}

            def fake_exec(_executable: str, argv: list[str]) -> None:
                recorded["argv"] = argv

            def weight(*_parts: object) -> str:
                raise AssertionError("automatic weight should be skipped")

            args = build_parser().parse_args(
                _train_command(paths, "--pos-weight", POS_WEIGHT)
            )
            run_train(args, exec_fn=fake_exec, weight_fn=weight)
            flags = recorded["argv"]
            self.assertEqual(flags[flags.index("--pos_weight") + 1], POS_WEIGHT)
            self.assertTrue(Path(paths["output"]).is_dir())

    def test_predict_creates_output_parent_before_exec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models"
            model.mkdir()
            aptamer = root / "positives.txt"
            fasta = root / "proteins.fasta.gz"
            aptamer.write_text("P12345\n")
            fasta.write_text(">p\nM\n")
            output = root / "nested" / "predictions.tsv"
            recorded: dict[str, list[str]] = {}
            args = build_parser().parse_args(
                [
                    "predict",
                    "--model-dir",
                    str(model),
                    "--fasta",
                    str(fasta),
                    "--output",
                    str(output),
                    "--aptamer-list",
                    str(aptamer),
                    "--prottrans",
                    str(root / "absent"),
                    "--esm2",
                    str(root / "absent"),
                    "--pdb-dir",
                    str(root / "absent"),
                    "--coordinates",
                    str(root / "absent"),
                    "--dssp",
                    str(root / "absent"),
                ]
            )
            run_predict(
                args,
                exec_fn=lambda _executable, argv: recorded.__setitem__("argv", argv),
            )
            self.assertTrue(output.parent.is_dir())
            self.assertFalse(output.exists())
            self.assertIn("--output_predictions_tsv", recorded["argv"])

    def test_check_checkpoint_argv(self) -> None:
        checkpoint = "/data/model_released_fold1_es.pt"
        args = build_parser().parse_args(["check-checkpoint", "--checkpoint", checkpoint])
        argv = build_check_argv(args, root=repository_root())
        self.assertEqual(Path(argv[1]).name, "check_checkpoint_keys.py")
        self.assertEqual(argv[2:], ["--checkpoint", checkpoint])


def _train_tree(root: Path) -> dict[str, Path]:
    aptamer = root / "positives.txt"
    fasta = root / "train.fasta.gz"
    aptamer.write_text("P12345\n")
    fasta.write_text(">p\nM\n")
    names = ("prottrans", "esm2", "structures", "coordinates", "dssp")
    for name in names:
        (root / name).mkdir()
    return {
        "aptamer": aptamer,
        "fasta": fasta,
        "output": root / "run",
        "prottrans": root / "prottrans",
        "esm2": root / "esm2",
        "pdb": root / "structures",
        "coordinates": root / "coordinates",
        "dssp": root / "dssp",
    }


def _train_command(paths: dict[str, Path], *extra: str) -> list[str]:
    return [
        "train",
        "--aptamer-list",
        str(paths["aptamer"]),
        "--train-fasta",
        str(paths["fasta"]),
        "--output-dir",
        str(paths["output"]),
        "--prottrans",
        str(paths["prottrans"]),
        "--esm2",
        str(paths["esm2"]),
        "--pdb-dir",
        str(paths["pdb"]),
        "--coordinates",
        str(paths["coordinates"]),
        "--dssp",
        str(paths["dssp"]),
        *extra,
    ]


if __name__ == "__main__":
    unittest.main()
