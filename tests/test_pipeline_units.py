from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from initialize_models import create_prototype
from make_splits import cross_group_split, speaker_dependent, validate
from prepare_dataset import discover, normalize_transcription, wav_duration_seconds
from evaluate import parse_mlf
from generate_htk_files import main as generate_htk_files
from select_vocabulary import is_monosyllabic, matches_group
from run_experiment_matrix import clear_model_outputs, completed_run_matches
from run_repeated_experiment_matrix import aggregate, completed_repeat_matches
from run import build_command
from run_pipeline import archive_existing_results
import common


class PipelineUnitTests(unittest.TestCase):
    def test_launcher_selects_single_or_repeated_runner(self) -> None:
        single = build_command(["mild"], ["small", "large"], 1, False)
        repeated = build_command(["medium"], ["large"], 10, True, Path("saved-results"))
        self.assertTrue(single[1].endswith("scripts/run_experiment_matrix.py"))
        self.assertNotIn("--repeats", single)
        self.assertTrue(repeated[1].endswith("scripts/run_repeated_experiment_matrix.py"))
        self.assertIn("10", repeated)
        self.assertIn("--skip-completed", repeated)
        self.assertIn("saved-results", repeated)
        cross_group = build_command(
            ["mild", "medium", "severe", "all"], ["large"], 3, False,
            Path("cross-results"), "control",
        )
        self.assertIn("--test-group", cross_group)
        self.assertIn("control", cross_group)

    def test_existing_output_paths_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "summary.csv"
            existing.write_text("old\n")
            new_output = common.unique_output_path(existing)
            self.assertNotEqual(new_output, existing)
            self.assertTrue(existing.exists())
            preserved = common.preserve_path(existing)
            self.assertIsNotNone(preserved)
            self.assertFalse(existing.exists())
            self.assertEqual(preserved.read_text(), "old\n")

    def test_result_paths_are_portable_inside_repository(self) -> None:
        path = common.ROOT / "results" / "experiment_matrix"
        self.assertEqual(common.portable_path(path), "results/experiment_matrix")

    def test_standalone_results_are_archived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results/logs").mkdir(parents=True)
            (root / "results/logs/.gitkeep").touch()
            (root / "results/logs/train.log").write_text("log\n")
            (root / "results/mixture_2/test").mkdir(parents=True)
            (root / "results/experiment_summary.json").write_text("{}\n")
            (root / "results/selected_mixture.txt").write_text("2\n")
            (root / "models/final_mix_2").mkdir(parents=True)
            (root / "models/final_mix_2/hmmdefs").write_text("model\n")
            with mock.patch.object(common, "ROOT", root):
                archive_existing_results()
            archives = list((root / "results/history").glob("run_*"))
            self.assertEqual(len(archives), 1)
            self.assertTrue((archives[0] / "experiment_summary.json").is_file())
            self.assertTrue((archives[0] / "mixture_2/test").is_dir())
            self.assertTrue((archives[0] / "selected_model/hmmdefs").is_file())
            self.assertFalse((root / "results/experiment_summary.json").exists())
            self.assertTrue((root / "results/logs/.gitkeep").exists())

    def test_normalize_transcription(self) -> None:
        self.assertEqual(normalize_transcription("  fivé!  "), "FIVE")
        self.assertEqual(normalize_transcription("Ice cream"), "ICE CREAM")

    def test_wav_duration_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audio.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\0\0" * 8000)
            self.assertAlmostEqual(wav_duration_seconds(path), 0.5)

    def test_repeated_experiment_aggregation(self) -> None:
        rows = [
            {
                "severity": "mild", "model_size": "small", "status": "complete",
                "accuracy": value, "macro_word_accuracy": value,
                "selected_mixtures": mixture, "hmm_states": 5,
                "emitting_states": 3, "maximum_mixtures": 3,
            }
            for value, mixture in ((0.2, 1), (0.4, 3))
        ]
        result = aggregate(rows)
        self.assertEqual(result[0]["successful_repeats"], 2)
        self.assertAlmostEqual(result[0]["mean_accuracy"], 0.3)
        self.assertAlmostEqual(result[0]["mean_selected_mixtures"], 2.0)

    def test_completed_run_requires_matching_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            (destination / "test").mkdir()
            (destination / "config.json").write_text('{"random_seed": 7}\n')
            (destination / "selected_mixture.txt").write_text("2\n")
            (destination / "test/experiment_summary.json").write_text("{}\n")
            self.assertTrue(completed_run_matches(destination, {"random_seed": 7}))
            self.assertFalse(completed_run_matches(destination, {"random_seed": 8}))

    def test_completed_repeat_requires_matching_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            fingerprint = {"models": ["small"], "severities": ["mild"]}
            (destination / "repeat_config.json").write_text(
                '{"models": ["small"], "severities": ["mild"]}\n'
            )
            (destination / "summary.csv").write_text("status\ncomplete\n")
            self.assertTrue(completed_repeat_matches(destination, fingerprint, 1))
            self.assertFalse(completed_repeat_matches(destination, {"models": ["large"]}, 1))

    def test_matrix_cleanup_keeps_feature_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "results/logs"
            logs.mkdir(parents=True)
            (logs / "hcopy_train.log").touch()
            (logs / "herest_01.log").touch()
            (root / "results/mixture_1").mkdir()
            with mock.patch.object(common, "ROOT", root):
                clear_model_outputs()
            self.assertTrue((logs / "hcopy_train.log").exists())
            self.assertFalse((logs / "herest_01.log").exists())
            self.assertFalse((root / "results/mixture_1").exists())

    def test_prototype_has_three_emitting_states_and_39_dimensions(self) -> None:
        prototype = create_prototype(39)
        self.assertEqual(prototype.count("<STATE>"), 3)
        self.assertIn("<NUMSTATES> 5", prototype)
        self.assertEqual(prototype.count("<MEAN> 39"), 3)
        self.assertEqual(prototype.count("<VARIANCE> 39"), 3)
        self.assertIn("0.0 0.0 0.0 0.6 0.4", prototype)

    def test_larger_prototype_has_five_emitting_states(self) -> None:
        prototype = create_prototype(39, 7)
        self.assertEqual(prototype.count("<STATE>"), 5)
        self.assertIn("<NUMSTATES> 7", prototype)
        self.assertIn("<TRANSP> 7", prototype)
        self.assertEqual(prototype.count("<MEAN> 39"), 5)
        self.assertIn("0.0 0.0 0.0 0.0 0.0 0.6 0.4", prototype)

    def test_cmudict_syllable_count(self) -> None:
        self.assertTrue(is_monosyllabic("FIVE")[0])
        self.assertTrue(is_monosyllabic("SIX")[0])
        self.assertFalse(is_monosyllabic("SEVEN")[0])

    def test_control_and_combined_group_selection(self) -> None:
        dysarthric = {
            "speaker_id": "F04", "dysarthric": "true", "isolated_word": "true",
        }
        control = {
            "speaker_id": "FC01", "dysarthric": "false", "isolated_word": "true",
        }
        self.assertTrue(matches_group(control, "control", set()))
        self.assertFalse(matches_group(dysarthric, "control", set()))
        self.assertTrue(matches_group(control, "combined", set()))
        self.assertTrue(matches_group(dysarthric, "combined", set()))
        self.assertTrue(matches_group(dysarthric, "mild", {"F04"}))

    def test_discovery_recognizes_control_speakers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "FC" / "FC01" / "Session1"
            (session / "wav_headMic").mkdir(parents=True)
            (session / "prompts").mkdir()
            (session / "wav_headMic/0001.wav").touch()
            (session / "prompts/0001.txt").write_text("five\n")
            rows = discover(root, ["headmic"])
            self.assertEqual(rows[0]["speaker_id"], "FC01")
            self.assertEqual(rows[0]["speaker_group"], "control")
            self.assertEqual(rows[0]["dysarthric"], "false")
            self.assertEqual(rows[0]["severity"], "control")

    def test_parse_plain_and_timed_mlf_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recognized.mlf"
            path.write_text(
                '#!MLF!#\n"*/F01_FIVE_0001.lab"\n0 100000 FIVE -12.3\n.\n'
                '"*/F01_SIX_0001.lab"\nSIX\n.\n'
            )
            self.assertEqual(parse_mlf(path), {"F01_FIVE_0001": "FIVE", "F01_SIX_0001": "SIX"})

    def test_discovery_pairs_prompts_and_prefers_head_mic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "F01" / "Session1"
            (session / "wav_headMic").mkdir(parents=True)
            (session / "wav_arrayMic").mkdir()
            (session / "prompts").mkdir()
            (session / "wav_headMic" / "0001.wav").touch()
            (session / "wav_arrayMic" / "0001.wav").touch()
            (session / "prompts" / "0001.txt").write_text("five\n")
            rows = discover(root, ["headmic", "arraymic"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["speaker_id"], "F01")
            self.assertEqual(rows[0]["word"], "FIVE")
            self.assertTrue(str(rows[0]["wav_path"]).endswith("wav_headMic/0001.wav"))

    def test_speaker_dependent_split_is_deterministic_and_leak_free(self) -> None:
        rows = [
            {"speaker_id": "F01", "word": "FIVE", "wav_path": f"/{index}.wav"}
            for index in range(10)
        ]
        config = {
            "random_seed": 7, "train_ratio": 0.6, "validation_ratio": 0.2,
            "test_ratio": 0.2, "minimum_examples_per_speaker_word": 3,
        }
        first = speaker_dependent(rows, config)
        second = speaker_dependent(rows, config)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["split"] == "train" for row in first), 6)
        self.assertEqual(sum(row["split"] == "validation" for row in first), 2)
        self.assertEqual(sum(row["split"] == "test" for row in first), 2)
        validate(first)

    def test_cross_group_split_uses_controls_only_for_test(self) -> None:
        rows = [
            {
                "speaker_id": "F04", "word": "FIVE", "wav_path": f"/train-{index}.wav",
                "dataset_role": "training",
            }
            for index in range(5)
        ] + [
            {
                "speaker_id": "FC01", "word": "FIVE", "wav_path": f"/test-{index}.wav",
                "dataset_role": "testing",
            }
            for index in range(2)
        ]
        config = {
            "random_seed": 7, "train_ratio": 0.6, "validation_ratio": 0.2,
            "test_ratio": 0.2, "minimum_examples_per_speaker_word": 3,
        }
        output = cross_group_split(rows, config)
        self.assertTrue(all(row["speaker_id"] == "FC01" for row in output if row["split"] == "test"))
        self.assertTrue(all(row["speaker_id"] == "F04" for row in output if row["split"] != "test"))
        self.assertEqual(sum(row["split"] == "test" for row in output), 2)
        validate(output)

    def test_generated_scp_mlf_dictionary_and_grammar_are_consistent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="htk project with spaces ") as temporary:
            root = Path(temporary).resolve()
            for directory in (
                "metadata", "lists", "labels", "dictionary", "grammar",
                "features/mfc/train", "features/mfc/validation", "features/mfc/test",
            ):
                (root / directory).mkdir(parents=True, exist_ok=True)
            rows = []
            for split, utterance in (
                ("train", "F01_FIVE_0001"),
                ("validation", "F01_FIVE_0002"),
                ("test", "F01_FIVE_0003"),
            ):
                wav = root / "wav files" / f"{utterance}.wav"
                wav.parent.mkdir(exist_ok=True)
                wav.touch()
                rows.append({
                    "speaker_id": "F01", "word": "FIVE", "split": split,
                    "utterance_id": utterance, "wav_path": str(wav),
                })
            with mock.patch.object(common, "ROOT", root):
                common.write_csv("metadata/metadata.csv", rows, rows[0].keys())
                (root / "metadata/vocabulary.txt").write_text("FIVE\n")
                with mock.patch.object(sys, "argv", ["generate_htk_files.py"]):
                    generate_htk_files()
            train_scp = (root / "lists/train_hcopy.scp").read_text()
            self.assertIn(f'"{root}/wav files/F01_FIVE_0001.wav"', train_scp)
            self.assertIn(f'"{root}/features/mfc/train/F01_FIVE_0001.mfc"', train_scp)
            self.assertIn('"*/F01_FIVE_0001.lab"\nFIVE\n.', (root / "labels/train.mlf").read_text())
            self.assertEqual((root / "dictionary/dict").read_text(), "FIVE FIVE\n")
            self.assertEqual((root / "grammar/grammar").read_text(), "$WORD = FIVE;\n\n( $WORD )\n")


if __name__ == "__main__":
    unittest.main()
