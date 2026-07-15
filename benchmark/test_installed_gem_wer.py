from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("installed_gem_wer.py")
SPEC = importlib.util.spec_from_file_location("installed_gem_wer", MODULE_PATH)
assert SPEC and SPEC.loader
installed_gem_wer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installed_gem_wer)


class InstalledGemWERRunnerTests(unittest.TestCase):
    def test_lane_scope_distinguishes_public_and_cross_file_batch_semantics(self) -> None:
        public = installed_gem_wer.lane_scope("public_api")
        native = installed_gem_wer.lane_scope("native_batch")

        self.assertIn("does not form cross-file batches", public)
        self.assertIn("native processor row", native)
        self.assertIn("not an identical batching schedule", native)

    def test_worker_environment_isolated_from_bundler(self) -> None:
        args = Namespace(
            gem_home=Path("/gem/home"),
            native_library=Path("/native/libcrispasr.so"),
            audio_library=Path("/native/libcohere_audio.so"),
        )
        original = installed_gem_wer.os.environ.copy()
        try:
            installed_gem_wer.os.environ.update(
                {
                    "BUNDLE_GEMFILE": "/source/Gemfile",
                    "RUBYLIB": "/source/lib",
                    "RUBYOPT": "-rbundler/setup",
                }
            )
            environment = installed_gem_wer.worker_environment(args)
        finally:
            installed_gem_wer.os.environ.clear()
            installed_gem_wer.os.environ.update(original)

        self.assertNotIn("BUNDLE_GEMFILE", environment)
        self.assertNotIn("RUBYLIB", environment)
        self.assertNotIn("RUBYOPT", environment)
        self.assertEqual(environment["GEM_HOME"], "/gem/home")
        self.assertEqual(
            environment["COHERE_TRANSCRIBE_NATIVE_LIBRARY"],
            "/native/libcrispasr.so",
        )

    def test_load_worker_rows_rejects_wrong_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            audio.write_bytes(b"fixture")
            sample = installed_gem_wer.Sample(
                index=0,
                sample_id="sample",
                audio_path=audio,
                reference="نص",
                dataset="fixture",
                domain="msa",
                dialect="MSA",
                speaker=None,
                duration=1.0,
            )
            rows = root / "rows.jsonl"
            rows.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "fingerprint": "wrong",
                        "recognition_status": "ok",
                        "hypothesis": "نص",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                installed_gem_wer.load_worker_rows(rows, [sample], "expected")

    def test_failed_worker_can_ignore_one_trailing_partial_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            audio.write_bytes(b"fixture")
            sample = installed_gem_wer.Sample(
                index=0,
                sample_id="sample",
                audio_path=audio,
                reference="نص",
                dataset="fixture",
                domain="msa",
                dialect="MSA",
                speaker=None,
                duration=1.0,
            )
            rows = root / "rows.jsonl"
            rows.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "fingerprint": "expected",
                        "recognition_status": "ok",
                        "hypothesis": "نص",
                    },
                    ensure_ascii=False,
                )
                + "\n"
                + '{"id":',
                encoding="utf-8",
            )
            diagnostics: list[dict] = []

            loaded = installed_gem_wer.load_worker_rows(
                rows,
                [sample],
                "expected",
                allow_trailing_partial=True,
                diagnostics=diagnostics,
            )

            self.assertEqual(["sample"], list(loaded))
            self.assertEqual("ignored_trailing_partial_worker_row", diagnostics[0]["kind"])

    def test_failed_worker_can_ignore_trailing_partial_utf8_codepoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            audio.write_bytes(b"fixture")
            sample = installed_gem_wer.Sample(
                index=0,
                sample_id="sample",
                audio_path=audio,
                reference="نص",
                dataset="fixture",
                domain="msa",
                dialect="MSA",
                speaker=None,
                duration=1.0,
            )
            rows = root / "rows.jsonl"
            valid = json.dumps(
                {
                    "id": "sample",
                    "fingerprint": "expected",
                    "recognition_status": "ok",
                    "hypothesis": "نص",
                },
                ensure_ascii=False,
            ).encode("utf-8")
            rows.write_bytes(valid + b'\n{"hypothesis":"\xd8')
            diagnostics: list[dict] = []

            loaded = installed_gem_wer.load_worker_rows(
                rows,
                [sample],
                "expected",
                allow_trailing_partial=True,
                diagnostics=diagnostics,
            )

            self.assertEqual(["sample"], list(loaded))
            self.assertEqual("ignored_trailing_partial_worker_row", diagnostics[0]["kind"])

    def test_canonical_rows_score_missing_outputs_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "sample.wav"
            audio.write_bytes(b"fixture")
            sample = installed_gem_wer.Sample(
                index=0,
                sample_id="sample",
                audio_path=audio,
                reference="مرحباً",
                dataset="fixture",
                domain="msa",
                dialect=None,
                speaker=None,
                duration=1.0,
            )
            rows = installed_gem_wer.canonical_rows(
                [sample], {}, "fixture_run", "f" * 64
            )
            row = rows["sample"]
            self.assertEqual(row["hypothesis"], "")
            self.assertEqual(row["recognition_status"], "worker_output_missing")
            metrics = installed_gem_wer.native_wer.aggregate_metrics(
                [sample], {"sample": row["hypothesis"]}
            )
            self.assertGreater(
                metrics["overall"]["lexical_normalized"]["wer"]["rate"], 0
            )


if __name__ == "__main__":
    unittest.main()
