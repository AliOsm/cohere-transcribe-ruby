#!/usr/bin/env python3
"""Benchmark an installed cohere-transcribe gem on frozen ASR manifests.

The benchmark is intentionally standalone and is not part of the gem test or CI tasks. It runs the public Ruby API from an explicit GEM_HOME, preserves compact hypothesis rows, and uses the retained Python benchmark's exact WER/CER scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPOSITORY.parent
if os.fspath(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(RESEARCH_ROOT))

from benchmark_wer import Sample, load_manifests  # noqa: E402
from native_benchmark import native_wer  # noqa: E402


RUNNER_VERSION = 3
RUNNER = Path(__file__).resolve()
WORKER = Path(__file__).with_name("installed_gem_worker.rb")
SCORERS = (RESEARCH_ROOT / "benchmark_wer.py", RESEARCH_ROOT / "native_benchmark/native_wer.py")
COMPLETE_STATUSES = frozenset({"ok", "empty"})
RUNTIME_ENVIRONMENT_NAMES = (
    "COHERE_TRANSCRIBE_THREADS",
    "COHERE_TRANSCRIBE_CACHE",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "HF_HOME",
    "HF_HUB_CACHE",
    "XDG_CACHE_HOME",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, include_sha256: bool = True) -> dict[str, int | str]:
    stat = path.stat()
    identity: dict[str, int | str] = {
        "path": os.fspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        identity["sha256"] = sha256(path)
    return identity


def directory_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": os.fspath(path),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def runtime_environment_identity() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in RUNTIME_ENVIRONMENT_NAMES}


def canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def benchmark_options(args: argparse.Namespace) -> dict:
    return {
        "language": args.language,
        "text_only": True,
        "recursive": False,
        "device": args.device,
        "dtype": args.dtype,
        "audio_backend": args.audio_backend,
        "audio_memory_gb": args.audio_memory_gb,
        "preprocess_workers": args.preprocess_workers,
        "pipeline_preparation": True,
        "vad": "none",
        "max_dur": args.max_dur,
        "batch_size": args.batch_size,
        "adaptive_batch": False,
        "pin_memory": False,
        "max_new_tokens": args.max_new_tokens,
        "max_retry_tokens": args.max_new_tokens,
        "truncation_policy": "warn",
        "stop_repetition_loops": True,
        "alignment": "none",
    }


def run_fingerprint(
    samples: Sequence[Sample], args: argparse.Namespace, options: dict
) -> tuple[str, dict]:
    payload = {
        "runner_version": RUNNER_VERSION,
        "runner": file_identity(RUNNER),
        "worker": file_identity(WORKER),
        "scorers": [file_identity(path) for path in SCORERS],
        "ruby": file_identity(args.ruby),
        "gem_home": directory_identity(args.gem_home),
        "gem_root": directory_identity(args.gem_root),
        "gem_artifact": file_identity(args.gem_artifact),
        "installed_artifact_verification": args.installed_artifact_verification,
        "native_library": file_identity(args.native_library),
        "audio_library": file_identity(args.audio_library),
        "expected_gem_version": args.gem_version,
        "references": [
            {"label": label, "hypotheses": file_identity(path)}
            for label, path in args.reference
        ],
        "runtime_environment": runtime_environment_identity(),
        "lane": args.lane,
        "chunk_size": args.chunk_size if args.lane == "public_api" else None,
        "options": options,
        "samples": [],
    }
    for sample in samples:
        audio_stat = sample.audio_path.stat()
        payload["samples"].append(
            {
                "id": sample.sample_id,
                "reference": sample.reference,
                "audio_path": os.fspath(sample.audio_path),
                "audio_size": audio_stat.st_size,
                "audio_mtime_ns": audio_stat.st_mtime_ns,
                "dataset": sample.dataset,
                "domain": sample.domain,
                "dialect": sample.dialect,
                "speaker": sample.speaker,
                "duration": sample.duration,
            }
        )
    encoded = canonical_json(payload)
    return hashlib.sha256(encoded).hexdigest(), payload


def worker_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("BUNDLE_") or name in {"RUBYLIB", "RUBYOPT"}:
            environment.pop(name, None)
    environment.update(
        {
            "GEM_HOME": os.fspath(args.gem_home),
            "GEM_PATH": os.fspath(args.gem_home),
            "COHERE_TRANSCRIBE_NATIVE_LIBRARY": os.fspath(args.native_library),
            "COHERE_TRANSCRIBE_AUDIO_LIBRARY": os.fspath(args.audio_library),
        }
    )
    return environment


def sample_payload(sample: Sample) -> dict:
    return {
        "id": sample.sample_id,
        "audio_path": os.fspath(sample.audio_path),
        "reference": sample.reference,
        "dataset": sample.dataset,
        "domain": sample.domain,
        "dialect": sample.dialect,
        "speaker": sample.speaker,
        "duration": sample.duration,
    }


def load_worker_rows(
    path: Path,
    samples: Sequence[Sample],
    fingerprint: str,
    *,
    allow_trailing_partial: bool = False,
    diagnostics: list[dict] | None = None,
) -> dict[str, dict]:
    samples_by_id = {sample.sample_id: sample for sample in samples}
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    lines = path.read_bytes().splitlines(keepends=True)
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            if (
                allow_trailing_partial
                and line_number == len(lines)
                and not raw_line.endswith(b"\n")
            ):
                if diagnostics is not None:
                    diagnostics.append(
                        {
                            "kind": "ignored_trailing_partial_worker_row",
                            "line_number": line_number,
                            "bytes": len(raw_line),
                        }
                    )
                continue
            raise SystemExit(
                f"Worker row is not UTF-8 at {path}:{line_number}: {exc}"
            ) from exc
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if (
                allow_trailing_partial
                and line_number == len(lines)
                and not raw_line.endswith(b"\n")
            ):
                if diagnostics is not None:
                    diagnostics.append(
                        {
                            "kind": "ignored_trailing_partial_worker_row",
                            "line_number": line_number,
                            "bytes": len(raw_line),
                        }
                    )
                continue
            raise SystemExit(f"Invalid worker row {path}:{line_number}: {exc}") from exc
        sample_id = payload.get("id")
        if sample_id not in samples_by_id:
            raise SystemExit(f"Worker row has unknown sample ID {sample_id!r}")
        if payload.get("fingerprint") != fingerprint:
            raise SystemExit(f"Worker fingerprint mismatch at {path}:{line_number}")
        if sample_id in rows:
            raise SystemExit(f"Duplicate worker sample ID {sample_id!r}")
        hypothesis = payload.get("hypothesis")
        status = payload.get("recognition_status")
        if not isinstance(hypothesis, str) or not isinstance(status, str):
            raise SystemExit(f"Invalid worker result at {path}:{line_number}")
        rows[sample_id] = payload
    return rows


def canonical_rows(
    samples: Sequence[Sample], worker_rows: dict[str, dict], run_name: str, fingerprint: str
) -> dict[str, dict]:
    rows = {}
    for sample in samples:
        worker = worker_rows.get(sample.sample_id)
        if worker is None:
            status = "worker_output_missing"
            hypothesis = ""
            worker = {}
        else:
            status = worker["recognition_status"]
            hypothesis = worker["hypothesis"]
        row = native_wer.hypothesis_row(
            sample,
            hypothesis,
            status,
            run_name,
            fingerprint,
            sample.audio_path,
            None,
            None,
        )
        row.update(
            {
                "gem_result_path": worker.get("result_path"),
                "gem_reported_duration": worker.get("duration"),
                "gem_error": worker.get("error"),
                "gem_provenance": worker.get("provenance"),
                "worker_chunk_index": worker.get("chunk_index"),
            }
        )
        rows[sample.sample_id] = row
    return rows


def markdown_report(summary: dict) -> str:
    metrics = summary["metrics"]
    timing = summary["timing"]
    failures = summary["failures"]
    peak_rss = timing["peak_rss_kib"]
    peak_rss_text = "n/a" if peak_rss is None else f"{peak_rss} KiB"
    lines = [
        f"# {summary['config']['run_name']}",
        "",
        "## Installed-gem run",
        "",
        f"- Samples: {summary['sample_count']}",
        f"- Manifest audio: {summary['audio_seconds']:.3f} seconds",
        f"- External wall: {optional_decimal(timing['external_wall_seconds'])} seconds",
        f"- External RTFx: {optional_decimal(timing['external_rtfx'])}",
        f"- Gem worker wall: {optional_decimal(timing['worker_wall_seconds'])} seconds",
        f"- Peak RSS: {peak_rss_text}",
        f"- Failed samples: {failures['count']}",
        f"- Scope: {lane_scope(summary['config']['lane'])}",
        "",
        "## Accuracy",
        "",
        "| Scope | Group | N | Audio s | Raw WER | Repo-exact WER | Intended WER | Lexical WER | Lexical CER |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def add_group(scope: str, name: str, group: dict) -> None:
        lines.append(
            "| "
            + " | ".join(
                [
                    scope,
                    name.replace("|", "\\|"),
                    str(group["samples"]),
                    f"{group['audio_seconds']:.2f}",
                    native_wer.percent(native_wer.metric_rate(group, "raw")),
                    native_wer.percent(
                        native_wer.metric_rate(group, "leaderboard_repo_exact")
                    ),
                    native_wer.percent(
                        native_wer.metric_rate(group, "leaderboard_intended")
                    ),
                    native_wer.percent(
                        native_wer.metric_rate(group, "lexical_normalized")
                    ),
                    native_wer.percent(
                        native_wer.metric_rate(group, "lexical_normalized", "cer")
                    ),
                ]
            )
            + " |"
        )

    add_group("overall", "all", metrics["overall"])
    for scope, key in (
        ("dataset", "by_dataset"),
        ("domain", "by_domain"),
        ("dialect", "by_dialect"),
    ):
        for name, group in metrics[key].items():
            add_group(scope, name, group)

    if failures["samples"]:
        lines.extend(
            [
                "",
                "## Failed files",
                "",
                "| ID | Status | Error |",
                "| --- | --- | --- |",
            ]
        )
        for failure in failures["samples"]:
            lines.append(
                "| "
                + " | ".join(
                    markdown_cell(failure.get(key))
                    for key in ("id", "status", "error")
                )
                + " |"
            )

    comparisons = summary["reference_comparisons"]
    if comparisons:
        lines.extend(
            [
                "",
                "## Python reference comparisons",
                "",
                "| Reference | Raw changes | Whitespace changes | Intended changes | Lexical changes | Lexical WER delta (pp) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in comparisons:
            changes = comparison["hypothesis_changes"]
            delta = comparison["wer_comparison"]["overall"]["lexical_normalized"][
                "delta_percentage_points_native_minus_reference"
            ]
            lines.append(
                "| "
                + " | ".join(
                    [
                        comparison["label"],
                        str(changes["raw_exact"]["changed_samples"]),
                        str(changes["whitespace_normalized"]["changed_samples"]),
                        str(
                            changes["leaderboard_intended_normalized"][
                                "changed_samples"
                            ]
                        ),
                        str(changes["lexical_normalized"]["changed_samples"]),
                        native_wer.percentage_points(delta),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Hypotheses: `{summary['hypotheses_path']}`",
            f"- Worker summary: `{summary['worker']['summary_path']}`",
            f"- Worker stdout: `{summary['logs']['stdout']}`",
            f"- Worker stderr: `{summary['logs']['stderr']}`",
            "",
        ]
    )
    return "\n".join(lines)


def optional_decimal(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def lane_scope(lane: str) -> str:
    if lane == "public_api":
        return "one installed Ruby gem process using the public API, including model load, native decoding, fixed whole-file planning, Dense inference, and result materialization. Public API files are independent inputs; batch size applies to segments within a file and does not form cross-file batches."
    if lane == "native_batch":
        return "one installed Ruby gem process using installed-gem runtime components directly. Each file up to 35 seconds occupies one native processor row in a length-ordered cross-file batch of at most 24 rows; longer files use the native single-row processor-compatible chunk planner. Python length-B24 batches 24 utterances before preprocessing and can expand them to more than 24 feature rows, so this lane exercises optimized native B24 execution but is not an identical batching schedule or public API file scheduling semantics."
    raise ValueError(f"Unknown benchmark lane: {lane!r}")


def resolve_file(path: Path, label: str, *, executable: bool = False) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise SystemExit(f"{label} is not a file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise SystemExit(f"{label} is not executable: {resolved}")
    return resolved


def resolve_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise SystemExit(f"{label} is not a directory: {resolved}")
    return resolved


def discover_gem_root(args: argparse.Namespace) -> Path:
    environment = worker_environment(args)
    probe = subprocess.run(
        [
            os.fspath(args.ruby),
            "-e",
            (
                'require "cohere/transcribe"; '
                'spec = Gem.loaded_specs.fetch("cohere-transcribe"); '
                'puts spec.version; puts spec.full_gem_path'
            ),
        ],
        cwd=tempfile.gettempdir(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "Cannot load cohere-transcribe from the requested GEM_HOME:\n"
            + probe.stderr.strip()
        )
    lines = probe.stdout.splitlines()
    if len(lines) != 2:
        raise SystemExit(f"Unexpected installed-gem probe output: {probe.stdout!r}")
    version, root = lines
    if version != args.gem_version:
        raise SystemExit(
            f"Installed gem version is {version}, expected {args.gem_version}"
        )
    return resolve_directory(Path(root), "installed gem root")


def verify_cached_gem_artifact(args: argparse.Namespace) -> dict:
    cached_path = resolve_file(
        args.gem_home / "cache" / args.gem_artifact.name,
        "GEM_HOME cached gem artifact",
    )
    requested = file_identity(args.gem_artifact)
    cached = file_identity(cached_path)
    if requested["sha256"] != cached["sha256"]:
        raise SystemExit(
            "The requested gem artifact does not match the artifact cached by "
            f"the installed GEM_HOME: {args.gem_artifact} != {cached_path}"
        )
    return {
        "verified": True,
        "method": "sha256_equal_to_gem_home_cache",
        "requested": requested,
        "cached": cached,
        "installed_root_content": verify_installed_gem_root(
            args.gem_artifact, args.gem_root
        ),
    }


def verify_installed_gem_root(gem_artifact: Path, gem_root: Path) -> dict:
    aggregate = hashlib.sha256()
    archive_paths: set[str] = set()
    regular_files = 0
    symlinks = 0
    with tarfile.open(gem_artifact, mode="r:*") as outer:
        data_member = outer.getmember("data.tar.gz")
        data_handle = outer.extractfile(data_member)
        if data_handle is None:
            raise SystemExit(f"Gem artifact has no readable data.tar.gz: {gem_artifact}")
        with tarfile.open(fileobj=io.BytesIO(data_handle.read()), mode="r:gz") as payload:
            for member in sorted(payload.getmembers(), key=lambda item: item.name):
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise SystemExit(
                        f"Gem artifact contains an invalid member path: {member.name!r}"
                    )
                relative_text = relative.as_posix()
                installed = gem_root.joinpath(*relative.parts)
                if member.isfile():
                    archive_handle = payload.extractfile(member)
                    if archive_handle is None:
                        raise SystemExit(
                            f"Gem artifact member is unreadable: {member.name!r}"
                        )
                    archive_digest = hashlib.sha256(archive_handle.read()).hexdigest()
                    if not installed.is_file():
                        raise SystemExit(
                            f"Installed gem root is missing artifact file: {installed}"
                        )
                    installed_digest = sha256(installed)
                    if installed_digest != archive_digest:
                        raise SystemExit(
                            f"Installed gem file differs from the requested artifact: {installed}"
                        )
                    aggregate.update(relative_text.encode("utf-8"))
                    aggregate.update(b"\0")
                    aggregate.update(bytes.fromhex(archive_digest))
                    archive_paths.add(relative_text)
                    regular_files += 1
                elif member.issym():
                    if not installed.is_symlink() or os.readlink(installed) != member.linkname:
                        raise SystemExit(
                            f"Installed gem symlink differs from the requested artifact: {installed}"
                        )
                    aggregate.update(relative_text.encode("utf-8"))
                    aggregate.update(b"\0symlink\0")
                    aggregate.update(member.linkname.encode("utf-8"))
                    archive_paths.add(relative_text)
                    symlinks += 1

    installed_paths = {
        path.relative_to(gem_root).as_posix()
        for path in gem_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    extras = sorted(installed_paths - archive_paths)
    return {
        "verified": True,
        "method": "all_archive_files_sha256_equal_in_installed_root",
        "regular_files": regular_files,
        "symlinks": symlinks,
        "aggregate_sha256": aggregate.hexdigest(),
        "installed_extra_file_count": len(extras),
        "installed_extra_files": extras,
    }


def parse_references(values: Sequence[str]) -> list[tuple[str, Path]]:
    references = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --reference {value!r}; expected LABEL=PATH")
        label, raw_path = value.split("=", maxsplit=1)
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label)
            or label in labels
        ):
            raise SystemExit(f"Invalid or duplicate reference label {label!r}")
        references.append((label, resolve_file(Path(raw_path), "reference hypotheses")))
        labels.add(label)
    return references


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Installed cohere-transcribe gem WER/CER benchmark"
    )
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ruby", type=Path, default=Path(shutil.which("ruby") or "ruby"))
    parser.add_argument("--gem-home", type=Path, required=True)
    parser.add_argument("--gem-artifact", type=Path, required=True)
    parser.add_argument("--gem-version", default="0.1.2")
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--audio-library", type=Path, required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument(
        "--lane",
        default="public_api",
        choices=["public_api", "native_batch"],
        help="Execution semantics: public API files or explicit installed-gem native cross-file batches.",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--language", default="ar", choices=["ar", "en"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--audio-backend", default="auto")
    parser.add_argument("--audio-memory-gb", type=float, default=4.0)
    parser.add_argument("--preprocess-workers", type=int, default=2)
    parser.add_argument("--max-dur", type=float, default=180.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=445)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("--run-name must be one filename-safe component")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    for name in ("preprocess_workers", "batch_size", "max_new_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.lane == "public_api" and args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.lane == "public_api" and args.max_dur <= 35:
        parser.error("--max-dur must exceed 35 seconds for reference chunk planning")
    if args.lane == "native_batch" and args.max_dur <= 0:
        parser.error("--max-dur must be positive")
    if args.timeout <= 0 or args.audio_memory_gb <= 0:
        parser.error("--timeout and --audio-memory-gb must be positive")

    args.ruby = resolve_file(args.ruby, "Ruby", executable=True)
    args.gem_home = resolve_directory(args.gem_home, "GEM_HOME")
    args.gem_artifact = resolve_file(args.gem_artifact, "gem artifact")
    args.native_library = resolve_file(args.native_library, "native library")
    args.audio_library = resolve_file(args.audio_library, "audio library")
    args.reference = parse_references(args.reference)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.gem_root = discover_gem_root(args)
    args.installed_artifact_verification = verify_cached_gem_artifact(args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples = load_manifests(args.manifest, args.max_samples)
    canonical_paths = [sample.audio_path.resolve(strict=True) for sample in samples]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise SystemExit("Selected manifest rows contain duplicate canonical audio paths")
    if (
        args.lane == "public_api"
        and max((sample.duration or 0.0) for sample in samples) >= args.max_dur
    ):
        raise SystemExit("--max-dur must exceed every selected manifest duration")

    options = benchmark_options(args)
    fingerprint, fingerprint_payload = run_fingerprint(samples, args, options)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    hypotheses_path = output_dir / f"{args.run_name}.hypotheses.jsonl"
    summary_path = output_dir / f"{args.run_name}.summary.json"
    markdown_path = output_dir / f"{args.run_name}.summary.md"
    config_path = output_dir / f"{args.run_name}.worker-config.json"
    worker_rows_path = output_dir / f"{args.run_name}.worker-rows.jsonl"
    worker_summary_path = output_dir / f"{args.run_name}.worker-summary.json"
    stdout_path = output_dir / f"{args.run_name}.stdout.log"
    stderr_path = output_dir / f"{args.run_name}.stderr.log"
    existing = [
        path
        for path in (
            hypotheses_path,
            summary_path,
            markdown_path,
            config_path,
            worker_rows_path,
            worker_summary_path,
            stdout_path,
            stderr_path,
        )
        if path.exists()
    ]
    if existing:
        raise SystemExit(f"Output already exists: {existing[0]}")

    worker_config = {
        "runner_version": RUNNER_VERSION,
        "fingerprint": fingerprint,
        "expected_gem_root": os.fspath(args.gem_root),
        "expected_gem_version": args.gem_version,
        "lane": args.lane,
        "worker_rows_path": os.fspath(worker_rows_path),
        "worker_summary_path": os.fspath(worker_summary_path),
        "options": options,
        "samples": [sample_payload(sample) for sample in samples],
    }
    if args.lane == "public_api":
        worker_config["chunk_size"] = args.chunk_size
    command = [os.fspath(args.ruby), os.fspath(WORKER), os.fspath(config_path)]
    if args.dry_run:
        print(json.dumps(worker_config, ensure_ascii=False, indent=2))
        print("Command:", subprocess.list2cmdline(command))
        print("Fingerprint:", fingerprint)
        return 0

    native_wer.atomic_json(config_path, worker_config)
    environment = worker_environment(args)
    started_utc = utc_now()
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=output_dir,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
    external_wall = time.perf_counter() - started
    finished_utc = utc_now()

    worker_row_diagnostics: list[dict] = []
    worker_rows = load_worker_rows(
        worker_rows_path,
        samples,
        fingerprint,
        allow_trailing_partial=timed_out or returncode != 0,
        diagnostics=worker_row_diagnostics,
    )
    rows = canonical_rows(samples, worker_rows, args.run_name, fingerprint)
    native_wer.atomic_jsonl(hypotheses_path, samples, rows)
    hypotheses = {sample.sample_id: rows[sample.sample_id]["hypothesis"] for sample in samples}
    metrics = native_wer.aggregate_metrics(samples, hypotheses)
    comparisons = [
        native_wer.compare_reference(label, path, samples, hypotheses, metrics)
        for label, path in args.reference
    ]
    status_counts: dict[str, int] = {}
    failures = []
    for sample in samples:
        status = rows[sample.sample_id]["recognition_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in COMPLETE_STATUSES:
            failures.append(
                {
                    "id": sample.sample_id,
                    "status": status,
                    "error": rows[sample.sample_id].get("gem_error"),
                }
            )

    worker_summary = (
        json.loads(worker_summary_path.read_text(encoding="utf-8"))
        if worker_summary_path.exists()
        else {}
    )
    audio_seconds = sum(sample.duration or 0.0 for sample in samples)
    summary = {
        "environment": {
            "created_utc": finished_utc,
            "started_utc": started_utc,
            "cwd": os.getcwd(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "runtime_environment": runtime_environment_identity(),
            "command": [os.fspath(Path(sys.argv[0]).resolve()), *(argv or sys.argv[1:])],
        },
        "config": {
            "run_name": args.run_name,
            "runner_version": RUNNER_VERSION,
            "runner": file_identity(RUNNER),
            "worker": file_identity(WORKER),
            "scorers": [file_identity(path) for path in SCORERS],
            "ruby": file_identity(args.ruby),
            "gem_home": directory_identity(args.gem_home),
            "gem_root": directory_identity(args.gem_root),
            "gem_artifact": file_identity(args.gem_artifact),
            "installed_artifact_verification": args.installed_artifact_verification,
            "native_library": file_identity(args.native_library),
            "audio_library": file_identity(args.audio_library),
            "lane": args.lane,
            "references": [
                {"label": label, "hypotheses": file_identity(path)}
                for label, path in args.reference
            ],
            "options": options,
            "chunk_size": args.chunk_size if args.lane == "public_api" else None,
        },
        "manifests": [os.fspath(path.expanduser().resolve()) for path in args.manifest],
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "sample_count": len(samples),
        "audio_seconds": audio_seconds,
        "hypotheses_path": os.fspath(hypotheses_path),
        "worker": {
            "config_path": os.fspath(config_path),
            "rows_path": os.fspath(worker_rows_path),
            "summary_path": os.fspath(worker_summary_path),
            "summary": worker_summary,
            "row_diagnostics": worker_row_diagnostics,
        },
        "logs": {
            "stdout": os.fspath(stdout_path),
            "stderr": os.fspath(stderr_path),
        },
        "timing": {
            "external_wall_seconds": external_wall,
            "external_rtfx": audio_seconds / external_wall if external_wall else None,
            "worker_wall_seconds": worker_summary.get("process", {}).get("wall_seconds"),
            "peak_rss_kib": worker_summary.get("process", {}).get("peak_rss_kib"),
            "returncode": returncode,
            "timed_out": timed_out,
        },
        "attempts": [
            {
                "started_utc": started_utc,
                "finished_utc": finished_utc,
                "wall_seconds": external_wall,
                "returncode": returncode,
                "timed_out": timed_out,
                "command": command,
            }
        ],
        "failures": {
            "count": len(failures),
            "by_status": dict(sorted(status_counts.items())),
            "samples": failures,
        },
        "metrics": metrics,
        "reference_comparisons": comparisons,
    }
    native_wer.atomic_json(summary_path, summary)
    markdown_path.write_text(markdown_report(summary), encoding="utf-8")

    lexical = native_wer.metric_rate(metrics["overall"], "lexical_normalized")
    print(
        f"{len(samples)} samples | lexical WER {native_wer.percent(lexical)} | "
        f"failures {len(failures)} | external RTFx {audio_seconds / external_wall:.6f}"
    )
    print(f"Summary: {summary_path}")
    return 1 if returncode != 0 or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
