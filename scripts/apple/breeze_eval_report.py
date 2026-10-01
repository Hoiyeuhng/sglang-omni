# SPDX-License-Identifier: Apache-2.0
"""Recompute paired accuracy and performance tables from saved request records."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import typer
from pydantic import BaseModel

from scripts.apple.breeze_eval_common import EvaluationManifest
from scripts.apple.breeze_eval_http import HttpResult


class PerformanceRecord(BaseModel):
    sample_id: str
    language: str
    warmup: bool
    status: Literal["success", "error", "timeout"]
    elapsed_seconds: float
    audio_seconds: float | None = None
    first_audio_seconds: float | None = None
    rtf: float | None = None
    termination: Literal["eos", "frame_limit", "context_limit"] | None = None
    mps_allocated_bytes: int | None = None
    mps_driver_bytes: int | None = None


class AccuracyRecord(BaseModel):
    sample_id: str
    language: str
    status: Literal["success", "generation_failed", "scoring_failed"]
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    error_rate: float | None = None


class HttpGroup(BaseModel):
    concurrency: int
    repetition: int
    elapsed_seconds: float
    requests: int
    successful: int


class HttpManifest(BaseModel):
    manifest_sha256: str
    sample_ids: list[str]
    client_concurrency: list[int]
    repetitions: int


class ComparisonConfiguration(BaseModel):
    manifest_sha256: str
    checkpoint_config_sha256: str
    checkpoint_weight_sha256: dict[str, str]
    dtype: str
    codec_dtype: str
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    seed: int
    max_new_tokens: int


def summarize_http(directory: Path, output: Path, manifest_sha256: str) -> None:
    manifest = HttpManifest.model_validate_json(
        (directory / "http-metadata.json").read_text()
    )
    if manifest.manifest_sha256 != manifest_sha256:
        raise ValueError("HTTP results use a different input manifest")
    else:
        pass
    requests = [
        HttpResult.model_validate_json(line)
        for line in (directory / "http-requests.jsonl").read_text().splitlines()
    ]
    groups = [
        HttpGroup.model_validate_json(line)
        for line in (directory / "http-groups.jsonl").read_text().splitlines()
    ]
    table = []
    for concurrency in manifest.client_concurrency:
        selected = [
            row
            for row in requests
            if row.concurrency == concurrency and row.repetition >= 0
        ]
        selected_groups = [row for row in groups if row.concurrency == concurrency]
        for repetition in range(manifest.repetitions):
            trial = [row for row in selected if row.repetition == repetition]
            trial_groups = [
                row for row in selected_groups if row.repetition == repetition
            ]
            if len(trial_groups) != 1 or len(trial) != len(manifest.sample_ids):
                raise ValueError(f"Incomplete HTTP trial: c{concurrency}/{repetition}")
            elif {row.sample_id for row in trial} != set(manifest.sample_ids):
                raise ValueError(f"HTTP sample IDs differ: c{concurrency}/{repetition}")
            elif trial_groups[0].successful != sum(
                row.status == "success" for row in trial
            ):
                raise ValueError(
                    f"HTTP success count differs: c{concurrency}/{repetition}"
                )
            else:
                pass
        successful = [row for row in selected if row.status == "success"]
        row = {
            "concurrency": concurrency,
            "requests": len(selected),
            "successful": len(successful),
            "successful_requests_per_second": len(successful)
            / sum(group.elapsed_seconds for group in selected_groups),
        }
        measurements = {
            "first_audio_seconds": [
                request.first_audio_seconds
                for request in successful
                if request.first_audio_seconds is not None
            ],
            "elapsed_seconds": [request.elapsed_seconds for request in successful],
            "rtf": [
                request.elapsed_seconds / request.audio_seconds
                for request in successful
                if request.audio_seconds is not None
            ],
        }
        for name, values in measurements.items():
            if values:
                row[f"{name}_mean"] = float(np.mean(values))
                row[f"{name}_p95"] = float(np.percentile(values, 95))
            else:
                row[f"{name}_mean"] = None
                row[f"{name}_p95"] = None
        table.append(row)
    changed_audio = []
    for sample_id in manifest.sample_ids:
        digests = {
            row.audio_sha256
            for row in requests
            if row.sample_id == sample_id
            and row.repetition >= 0
            and row.status == "success"
        }
        if len(digests) > 1:
            changed_audio.append(sample_id)
        else:
            pass
    (output / "http-comparison.json").write_text(
        json.dumps(
            {
                "percentiles": "Pooled requests across measured repetitions; linear interpolation",
                "audio_hash_mismatches": changed_audio,
                "rows": table,
            },
            indent=2,
        )
        + "\n"
    )
    with (output / "http-comparison.csv").open("w") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main(
    manifest: Path,
    output: Path,
    directories: list[Path],
    http_directory: Path | None = None,
) -> None:
    inputs = EvaluationManifest.model_validate_json(manifest.read_text())
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    configurations = [
        ComparisonConfiguration.model_validate_json(
            (directory / "metadata.json").read_text()
        )
        for directory in directories
    ]
    if not configurations or any(
        configuration.manifest_sha256 != manifest_sha256
        or configuration != configurations[0]
        for configuration in configurations
    ):
        raise ValueError(
            "Comparison requires identical manifests, weights and sampling configuration"
        )
    else:
        pass
    scorer_configurations = [
        json.loads((directory / "scorer.json").read_text()) for directory in directories
    ]
    if any(
        configuration != scorer_configurations[0]
        for configuration in scorer_configurations
    ):
        raise ValueError(
            "Comparison requires identical ASR and normalization configuration"
        )
    else:
        pass
    table = []
    for directory in directories:
        performance = [
            PerformanceRecord.model_validate_json(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
        ]
        accuracy = [
            AccuracyRecord.model_validate_json(line)
            for line in (directory / "scores.jsonl").read_text().splitlines()
        ]
        for language in ("en", "zh"):
            expected = {
                sample.sample_id
                for sample in inputs.samples
                if sample.language == language
            }
            requests = [
                row
                for row in performance
                if row.language == language and not row.warmup
            ]
            scores = [row for row in accuracy if row.language == language]
            if len(requests) != len({row.sample_id for row in requests}) or len(
                scores
            ) != len({row.sample_id for row in scores}):
                raise ValueError(f"Duplicate records in {directory}/{language}")
            elif {row.sample_id for row in requests} != expected or {
                row.sample_id for row in scores
            } != expected:
                raise ValueError(
                    f"Missing or unexpected sample IDs in {directory}/{language}"
                )
            else:
                pass
            successful = [row for row in requests if row.status == "success"]
            scored = [row for row in scores if row.status == "success"]
            reference_units = sum(
                row.substitutions + row.deletions + row.hits for row in scored
            )
            errors = sum(
                row.substitutions + row.deletions + row.insertions for row in scored
            )
            elapsed_seconds = sum(row.elapsed_seconds for row in requests)
            table.append(
                {
                    "implementation": directory.name,
                    "language": language,
                    "requested": len(requests),
                    "generated": len(successful),
                    "scored": len(scored),
                    "timeouts": sum(row.status == "timeout" for row in requests),
                    "frame_limit": sum(
                        row.termination == "frame_limit" for row in requests
                    ),
                    "context_limit": sum(
                        row.termination == "context_limit" for row in requests
                    ),
                    "corpus_error_rate": (
                        errors / reference_units if reference_units else None
                    ),
                    "reference_units": reference_units,
                    "errors": errors,
                    "above_50_percent_error": sum(
                        row.error_rate > 0.5
                        for row in scored
                        if row.error_rate is not None
                    ),
                    "successful_requests_per_compute_second": len(successful)
                    / elapsed_seconds,
                }
            )
            measurements = {
                "elapsed_seconds": [row.elapsed_seconds for row in successful],
                "audio_seconds": [
                    row.audio_seconds
                    for row in successful
                    if row.audio_seconds is not None
                ],
                "first_audio_seconds": [
                    row.first_audio_seconds
                    for row in successful
                    if row.first_audio_seconds is not None
                ],
                "rtf": [row.rtf for row in successful if row.rtf is not None],
                "mps_allocated_bytes": [
                    row.mps_allocated_bytes
                    for row in successful
                    if row.mps_allocated_bytes is not None
                ],
                "mps_driver_bytes": [
                    row.mps_driver_bytes
                    for row in successful
                    if row.mps_driver_bytes is not None
                ],
            }
            for field, values in measurements.items():
                if values:
                    table[-1][f"{field}_mean"] = float(np.mean(values))
                    table[-1][f"{field}_median"] = float(np.median(values))
                    table[-1][f"{field}_p95"] = float(np.percentile(values, 95))
                else:
                    table[-1][f"{field}_mean"] = None
                    table[-1][f"{field}_median"] = None
                    table[-1][f"{field}_p95"] = None
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(table, indent=2) + "\n")
    with (output / "comparison.csv").open("w") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(table, indent=2))
    if http_directory is not None:
        summarize_http(http_directory, output, manifest_sha256)
    else:
        pass


if __name__ == "__main__":
    typer.run(main)
else:
    pass
