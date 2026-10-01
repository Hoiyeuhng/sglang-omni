# SPDX-License-Identifier: Apache-2.0
"""Recompute paired accuracy and performance tables from saved request records."""

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np
import typer
from pydantic import BaseModel

from scripts.apple.breeze_eval_common import EvaluationManifest


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


def main(manifest: Path, output: Path, directories: list[Path]) -> None:
    inputs = EvaluationManifest.model_validate_json(manifest.read_text())
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


if __name__ == "__main__":
    typer.run(main)
else:
    pass
