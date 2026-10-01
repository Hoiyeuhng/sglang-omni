# SPDX-License-Identifier: Apache-2.0
"""Prepare fixed length probes and summarize paired generation and playback."""

import asyncio
import hashlib
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Literal

import httpx
import typer
from jiwer import process_words
from pydantic import BaseModel, Field

from scripts.apple.breeze_eval_common import AudioChunk, EvaluationManifest
from scripts.apple.breeze_eval_http import request_audio

app = typer.Typer()
BUCKETS = ("15s", "30s", "45-55s", "over60s")


class LengthPassages(BaseModel):
    en: list[list[str]]
    zh: list[list[str]]


class LengthRequest(BaseModel):
    sample_id: str
    language: str
    warmup: bool
    status: Literal["success", "error", "timeout"]
    termination: str = ""
    elapsed_seconds: float = 0
    audio_seconds: float = 0
    first_audio_seconds: float = 0
    rtf: float = 0
    chunks: list[AudioChunk] = Field(default_factory=list)


class LengthScore(BaseModel):
    sample_id: str
    status: str
    reference_normalized: str = ""
    hypothesis_normalized: str = ""
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0


@app.command()
def prepare(reference_manifest: Path, passages: Path, output: Path) -> None:
    references = EvaluationManifest.model_validate_json(reference_manifest.read_text())
    paragraphs = LengthPassages.model_validate_json(passages.read_text())
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for language, stories in (("en", paragraphs.en), ("zh", paragraphs.zh)):
        selected = [
            sample for sample in references.samples if sample.language == language
        ][:3]
        if (
            len(selected) != 3
            or len(stories) != 3
            or any(len(story) != 4 for story in stories)
        ):
            raise ValueError(
                "Require three references and three four-part stories per language"
            )
        else:
            pass
        for level, bucket in enumerate(BUCKETS):
            for index, (reference, story) in enumerate(
                zip(selected, stories, strict=True)
            ):
                sample = reference.model_copy()
                sample.sample_id = f"{language}-{bucket}-{index + 1}"
                sample.text = (" " if language == "en" else "").join(story[: level + 1])
                source = reference_manifest.parent / sample.ref_audio
                if (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    != sample.reference_sha256
                ):
                    raise ValueError(f"Reference changed: {source}")
                else:
                    pass
                destination = output / sample.ref_audio
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                samples.append(sample)
    manifest = EvaluationManifest(
        dataset="Handwritten length passages with fixed SeedTTS speaker references",
        revision=hashlib.sha256(passages.read_bytes()).hexdigest(),
        selection="Three passages per language, cumulative paragraphs targeting 15s, 30s, 45-55s and over60s; retain all actual durations. First three references per language from the supplied frozen manifest.",
        samples=samples,
    )
    (output / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")


@app.command()
def report(
    results: list[Path],
    output: Path = typer.Option(...),
    initial_buffer_seconds: float = 1,
) -> None:
    if initial_buffer_seconds < 0:
        raise ValueError("Initial buffer cannot be negative")
    else:
        pass
    rows = []
    details = []
    for directory in results:
        requests = [
            LengthRequest.model_validate_json(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
        ]
        scores = {
            score.sample_id: score
            for line in (directory / "scores.jsonl").read_text().splitlines()
            for score in [LengthScore.model_validate_json(line)]
        }
        for language in ("en", "zh"):
            for bucket in BUCKETS:
                selected = [
                    request
                    for request in requests
                    if not request.warmup
                    and request.sample_id.startswith(f"{language}-{bucket}-")
                ]
                if len(selected) != 3:
                    raise ValueError(
                        f"Expected three requests: {directory}/{language}/{bucket}"
                    )
                else:
                    pass
                errors = 0
                reference_units = 0
                stalls = []
                gaps = []
                required_buffers = []
                successful = [
                    request for request in selected if request.status == "success"
                ]
                scored = 0
                for request in selected:
                    score = scores[request.sample_id]
                    tail_deleted = 0
                    tail_units = 0
                    if score.status == "success":
                        scored += 1
                        errors += (
                            score.substitutions + score.deletions + score.insertions
                        )
                        reference_units += (
                            score.hits + score.substitutions + score.deletions
                        )
                        alignment = process_words(
                            score.reference_normalized, score.hypothesis_normalized
                        )
                        unit_count = len(alignment.references[0])
                        tail_start = int(unit_count * 0.8)
                        tail_units = unit_count - tail_start
                        tail_deleted = sum(
                            max(
                                0,
                                chunk.ref_end_idx
                                - max(tail_start, chunk.ref_start_idx),
                            )
                            for chunk in alignment.alignments[0]
                            if chunk.type == "delete"
                        )
                    else:
                        pass
                    total_stall_seconds = 0.0
                    minimum_buffer_seconds = 0.0
                    max_gap_seconds = 0.0
                    if request.chunks:
                        first_seconds = request.chunks[0].arrival_seconds
                        playback_end_seconds = first_seconds + initial_buffer_seconds
                        available_seconds = 0.0
                        previous_seconds = first_seconds
                        for chunk in request.chunks:
                            total_stall_seconds += max(
                                0, chunk.arrival_seconds - playback_end_seconds
                            )
                            playback_end_seconds = (
                                max(playback_end_seconds, chunk.arrival_seconds)
                                + chunk.duration_seconds
                            )
                            minimum_buffer_seconds = max(
                                minimum_buffer_seconds,
                                chunk.arrival_seconds
                                - first_seconds
                                - available_seconds,
                            )
                            max_gap_seconds = max(
                                max_gap_seconds,
                                chunk.arrival_seconds - previous_seconds,
                            )
                            previous_seconds = chunk.arrival_seconds
                            available_seconds += chunk.duration_seconds
                        stalls.append(total_stall_seconds)
                        gaps.append(max_gap_seconds)
                        required_buffers.append(minimum_buffer_seconds)
                    else:
                        pass
                    details.append(
                        {
                            "runtime": directory.name,
                            "sample_id": request.sample_id,
                            "status": request.status,
                            "termination": request.termination,
                            "audio_seconds": request.audio_seconds,
                            "tail_deleted_units": tail_deleted,
                            "tail_units": tail_units,
                            "score_status": score.status,
                            "streaming": bool(request.chunks),
                            "stall_seconds": (
                                total_stall_seconds if request.chunks else None
                            ),
                            "minimum_buffer_seconds": (
                                minimum_buffer_seconds if request.chunks else None
                            ),
                            "max_chunk_gap_seconds": (
                                max_gap_seconds if request.chunks else None
                            ),
                        }
                    )
                rows.append(
                    {
                        "runtime": directory.name,
                        "language": language,
                        "target_bucket": bucket,
                        "requests": len(selected),
                        "successes": len(successful),
                        "scored": scored,
                        "termination": dict(
                            Counter(request.termination for request in selected)
                        ),
                        "errors": errors,
                        "reference_units": reference_units,
                        "error_rate": (
                            errors / reference_units if reference_units else None
                        ),
                        "audio_seconds_min": min(
                            (request.audio_seconds for request in successful),
                            default=None,
                        ),
                        "audio_seconds_max": max(
                            (request.audio_seconds for request in successful),
                            default=None,
                        ),
                        "mean_rtf": (
                            statistics.mean(request.rtf for request in successful)
                            if successful
                            else None
                        ),
                        "mean_first_audio_seconds": (
                            statistics.mean(
                                request.first_audio_seconds for request in successful
                            )
                            if successful
                            else None
                        ),
                        "mean_stall_seconds": (
                            statistics.mean(stalls) if stalls else None
                        ),
                        "max_chunk_gap_seconds": max(gaps, default=None),
                        "max_minimum_buffer_seconds": max(
                            required_buffers, default=None
                        ),
                    }
                )
    output.write_text(
        json.dumps(
            {
                "initial_buffer_seconds": initial_buffer_seconds,
                "playback_definition": "Start playback one initial buffer after first arrival; consume at 1x; count stalls only while waiting for subsequent chunks. Excludes network and device playback.",
                "tail_definition": "Deleted reference units in final 20 percent under ASR alignment; a diagnostic, not human judgment or a completeness guarantee.",
                "groups": rows,
                "samples": details,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


async def probe_recovery(manifest: Path, base_url: str, output: Path) -> None:
    records = EvaluationManifest.model_validate_json(manifest.read_text())
    by_id = {sample.sample_id: sample for sample in records.samples}
    async with httpx.AsyncClient(base_url=base_url, timeout=600) as client:
        with output.open("x") as stream:
            for language in ("en", "zh"):
                for index in range(1, 4):
                    for bucket in ("over60s", "15s"):
                        sample = by_id[f"{language}-{bucket}-{index}"]
                        response = await request_audio(
                            client,
                            sample,
                            manifest.parent,
                            1,
                            0,
                            repetition_penalty=1.1,
                        )
                        health = await client.get("/health")
                        record = response.model_dump()
                        record["health_status_after_request"] = health.status_code
                        record["repetition_penalty"] = 1.1
                        record["role"] = (
                            "over_limit" if bucket == "over60s" else "recovery"
                        )
                        serialized = json.dumps(record)
                        stream.write(serialized + "\n")
                        stream.flush()
                        print(serialized, flush=True)


@app.command()
def http(manifest: Path, base_url: str, output: Path) -> None:
    asyncio.run(probe_recovery(manifest, base_url, output))


if __name__ == "__main__":
    app()
else:
    pass
