# SPDX-License-Identifier: Apache-2.0
"""Measure streamed HTTP requests queued behind Breeze serial execution."""

import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import httpx
import typer
from pydantic import BaseModel, Field

from scripts.apple.breeze_eval_common import (
    AudioChunk,
    EvaluationManifest,
    EvaluationSample,
)


class HttpResult(BaseModel):
    sample_id: str
    language: str
    concurrency: int
    repetition: int
    status: Literal["success", "error"]
    elapsed_seconds: float
    first_audio_seconds: float | None = None
    audio_seconds: float | None = None
    audio_sha256: str | None = None
    error: str | None = None
    chunks: list[AudioChunk] = Field(default_factory=list)


async def request_audio(
    client: httpx.AsyncClient,
    sample: EvaluationSample,
    manifest_directory: Path,
    concurrency: int,
    repetition: int,
    repetition_penalty: float = 1.0,
) -> HttpResult:
    reference = (manifest_directory / sample.ref_audio).read_bytes()
    if hashlib.sha256(reference).hexdigest() != sample.reference_sha256:
        raise ValueError(f"Reference changed: {sample.sample_id}")
    else:
        pass
    payload = {
        "input": sample.text,
        "language": sample.language,
        "ref_audio": "data:audio/wav;base64," + base64.b64encode(reference).decode(),
        "ref_text": sample.ref_text,
        "instructions": sample.instructions,
        "cfg_scale": sample.cfg_scale,
        "seed": 42,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": 750,
        "stream": True,
        "response_format": "pcm",
    }
    started_seconds = time.perf_counter()
    first_audio_seconds = None
    chunks = []
    chunk_timings: list[AudioChunk] = []
    try:
        async with client.stream("POST", "/v1/audio/speech", json=payload) as response:
            if response.status_code != 200:
                raise ValueError(
                    f"HTTP {response.status_code}: {(await response.aread()).decode()}"
                )
            else:
                pass
            async for chunk in response.aiter_bytes():
                if chunk and first_audio_seconds is None:
                    first_audio_seconds = time.perf_counter() - started_seconds
                else:
                    pass
                chunks.append(chunk)
                chunk_timings.append(
                    AudioChunk(
                        arrival_seconds=time.perf_counter() - started_seconds,
                        duration_seconds=len(chunk) / (24000 * 2),
                    )
                )
        waveform = b"".join(chunks)
        if not waveform or len(waveform) % 2:
            raise ValueError("Empty or malformed signed-16-bit PCM")
        else:
            pass
        return HttpResult(
            sample_id=sample.sample_id,
            language=sample.language,
            concurrency=concurrency,
            repetition=repetition,
            status="success",
            elapsed_seconds=time.perf_counter() - started_seconds,
            first_audio_seconds=first_audio_seconds,
            audio_seconds=len(waveform) / (24000 * 2),
            audio_sha256=hashlib.sha256(waveform).hexdigest(),
            chunks=chunk_timings,
        )
    except (httpx.HTTPError, ValueError) as error:
        return HttpResult(
            sample_id=sample.sample_id,
            language=sample.language,
            concurrency=concurrency,
            repetition=repetition,
            status="error",
            elapsed_seconds=time.perf_counter() - started_seconds,
            error=f"{type(error).__name__}: {error}",
        )


async def run_http(
    base_url: str, manifest: Path, output: Path, count: int, repetitions: int
) -> None:
    inputs = EvaluationManifest.model_validate_json(manifest.read_text())
    samples = [
        sample
        for language in ("en", "zh")
        for sample in [row for row in inputs.samples if row.language == language][
            :count
        ]
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "http-metadata.json").write_text(
        json.dumps(
            {
                "sample_ids": [sample.sample_id for sample in samples],
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "client_concurrency": [1, 2, 4],
                "repetitions": repetitions,
                "scope": "HTTP PCM streaming, includes server queueing; serial model execution",
                "sampling": {
                    "seed": 42,
                    "temperature": 0.9,
                    "top_k": 50,
                    "top_p": 1.0,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": 750,
                },
            },
            indent=2,
        )
        + "\n"
    )
    async with httpx.AsyncClient(base_url=base_url, timeout=900) as client:
        with (
            (output / "http-requests.jsonl").open("x") as request_records,
            (output / "http-groups.jsonl").open("x") as group_records,
        ):
            for concurrency in (1, 2, 4):
                warmup = await request_audio(
                    client, samples[0], manifest.parent, concurrency, -1
                )
                request_records.write(warmup.model_dump_json() + "\n")
                request_records.flush()
                semaphore = asyncio.Semaphore(concurrency)

                async def measured(
                    sample: EvaluationSample, repetition: int
                ) -> HttpResult:
                    async with semaphore:
                        result = await request_audio(
                            client, sample, manifest.parent, concurrency, repetition
                        )
                        request_records.write(result.model_dump_json() + "\n")
                        request_records.flush()
                        print(result.model_dump_json(), flush=True)
                        return result

                for repetition in range(repetitions):
                    started_seconds = time.perf_counter()
                    results = await asyncio.gather(
                        *(measured(sample, repetition) for sample in samples)
                    )
                    elapsed_seconds = time.perf_counter() - started_seconds
                    successes = sum(result.status == "success" for result in results)
                    group_records.write(
                        json.dumps(
                            {
                                "concurrency": concurrency,
                                "repetition": repetition,
                                "elapsed_seconds": elapsed_seconds,
                                "requests": len(results),
                                "successful": successes,
                                "successful_requests_per_second": successes
                                / elapsed_seconds,
                            }
                        )
                        + "\n"
                    )
                    group_records.flush()


def main(
    base_url: str,
    manifest: Path,
    output: Path,
    count: int = 4,
    repetitions: int = 2,
) -> None:
    if count < 1 or repetitions < 1:
        raise ValueError("count and repetitions must be positive")
    else:
        asyncio.run(run_http(base_url, manifest, output, count, repetitions))


if __name__ == "__main__":
    typer.run(main)
else:
    pass
