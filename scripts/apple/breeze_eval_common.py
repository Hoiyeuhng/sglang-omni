# SPDX-License-Identifier: Apache-2.0
"""Shared records and timing boundary for paired Breeze evaluation."""

import hashlib
import json
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FrameType
from typing import Literal, Protocol

import numpy as np
import soundfile as sf
import torch
import transformers
from numpy.typing import NDArray
from pydantic import BaseModel


class EvaluationSample(BaseModel):
    sample_id: str
    language: Literal["en", "zh"]
    text: str
    ref_text: str
    ref_audio: str
    reference_sha256: str
    instructions: str
    cfg_scale: float


class EvaluationManifest(BaseModel):
    dataset: str
    revision: str
    selection: str
    samples: list[EvaluationSample]


@dataclass(kw_only=True)
class AudioChunk:
    arrival_seconds: float
    duration_seconds: float


@dataclass(kw_only=True)
class GeneratedAudio:
    waveform: NDArray[np.float32]
    sample_rate_hz: int
    frames: int
    termination: Literal["eos", "frame_limit", "context_limit"]
    first_audio_seconds: float
    chunks: list[AudioChunk] = field(default_factory=list)


class EvaluationEngine(Protocol):
    def generate(
        self, sample: EvaluationSample, started_seconds: float
    ) -> GeneratedAudio: ...


def timeout_request(signal_number: int, frame: FrameType | None) -> None:
    """SIGALRM retains the standard signal callback signature."""
    raise TimeoutError("Request exceeded the evaluation deadline")


def evaluate(
    engine: EvaluationEngine,
    manifest: Path,
    checkpoint: Path,
    output: Path,
    implementation: str,
    repetition_penalty: float,
    max_new_tokens: int,
    limit_per_language: int,
    timeout_seconds: float,
) -> None:
    if limit_per_language < 1 or timeout_seconds <= 0:
        raise ValueError("Sample limit and timeout must be positive")
    elif (output / "requests.jsonl").exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    else:
        pass
    records = EvaluationManifest.model_validate_json(manifest.read_text())
    output.mkdir(parents=True, exist_ok=True)
    weight_hashes = {}
    for weight_file in sorted(checkpoint.rglob("*.safetensors")):
        with weight_file.open("rb") as weight_stream:
            weight_hashes[str(weight_file.relative_to(checkpoint))] = (
                hashlib.file_digest(weight_stream, "sha256").hexdigest()
            )
    metadata = {
        "implementation": implementation,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_diff": subprocess.check_output(["git", "diff"], text=True),
        "script_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("breeze_eval_*.py")
        },
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "checkpoint_config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        "checkpoint_weight_sha256": weight_hashes,
        "dtype": "bfloat16",
        "codec_dtype": "float32",
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": repetition_penalty,
        "seed": 42,
        "rng_note": "Equal seeds do not imply equal random draws across the CPU and MPS samplers",
        "max_new_tokens": max_new_tokens,
        "timeout_seconds": timeout_seconds,
        "warmup": "First selected input per language, excluded from summaries",
        "timing_scope": "Prompt preparation through CPU waveform availability; no HTTP or file write",
        "memory_scope": "Post-request live MPS tensor and driver allocations, not peaks",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    previous_signal = signal.signal(signal.SIGALRM, timeout_request)
    try:
        with (output / "requests.jsonl").open("x") as results:
            for language in dict.fromkeys(
                sample.language for sample in records.samples
            ):
                samples = [
                    sample for sample in records.samples if sample.language == language
                ][:limit_per_language]
                for index, sample in enumerate([samples[0], *samples]):
                    is_warmup = index == 0
                    sample = sample.model_copy()
                    if sample.ref_audio:
                        reference = manifest.parent / sample.ref_audio
                        if (
                            hashlib.sha256(reference.read_bytes()).hexdigest()
                            != sample.reference_sha256
                        ):
                            raise ValueError(f"Reference changed: {sample.sample_id}")
                        else:
                            sample.ref_audio = str(reference.resolve())
                    else:
                        pass
                    torch.mps.synchronize()
                    started_seconds = time.perf_counter()
                    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
                    try:
                        audio = engine.generate(sample, started_seconds)
                        torch.mps.synchronize()
                        elapsed_seconds = time.perf_counter() - started_seconds
                        if (
                            not audio.waveform.size
                            or not np.isfinite(audio.waveform).all()
                        ):
                            raise ValueError("Empty or non-finite generated audio")
                        else:
                            pass
                        duration_seconds = audio.waveform.size / audio.sample_rate_hz
                        audio_name = (
                            f"{'warmup-' if is_warmup else ''}{sample.sample_id}.wav"
                        )
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        sf.write(
                            output / audio_name, audio.waveform, audio.sample_rate_hz
                        )
                        record = {
                            "sample_id": sample.sample_id,
                            "language": language,
                            "warmup": is_warmup,
                            "status": "success",
                            "audio": audio_name,
                            "text": sample.text,
                            "elapsed_seconds": elapsed_seconds,
                            "audio_seconds": duration_seconds,
                            "first_audio_seconds": audio.first_audio_seconds,
                            "rtf": elapsed_seconds / duration_seconds,
                            "frames": audio.frames,
                            "chunks": [asdict(chunk) for chunk in audio.chunks],
                            "termination": audio.termination,
                            "peak_amplitude": float(np.abs(audio.waveform).max()),
                            "rms_amplitude": float(np.sqrt(np.mean(audio.waveform**2))),
                            "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                            "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                        }
                    except (
                        RuntimeError,
                        ValueError,
                        TimeoutError,
                        IndexError,
                    ) as error:
                        record = {
                            "sample_id": sample.sample_id,
                            "language": language,
                            "warmup": is_warmup,
                            "status": (
                                "timeout"
                                if isinstance(error, TimeoutError)
                                else "error"
                            ),
                            "elapsed_seconds": time.perf_counter() - started_seconds,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    serialized = json.dumps(record, ensure_ascii=False)
                    results.write(serialized + "\n")
                    results.flush()
                    print(serialized, flush=True)
    finally:
        signal.signal(signal.SIGALRM, previous_signal)
