# SPDX-License-Identifier: Apache-2.0
"""Score saved Breeze waveforms with one pinned ASR model and normalization."""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import typer
from jiwer import process_words
from opencc import OpenCC
from pydantic import BaseModel
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from benchmarks.tasks.asr import normalize_text


class RequestRecord(BaseModel):
    sample_id: str
    language: str
    warmup: bool
    status: Literal["success", "error", "timeout"]
    audio: str = ""
    text: str = ""
    error: str = ""


def main(model_path: Path, results: list[Path], long_form: bool = False) -> None:
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="sdpa"
    )
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device="mps",
        chunk_length_s=None if long_form else 30,
        stride_length_s=None if long_form else 5,
    )
    simplified = OpenCC("t2s")
    for directory in results:
        score_path = directory / "scores.jsonl"
        if score_path.exists():
            raise FileExistsError(score_path)
        else:
            pass
        (directory / "scorer.json").write_text(
            json.dumps(
                {
                    "model": str(model_path),
                    "script_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "model_pin": json.loads(
                        (model_path.parent / "asr-model.json").read_text()
                    ),
                    "packages": {
                        name: importlib.metadata.version(name)
                        for name in (
                            "torch",
                            "transformers",
                            "jiwer",
                            "openai-whisper",
                            "opencc-python-reimplemented",
                        )
                    },
                    "device": "mps",
                    "dtype": "float32",
                    "normalization": "Repository benchmarks.tasks.asr.normalize_text; t2s on both Chinese texts before character scoring",
                    "generation": {
                        "do_sample": False,
                        "task": "transcribe",
                        "language": "per-sample",
                        "chunk_seconds": None if long_form else 30,
                        "stride_seconds": None if long_form else 5,
                        "long_form": long_form,
                        "return_timestamps": long_form,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        with score_path.open("x") as scores:
            for line in (directory / "requests.jsonl").read_text().splitlines():
                request = RequestRecord.model_validate_json(line)
                if request.warmup:
                    continue
                elif request.status != "success":
                    record = {
                        "sample_id": request.sample_id,
                        "language": request.language,
                        "status": "generation_failed",
                        "error": request.error,
                    }
                else:
                    try:
                        waveform, sample_rate_hz = sf.read(
                            directory / request.audio, dtype="float32"
                        )
                        divisor = math.gcd(sample_rate_hz, 16000)
                        waveform = resample_poly(
                            waveform, 16000 // divisor, sample_rate_hz // divisor
                        ).astype(np.float32)
                        transcript = transcriber(
                            {"raw": waveform, "sampling_rate": 16000},
                            return_timestamps=True if long_form else None,
                            generate_kwargs={
                                "language": request.language,
                                "task": "transcribe",
                                "do_sample": False,
                            },
                        )["text"]
                        target = request.text
                        if request.language == "zh":
                            transcript = simplified.convert(transcript)
                            target = simplified.convert(target)
                        else:
                            pass
                        reference = normalize_text(target, request.language)
                        hypothesis = normalize_text(transcript, request.language)
                        comparison = process_words(reference, hypothesis)
                        record = {
                            "sample_id": request.sample_id,
                            "language": request.language,
                            "status": "success",
                            "transcript": transcript,
                            "reference_normalized": reference,
                            "hypothesis_normalized": hypothesis,
                            "substitutions": comparison.substitutions,
                            "deletions": comparison.deletions,
                            "insertions": comparison.insertions,
                            "hits": comparison.hits,
                            "error_rate": comparison.wer,
                        }
                    except (RuntimeError, ValueError) as error:
                        record = {
                            "sample_id": request.sample_id,
                            "language": request.language,
                            "status": "scoring_failed",
                            "error": f"{type(error).__name__}: {error}",
                        }
                serialized = json.dumps(record, ensure_ascii=False)
                scores.write(serialized + "\n")
                scores.flush()
                print(serialized, flush=True)


if __name__ == "__main__":
    typer.run(main)
else:
    pass
