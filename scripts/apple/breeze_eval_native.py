# SPDX-License-Identifier: Apache-2.0
"""Measure the native Breeze BF16 runtime on frozen evaluation inputs."""

import time
from pathlib import Path
from threading import Event

import numpy as np
import torch
import typer

from scripts.apple.breeze_eval_common import (
    AudioChunk,
    EvaluationSample,
    GeneratedAudio,
    evaluate,
)
from sglang_omni.models.breeze_tts.request import BreezeSpeechRequest
from sglang_omni.models.breeze_tts.runtime import BreezeRuntime
from sglang_omni.models.breeze_tts.sampling import BreezeSamplingParams


class NativeEvaluation:
    def __init__(
        self, checkpoint: Path, repetition_penalty: float, max_new_tokens: int
    ) -> None:
        self.runtime: BreezeRuntime = BreezeRuntime.from_checkpoint(
            checkpoint, torch.device("mps"), torch.bfloat16, 2
        )
        self.repetition_penalty: float = repetition_penalty
        self.max_new_tokens: int = max_new_tokens

    @torch.inference_mode()
    def generate(
        self, sample: EvaluationSample, started_seconds: float
    ) -> GeneratedAudio:
        sampling = BreezeSamplingParams(
            seed=42,
            cfg_scale=sample.cfg_scale,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
        )
        request = BreezeSpeechRequest(
            text=sample.text,
            ref_text=sample.ref_text,
            ref_audio=sample.ref_audio or None,
            instructions=sample.instructions,
            sampling=sampling,
        )
        cancellation = Event()
        prompts = self.runtime.prepare_prompts(request, cancellation)
        remaining = self.runtime.model.configuration.max_position_embeddings - max(
            prompt.shape[1] for prompt in prompts
        )
        state = self.runtime.codec.init_state(
            batch_size=1, device=self.runtime.device, dtype=torch.float32
        )
        pending = []
        chunks = []
        chunk_timings: list[AudioChunk] = []
        frame_count = 0
        first_audio_seconds = None
        for frame in self.runtime.model.generate_frames(
            prompts, sampling, cancellation
        ):
            pending.append(frame)
            frame_count += 1
            if len(pending) == self.runtime.chunk_frames:
                chunks.append(self.runtime.decode_frames(pending, state))
                chunk_timings.append(
                    AudioChunk(
                        arrival_seconds=time.perf_counter() - started_seconds,
                        duration_seconds=chunks[-1].size / self.runtime.sample_rate_hz,
                    )
                )
                pending.clear()
                if first_audio_seconds is None:
                    first_audio_seconds = time.perf_counter() - started_seconds
                else:
                    pass
            else:
                pass
        if pending:
            chunks.append(self.runtime.decode_frames(pending, state))
            chunk_timings.append(
                AudioChunk(
                    arrival_seconds=time.perf_counter() - started_seconds,
                    duration_seconds=chunks[-1].size / self.runtime.sample_rate_hz,
                )
            )
        else:
            pass
        if first_audio_seconds is None:
            first_audio_seconds = time.perf_counter() - started_seconds
        else:
            pass
        if frame_count >= remaining:
            termination = "context_limit"
        elif frame_count >= self.max_new_tokens:
            termination = "frame_limit"
        else:
            termination = "eos"
        return GeneratedAudio(
            waveform=np.concatenate(chunks) if chunks else np.empty(0, np.float32),
            sample_rate_hz=self.runtime.sample_rate_hz,
            frames=frame_count,
            termination=termination,
            first_audio_seconds=first_audio_seconds,
            chunks=chunk_timings,
        )


def main(
    checkpoint: Path,
    manifest: Path,
    output: Path,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 750,
    limit_per_language: int = 50,
    timeout_seconds: float = 300,
) -> None:
    engine = NativeEvaluation(checkpoint, repetition_penalty, max_new_tokens)
    evaluate(
        engine,
        manifest,
        checkpoint,
        output,
        "native-mps-sdpa-streaming",
        repetition_penalty,
        max_new_tokens,
        limit_per_language,
        timeout_seconds,
    )


if __name__ == "__main__":
    typer.run(main)
else:
    pass
