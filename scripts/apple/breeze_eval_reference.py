# SPDX-License-Identifier: Apache-2.0
"""Measure the standard Breeze BF16 runtime on frozen evaluation inputs."""

import time
from pathlib import Path

import torch
import typer
from breeze_infer.runtime import (
    load_runtime,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from models.breeze import BreezeForConditionalGeneration
from qwen_tts import Qwen3TTSTokenizer
from transformers import PreTrainedTokenizerBase

from scripts.apple.breeze_eval_common import EvaluationSample, GeneratedAudio, evaluate


class ReferenceEvaluation:
    def __init__(
        self, checkpoint: Path, repetition_penalty: float, max_new_tokens: int
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase
        self.model: BreezeForConditionalGeneration
        self.codec: Qwen3TTSTokenizer
        self.tokenizer, self.model, self.codec = load_runtime(
            checkpoint, device="mps", attn_implementation="eager"
        )
        self.codec.model.float()
        update_generation_config_for_breeze(self.model)
        self.repetition_penalty: float = repetition_penalty
        self.max_new_tokens: int = max_new_tokens

    @torch.inference_mode()
    def generate(
        self, sample: EvaluationSample, started_seconds: float
    ) -> GeneratedAudio:
        set_all_seeds(42)
        request = {"id": sample.sample_id, "text": sample.text, "speaker": "S0"}
        if sample.instructions:
            request["instruction"] = sample.instructions
        else:
            pass
        if sample.ref_audio:
            request["ref_audio_path"] = sample.ref_audio
            request["ref_text"] = sample.ref_text
        else:
            pass
        inputs = prepare_inputs(
            self.tokenizer,
            self.codec,
            self.model,
            [request],
            get_template(select_template_name(request)),
            guidance_scale=sample.cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        codes = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            output_audio=False,
        )[0]
        eos = (codes == self.model.config.codebook_pad_token_id).all(dim=-1).nonzero()
        if eos.numel():
            codes = codes[: int(eos[0])]
            termination = "eos"
        else:
            termination = "frame_limit"
        waveform, sample_rate_hz = self.codec.decode({"audio_codes": [codes]})
        torch.mps.synchronize()
        return GeneratedAudio(
            waveform=waveform[0],
            sample_rate_hz=sample_rate_hz,
            frames=len(codes),
            termination=termination,
            first_audio_seconds=time.perf_counter() - started_seconds,
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
    engine = ReferenceEvaluation(checkpoint, repetition_penalty, max_new_tokens)
    evaluate(
        engine,
        manifest,
        checkpoint,
        output,
        "reference-mps-eager-nonstreaming",
        repetition_penalty,
        max_new_tokens,
        limit_per_language,
        timeout_seconds,
    )


if __name__ == "__main__":
    typer.run(main)
else:
    pass
