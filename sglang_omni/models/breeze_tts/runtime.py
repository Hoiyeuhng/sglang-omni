# SPDX-License-Identifier: Apache-2.0
"""Breeze prompt preparation and incremental audio generation."""

import importlib
from collections.abc import Generator
from contextlib import closing
from pathlib import Path
from threading import Event
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from transformers import GemmaTokenizer

from sglang_omni.audio.qwen3_tts_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)
from sglang_omni.audio.qwen3_tts_compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.breeze_tts.model import BreezeModel
from sglang_omni.models.breeze_tts.request import (
    BreezeRequestError,
    BreezeSpeechRequest,
)
from sglang_omni.utils.audio import AudioDecodeError, load_audio


class EncodedAudio(Protocol):
    audio_codes: list[torch.Tensor]


class CodecModules(Protocol):
    decoder: nn.Module


class AudioTokenizer(Protocol):
    model: CodecModules

    def encode(self, audio: NDArray[np.float32], sr: int) -> EncodedAudio: ...

    def get_input_sample_rate(self) -> int: ...

    def get_output_sample_rate(self) -> int: ...


class BreezeRuntime:
    def __init__(
        self,
        model: BreezeModel,
        tokenizer: GemmaTokenizer,
        audio_tokenizer: AudioTokenizer,
        chunk_frames: int,
    ) -> None:
        if chunk_frames < 1:
            raise ValueError("Breeze chunk_frames must be positive")
        else:
            pass
        self.model: BreezeModel = model
        self.tokenizer: GemmaTokenizer = tokenizer
        self.audio_tokenizer: AudioTokenizer = audio_tokenizer
        self.chunk_frames: int = chunk_frames
        self.codec: Qwen3TTSIncrementalDecoder = Qwen3TTSIncrementalDecoder(
            audio_tokenizer.model.decoder
        )
        self.sample_rate_hz: int = audio_tokenizer.get_output_sample_rate()
        self.device: torch.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        device: torch.device,
        dtype: torch.dtype,
        chunk_frames: int,
    ) -> "BreezeRuntime":
        apply_qwen_tts_transformers_compatibility_patches()
        codec_package = importlib.import_module("qwen_tts")
        audio_tokenizer = codec_package.Qwen3TTSTokenizer.from_pretrained(
            str(checkpoint / "audio_tokenizer"),
            device_map=str(device),
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        model = BreezeModel.from_checkpoint(checkpoint, device, dtype)
        tokenizer = GemmaTokenizer.from_pretrained(checkpoint)
        return cls(model, tokenizer, audio_tokenizer, chunk_frames)

    @torch.inference_mode()
    def prepare_prompts(
        self, request: BreezeSpeechRequest, cancelled: Event
    ) -> list[torch.Tensor]:
        if cancelled.is_set():
            raise InterruptedError("Breeze request cancelled")
        else:
            pass
        reference_segments: list[torch.Tensor] = []
        if request.ref_audio:
            sample_rate = self.audio_tokenizer.get_input_sample_rate()
            try:
                waveform = load_audio(request.ref_audio, target_sample_rate=sample_rate)
            except AudioDecodeError as error:
                raise BreezeRequestError("Could not decode reference audio") from error
            maximum_samples = (
                (self.model.configuration.max_position_embeddings - 1)
                * self.codec.total_upsample
                * sample_rate
                // self.sample_rate_hz
            )
            if waveform.size == 0:
                raise BreezeRequestError("Reference audio is empty")
            elif waveform.size > maximum_samples:
                raise BreezeRequestError(
                    "Reference audio exceeds the model context window"
                )
            else:
                pass
            if cancelled.is_set():
                raise InterruptedError("Breeze request cancelled")
            else:
                pass
            encoded = self.audio_tokenizer.encode(waveform, sr=sample_rate)
            if cancelled.is_set():
                raise InterruptedError("Breeze request cancelled")
            else:
                pass
            codes = encoded.audio_codes[0].to(device=self.device, dtype=torch.long)
            eos = codes.new_full(
                (1, self.model.configuration.num_codebooks),
                self.model.configuration.codebook_eos_token_id,
            )
            reference_segments = [
                self.encode_text(f"[S0]{request.ref_text}"),
                self.model.embed_audio(torch.cat((codes, eos)).unsqueeze(0)),
            ]
        else:
            pass
        target = f"[S0]{request.text}"
        if request.instructions:
            target = f"[S0]<ins_bos>{request.instructions}<ins_eos>{request.text}"
        else:
            pass
        prompts = [torch.cat([*reference_segments, self.encode_text(target)], dim=1)]
        if cancelled.is_set():
            raise InterruptedError("Breeze request cancelled")
        else:
            pass
        if request.instructions and request.sampling.cfg_scale != 1:
            prompts.append(
                torch.cat(
                    [*reference_segments, self.encode_text(f"[S0]{request.text}")],
                    dim=1,
                )
            )
        else:
            pass
        if cancelled.is_set():
            raise InterruptedError("Breeze request cancelled")
        elif (
            max(prompt.shape[1] for prompt in prompts)
            >= self.model.configuration.max_position_embeddings
        ):
            raise BreezeRequestError("Prompt leaves no generation room")
        else:
            return prompts

    def encode_text(self, text: str) -> torch.Tensor:
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] >= self.model.configuration.max_position_embeddings:
            raise BreezeRequestError("Text exceeds the model context window")
        else:
            return self.model.encode_text(input_ids.to(self.device))

    @torch.inference_mode()
    def decode_frames(
        self, frames: list[torch.Tensor], state: Qwen3TTSIncrementalCodecState
    ) -> NDArray[np.float32]:
        codes = torch.stack(frames).transpose(0, 1).unsqueeze(0)
        waveform = self.codec.decode(codes, state).float().cpu().numpy().reshape(-1)
        if not np.isfinite(waveform).all():
            raise RuntimeError("Breeze codec produced non-finite audio")
        else:
            return waveform

    @torch.inference_mode()
    def stream(
        self, request: BreezeSpeechRequest, cancelled: Event
    ) -> Generator[NDArray[np.float32], None, None]:
        if cancelled.is_set():
            return
        else:
            pass
        prompts = self.prepare_prompts(request, cancelled)
        state = self.codec.init_state(
            batch_size=1, device=self.device, dtype=torch.float32
        )
        pending_frames: list[torch.Tensor] = []
        with closing(
            self.model.generate_frames(prompts, request.sampling, cancelled)
        ) as frames:
            for frame in frames:
                pending_frames.append(frame)
                if len(pending_frames) == self.chunk_frames:
                    audio = self.decode_frames(pending_frames, state)
                    pending_frames.clear()
                    yield audio
                else:
                    pass
        if pending_frames and not cancelled.is_set():
            yield self.decode_frames(pending_frames, state)
        else:
            pass
