# SPDX-License-Identifier: Apache-2.0
"""Validated Breeze speech requests at the pipeline boundary."""

import secrets
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sglang_omni.models.breeze_tts.sampling import BreezeSamplingParams
from sglang_omni.proto.request import StagePayload


class BreezeRequestError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(f"Invalid Breeze-TTS-2 request: {message}")


class BreezeSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    text: str = Field(min_length=1)
    instructions: str = ""
    ref_audio: str | None = None
    ref_text: str = ""
    sampling: BreezeSamplingParams = Field(default_factory=BreezeSamplingParams)

    @model_validator(mode="after")
    def validate_reference(self) -> "BreezeSpeechRequest":
        if bool(self.ref_audio) != bool(self.ref_text):
            raise ValueError(
                "Reference audio and its transcript must be provided together"
            )
        elif self.sampling.cfg_scale != 1 and not self.instructions:
            raise ValueError("cfg_scale other than 1 requires voice instructions")
        else:
            return self


class SpeechReference(BaseModel):
    text: str = ""
    audio_path: str | None = None
    audio: str | None = None
    ref_audio: str | None = None
    data: str | None = None
    media_type: str = "audio/wav"
    vq_codes: None = None


class SpeechInput(BaseModel):
    text: str
    references: list[SpeechReference] = Field(default_factory=list, max_length=1)


class SpeechOptions(BaseModel):
    instructions: str = ""
    ref_audio: str | None = None
    ref_text: str = ""
    cfg_scale: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=0, lt=2**64, strict=True)
    voice: str = "default"
    uploaded_voice_name: str | None = None
    speed: Literal[1.0] = 1.0
    language: Literal["auto", "Auto", "en", "zh", "English", "Chinese"] | None = None
    task_type: Literal["Base", "VoiceDesign"] | None = None
    x_vector_only_mode: Literal[False] | None = None
    token_count: None = None
    duration_tokens: None = None
    suppress_bootstrap_silence: Literal[False] | None = None
    explicit_generation_params: list[str] = Field(default_factory=list)


def parse_request(payload: StagePayload) -> BreezeSpeechRequest:
    try:
        inputs = payload.request.inputs
        if isinstance(inputs, str):
            speech_input = SpeechInput(text=inputs)
        else:
            speech_input = SpeechInput.model_validate(inputs)
        metadata = payload.request.metadata
        parameters = payload.request.params
        options = SpeechOptions.model_validate(metadata.get("tts_params", {}))
        if options.voice not in ("", "default") and not options.uploaded_voice_name:
            raise BreezeRequestError(
                "Use voice-design instructions or a reference voice"
            )
        else:
            pass
        reference_audio = options.ref_audio
        reference_text = options.ref_text
        if speech_input.references:
            reference = speech_input.references[0]
            reference_text = reference.text or reference_text
            if reference.data:
                reference_audio = f"data:{reference.media_type};base64,{reference.data}"
            else:
                reference_audio = (
                    reference.audio_path or reference.ref_audio or reference.audio
                )
        else:
            pass
        explicit = (
            options.explicit_generation_params
            if "tts_params" in metadata
            else parameters.keys()
        )
        sampling_values = {
            name: parameters[name]
            for name in BreezeSamplingParams.model_fields
            if name in explicit and parameters.get(name) is not None
        }
        sampling_values["cfg_scale"] = parameters.get("cfg_scale", options.cfg_scale)
        seed = options.seed if options.seed is not None else parameters.get("seed")
        sampling_values["seed"] = secrets.randbits(63) if seed is None else seed
        return BreezeSpeechRequest(
            text=speech_input.text,
            instructions=options.instructions,
            ref_audio=reference_audio,
            ref_text=reference_text,
            sampling=BreezeSamplingParams.model_validate(sampling_values),
        )
    except ValidationError as error:
        raise BreezeRequestError(str(error)) from error
