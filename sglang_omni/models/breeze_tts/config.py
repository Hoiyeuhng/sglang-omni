# SPDX-License-Identifier: Apache-2.0
"""Apple Silicon deployment configuration for Breeze TTS 2."""

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, PipelineConfig, StageConfig


class BreezeFactoryArgs(FactoryArgs):
    device: Literal["mps", "cpu"] = "mps"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    chunk_frames: int = Field(default=2, ge=1, le=32)


class BreezeStageConfig(StageConfig):
    factory: BreezeFactoryArgs = Field(default_factory=BreezeFactoryArgs)


class BreezeTTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "BreezeForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    speech_reference_text_required: ClassVar[bool] = True
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset({"en", "zh"})
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts": BreezeStageConfig
    }
    entry_stage: str = "tts"
    stages: list[StageConfig] = [
        BreezeStageConfig(
            name="tts",
            process="tts",
            factory_path="sglang_omni.models.breeze_tts.stages.create_executor",
            terminal=True,
        )
    ]


EntryClass = BreezeTTSPipelineConfig
