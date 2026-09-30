# SPDX-License-Identifier: Apache-2.0
"""Native Torch Breeze modules with request-local autoregressive state."""

from collections.abc import Generator
from pathlib import Path
from threading import Event

import torch
from accelerate import init_empty_weights
from pydantic import BaseModel, Field, JsonValue
from safetensors import safe_open
from torch import nn
from transformers import LlamaConfig, LlamaModel, Qwen3Config, Qwen3Model
from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2TextConfig
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder

from sglang_omni.models.breeze_tts.sampling import BreezeSamplingParams, sample_token


class BreezeCodecConfig(BaseModel):
    codebook_size: int = Field(gt=0)
    sampling_rate: int = Field(gt=0)


class BreezeCheckpointConfig(BaseModel):
    model_type: str
    backbone_model_type: str
    text_encoder_proj_type: str
    tie_codebooks_embeddings: bool
    hidden_size: int = Field(gt=0)
    audio_embed_size: int = Field(gt=0)
    num_codebooks: int = Field(gt=1)
    vocab_size: int = Field(gt=0)
    codebook_eos_token_id: int
    max_position_embeddings: int = Field(gt=0)
    backbone_config: dict[str, JsonValue]
    depth_decoder_config: dict[str, JsonValue]
    text_encoder_config: dict[str, JsonValue]
    codec_config: BreezeCodecConfig


class BreezeDepthDecoder(nn.Module):
    def __init__(self, configuration: BreezeCheckpointConfig) -> None:
        super().__init__()
        depth_config = LlamaConfig(
            **configuration.depth_decoder_config, attn_implementation="sdpa"
        )
        self.model: LlamaModel = LlamaModel(depth_config)
        self.model.embed_tokens = nn.Embedding(
            configuration.num_codebooks * configuration.vocab_size,
            configuration.audio_embed_size,
        )
        self.model.inputs_embeds_projector = nn.Linear(
            configuration.audio_embed_size, depth_config.hidden_size, bias=False
        )
        self.codebooks_head: nn.Module = nn.Module()
        self.codebooks_head.weight = nn.Parameter(
            torch.empty(
                configuration.num_codebooks - 1,
                depth_config.hidden_size,
                configuration.vocab_size,
            )
        )


class BreezeModel(nn.Module):
    def __init__(self, configuration: BreezeCheckpointConfig) -> None:
        super().__init__()
        self.configuration: BreezeCheckpointConfig = configuration
        if (
            configuration.model_type != "breeze"
            or configuration.backbone_model_type != "qwen3"
            or configuration.text_encoder_proj_type != "linear"
            or not configuration.tie_codebooks_embeddings
            or configuration.audio_embed_size != configuration.hidden_size
        ):
            raise ValueError("Unsupported Breeze checkpoint architecture")
        else:
            pass
        text_config = T5Gemma2TextConfig(
            **configuration.text_encoder_config, attn_implementation="sdpa"
        )
        self.text_encoder: T5Gemma2TextEncoder = T5Gemma2TextEncoder(text_config)
        self.text_encoder_proj: nn.Linear = nn.Linear(
            text_config.hidden_size, configuration.hidden_size, bias=False
        )
        backbone_config = Qwen3Config(
            **configuration.backbone_config, attn_implementation="sdpa"
        )
        self.backbone_model: Qwen3Model = Qwen3Model(backbone_config)
        self.backbone_model.embed_tokens = None
        self.lm_head: nn.Linear = nn.Linear(
            configuration.hidden_size, configuration.vocab_size + 1, bias=False
        )
        self.depth_decoder: BreezeDepthDecoder = BreezeDepthDecoder(configuration)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, device: torch.device, dtype: torch.dtype
    ) -> "BreezeModel":
        configuration = BreezeCheckpointConfig.model_validate_json(
            (checkpoint / "config.json").read_text()
        )
        with init_empty_weights(include_buffers=False):
            model = cls(configuration)
        weight_files = sorted(checkpoint.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No Breeze safetensors found in {checkpoint}")
        else:
            pass
        parameters: dict[str, torch.Tensor] = {}
        for weight_file in weight_files:
            with safe_open(weight_file, framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    if (
                        name.startswith("codec_model.")
                        or name == "embed_text_tokens.weight"
                    ):
                        continue
                    elif name in parameters:
                        raise ValueError(f"Duplicate Breeze parameter: {name}")
                    else:
                        parameters[name] = shard.get_tensor(name)
        model.load_state_dict(parameters, strict=True, assign=True)
        model.to(device=device, dtype=dtype).eval()
        model.lm_head.float()
        model.depth_decoder.codebooks_head.float()
        return model

    def embed_audio(self, codes: torch.Tensor) -> torch.Tensor:
        offsets = (
            torch.arange(self.configuration.num_codebooks, device=codes.device)
            * self.configuration.vocab_size
        )
        return self.depth_decoder.model.embed_tokens(codes + offsets).sum(-2)

    @torch.inference_mode()
    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.text_encoder(input_ids=input_ids).last_hidden_state
        return self.text_encoder_proj(hidden)

    @torch.inference_mode()
    def generate_frames(
        self,
        prompts: list[torch.Tensor],
        params: BreezeSamplingParams,
        cancelled: Event,
    ) -> Generator[torch.Tensor, None, None]:
        """Yield complete frames; caches and RNG state belong to this iteration."""
        if cancelled.is_set():
            return
        elif len(prompts) not in (1, 2):
            raise ValueError("Breeze requires one or two prompt branches")
        else:
            pass
        remaining = self.configuration.max_position_embeddings - max(
            prompt.shape[1] for prompt in prompts
        )
        if remaining <= 0:
            raise ValueError("Breeze prompt leaves no generation room")
        else:
            pass
        generator = torch.Generator(device="cpu").manual_seed(params.seed)
        outputs = [
            self.backbone_model(inputs_embeds=prompt, use_cache=True)
            for prompt in prompts
        ]
        history: list[int] = []
        codebook_size = self.configuration.codec_config.codebook_size
        frame_limit = min(remaining, params.max_new_tokens)
        for frame_index in range(frame_limit):
            if cancelled.is_set():
                return
            else:
                pass
            hidden = torch.cat(
                [output.last_hidden_state[:, -1] for output in outputs], dim=0
            )
            first_code = sample_token(
                self.lm_head(hidden.float()),
                params,
                generator,
                history,
                codebook_size=codebook_size,
                allow_eos=True,
            )
            if first_code == self.configuration.vocab_size:
                return
            else:
                pass
            codes = [first_code]
            first_codes = torch.full(
                (len(prompts),), first_code, device=hidden.device, dtype=torch.long
            )
            embeddings = torch.stack(
                (hidden, self.depth_decoder.model.embed_tokens(first_codes)), dim=1
            )
            depth_cache = None
            for codebook_index in range(1, self.configuration.num_codebooks):
                if cancelled.is_set():
                    return
                else:
                    pass
                depth_output = self.depth_decoder.model(
                    inputs_embeds=self.depth_decoder.model.inputs_embeds_projector(
                        embeddings
                    ),
                    past_key_values=depth_cache,
                    use_cache=True,
                )
                depth_cache = depth_output.past_key_values
                logits = (
                    depth_output.last_hidden_state[:, -1].float()
                    @ self.depth_decoder.codebooks_head.weight[codebook_index - 1]
                )
                code = sample_token(
                    logits,
                    params,
                    generator,
                    [],
                    codebook_size=codebook_size,
                    allow_eos=False,
                )
                codes.append(code)
                code_tokens = torch.full_like(
                    first_codes, code + codebook_index * self.configuration.vocab_size
                )
                embeddings = self.depth_decoder.model.embed_tokens(
                    code_tokens
                ).unsqueeze(1)
            frame = torch.tensor(codes, device=hidden.device, dtype=torch.long)
            yield frame
            if cancelled.is_set() or frame_index + 1 == frame_limit:
                return
            else:
                pass
            history.append(first_code)
            next_embedding = self.embed_audio(frame.reshape(1, 1, -1))
            outputs = [
                self.backbone_model(
                    inputs_embeds=next_embedding,
                    past_key_values=output.past_key_values,
                    use_cache=True,
                )
                for output in outputs
            ]
