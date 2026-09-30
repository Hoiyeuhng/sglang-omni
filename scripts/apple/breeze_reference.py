# SPDX-License-Identifier: Apache-2.0
"""Export pinned Breeze reference generations for the opt-in parity tests."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import transformers
import typer
from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
from breeze_infer.templates import get_template, prepare_inputs, select_template_name

REFERENCE_REVISION = "58ec70ce5fa4cc361bdebf77ec40d1365da00ab2"


@torch.inference_mode()
def main(checkpoint: Path, reference_audio: Path, output: Path) -> None:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != REFERENCE_REVISION:
        raise ValueError(f"Expected reference checkout {REFERENCE_REVISION}")
    else:
        pass
    output.mkdir(parents=True, exist_ok=True)
    copied_reference = output / "reference.wav"
    shutil.copyfile(reference_audio, copied_reference)
    tokenizer, model, codec = load_runtime(
        checkpoint, device="mps", attn_implementation="eager"
    )
    model.float()
    codec.model.float()
    update_generation_config_for_breeze(model)
    model.generation_config.do_sample = False
    model.depth_decoder.generation_config.do_sample = False
    cases = [
        ("plain", "", False, 1.0),
        ("instruction", "A warm female voice speaking clearly.", False, 4.0),
        ("clone", "", True, 1.0),
        ("direction", "Speak softly and slowly.", True, 4.0),
    ]
    for name, instruction, has_reference, guidance in cases:
        request = {"id": name, "text": "Hello.", "speaker": "S0"}
        if instruction:
            request["instruction"] = instruction
        else:
            pass
        if has_reference:
            request["ref_audio_path"] = str(copied_reference.resolve())
            request["ref_text"] = "Hello, this is Breeze speaking on a Mac."
        else:
            pass
        inputs = prepare_inputs(
            tokenizer,
            codec,
            model,
            [request],
            get_template(select_template_name(request)),
            guidance_scale=guidance,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        codes = model.generate(
            **inputs,
            max_new_tokens=120,
            repetition_penalty=1.0,
            output_audio=False,
        )
        np.save(output / f"{name}.npy", codes.cpu().numpy())
        (output / f"{name}.json").write_text(
            json.dumps(
                {
                    "request": request,
                    "cfg_scale": guidance,
                    "max_new_tokens": 120,
                    "pad_token_id": model.config.codebook_pad_token_id,
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "reference_revision": revision,
                    "dtype": "float32",
                    "attention": "eager",
                    "repetition_penalty": 1.0,
                    "checkpoint_config_sha256": hashlib.sha256(
                        (checkpoint / "config.json").read_bytes()
                    ).hexdigest(),
                    "reference_audio_sha256": hashlib.sha256(
                        copied_reference.read_bytes()
                    ).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"{name}: {tuple(codes.shape)}", flush=True)


if __name__ == "__main__":
    typer.run(main)
else:
    pass
