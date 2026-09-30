"""Compare Breeze's bundled codec on CPU and Apple MPS using real weights."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from numpy.typing import NDArray
from qwen_tts import Qwen3TTSTokenizer

MODEL_ID = "BreezeBlue/Breeze-TTS-2"
MODEL_REVISION = "3e28c5151381a722f1d8661b4118c298caa77aa4"
CODEBOOK_COUNT = 16
SAMPLE_RATE_HZ = 24000
SAMPLES_PER_FRAME = 1920


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    options = parser.parse_args()
    if options.frames <= 0:
        parser.error("--frames must be positive")
    elif os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Disable MPS fallback to expose unsupported operators")
    elif not torch.backends.mps.is_available():
        parser.error("An available Apple MPS device is required")
    else:
        pass

    if options.download:
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=options.model_path,
            allow_patterns=["audio_tokenizer/*", "LICENSE", "config.json"],
        )
    else:
        pass

    codec_path = options.model_path / "audio_tokenizer"
    if not (codec_path / "model.safetensors").is_file():
        parser.error("Missing bundled codec; use --download or a complete checkpoint")
    else:
        pass

    audio_codes = torch.zeros((options.frames, CODEBOOK_COUNT), dtype=torch.long)
    waveforms: dict[str, NDArray[np.float32]] = {}
    elapsed_seconds: dict[str, float] = {}
    for device in ("cpu", "mps"):
        codec = Qwen3TTSTokenizer.from_pretrained(
            str(codec_path), device_map=device, dtype=torch.float32
        )
        if next(codec.model.parameters()).device.type != device:
            raise RuntimeError(f"Codec was not loaded on {device}")
        else:
            pass
        start_seconds = time.perf_counter()
        decoded_audio, sample_rate_hz = codec.decode({"audio_codes": [audio_codes]})
        elapsed_seconds[device] = time.perf_counter() - start_seconds
        if sample_rate_hz != SAMPLE_RATE_HZ or len(decoded_audio) != 1:
            raise RuntimeError(f"Unexpected codec output on {device}")
        else:
            pass
        waveform = decoded_audio[0]
        if waveform.shape != (options.frames * SAMPLES_PER_FRAME,):
            raise RuntimeError(
                f"Unexpected waveform shape on {device}: {waveform.shape}"
            )
        elif not np.isfinite(waveform).all():
            raise RuntimeError(f"Non-finite waveform on {device}")
        else:
            waveforms[device] = waveform
        del codec

    absolute_error = np.abs(waveforms["cpu"] - waveforms["mps"])
    report = {
        "scope": "codec execution only; synthetic codes, not speech quality or full TTS",
        "model_id": MODEL_ID,
        "download_revision": MODEL_REVISION if options.download else None,
        "model_path": str(options.model_path.resolve()),
        "macos_version": platform.mac_ver()[0],
        "torch_version": torch.__version__,
        "dtype": "float32",
        "frames": options.frames,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "samples": int(waveforms["mps"].size),
        "cold_decode_seconds": elapsed_seconds,
        "maximum_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
    }
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
else:
    pass
