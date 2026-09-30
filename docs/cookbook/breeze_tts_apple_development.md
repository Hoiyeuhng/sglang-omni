# Breeze TTS 2: local Apple development

This is a development starting point, not an Omni model integration or a claim
that Breeze text-to-speech works on macOS. The first executable probe checks the
released audio codec on CPU and MPS. It does not generate speech from text.

## Isolated environment

Run these commands from the Omni repository root. They require Git, uv, an Apple
Silicon Mac with MPS support, and network access. Keep the reference dependencies
separate from Omni: the reference pins Transformers 4.57.3 and Torch 2.9.1, while
Omni uses a different stack.

```bash
mkdir -p .venv
git clone https://github.com/breezeblue-ai/breeze-tts.git .venv/breeze-reference
git -C .venv/breeze-reference checkout 58ec70ce5fa4cc361bdebf77ec40d1365da00ab2
uv venv .venv/breeze-tts --python 3.12
uv pip install --python .venv/breeze-tts/bin/python \
  -r .venv/breeze-reference/requirements.txt
```

The commands create local, ignored directories. They do not change Omni's
dependency pins. The reference requirements pin the main inference libraries;
their transitive dependencies are not a complete lockfile.

## Real-weight codec probe

The model was public and ungated when checked on September 30, 2026. A Hugging
Face token is not required for anonymous downloads. If authentication is needed
later, use the local Hugging Face login; never put a token in source or a PR.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/breeze-tts/bin/python \
  scripts/apple/breeze_tts_codec_probe.py \
  --model-path .venv/breeze-checkpoint --download \
  --report results/breeze-tts/codec.json
```

This downloads only the bundled codec, model configuration and license at
checkpoint revision 3e28c5151381a722f1d8661b4118c298caa77aa4. Omit --download
to reuse those files offline. It decodes the same four synthetic codec frames
with the real FP32 weights on CPU and MPS, checks placement, output dimensions
and finite samples, and reports numerical differences. No reference voice or
voice-cloning consent is needed for these synthetic codes.

Timing includes the first decode and is not a warmed benchmark. Numerical error
is reported without a quality threshold; success does not establish perceptual
equivalence. Unsupported MPS operators fail visibly with CPU fallback disabled.

Initial verification on an M5 Pro with 48 GB unified memory, macOS 26.6 and
Torch 2.9.1 decoded four frames into 7,680 finite samples on both devices. The
maximum CPU/MPS absolute difference was 2.92e-6. This is a synthetic-code codec
check, not a speech-quality measurement. Qwen's optional FlashAttention and SoX
import warnings did not prevent this decode-only check.

## Remaining model work

The pinned reference FastBreezeStreamingRuntime constructor requires CUDA even
when all fast flags are disabled. Its generation path also contains CUDA event
timing and seeding. Changing the loader's device alone is insufficient.

1. Establish eager text-encoder, backbone and depth-decoder execution on MPS.
2. Preserve the complete depth-decoder frame in the autoregressive feedback loop.
3. Generate real English and Chinese speech and compare with the reference.
4. Add Omni stage integration, cancellation and streaming tests once generation works.
5. Measure warmed latency, memory and quality on the actual Mac.

The existing CUDA integration is tracked in
[upstream PR #1974](https://github.com/sgl-project/sglang-omni/pull/1974).
This development branch does not import its CUDA scheduler or claim continuous
batching, voice cloning, streaming, or /v1/audio/speech support.

## License

Reference source code and the Qwen3-TTS tokenizer code are Apache-2.0. Breeze
weights, converted weights and self-hosted outputs have separate research and
non-commercial terms. See the
[model license](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/3e28c5151381a722f1d8661b4118c298caa77aa4/LICENSE).
Local weights and generated artifacts stay out of the Git repository.
