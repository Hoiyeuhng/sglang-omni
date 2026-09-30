# Breeze TTS 2 on Apple Silicon

Breeze TTS 2 can generate English and Chinese speech on the Apple GPU through
Omni's `/v1/audio/speech` API. The implementation supports voice design,
reference-audio cloning, reference-guided voice direction, seeded sampling and
incremental PCM streaming. Requests run serially; concurrent clients queue.
There is no claim of continuous batching or guaranteed real-time playback.

## Install and launch

Use the repository's [Apple environment setup](qwen3_asr.md#apple-silicon-mlx) with
Python 3.12. The native model uses Omni's pinned Torch and Transformers versions.
Install the codec package without replacing those dependencies:

```bash
uv pip install --python .venv-apple/bin/python --no-deps qwen-tts==0.1.1

PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-apple/bin/python -m sglang_omni.cli serve \
  --config examples/configs/breeze_tts_apple.yaml \
  --host 127.0.0.1 --port 8000 --model-name breeze-tts-2
```

The config pins the checkpoint revision. It is public and ungated, so anonymous
Hugging Face downloads work; no token needs to be added to the code or PR. A full
checkpoint takes several GB. For an existing local download, add
`--model-path /path/to/breeze-checkpoint`. Model files remain outside Git.

The tested machine is an M5 Pro with 48 GB unified memory and macOS 26.6. A
minimum-memory configuration has not been established. The model uses BF16 and
the codec uses FP32 on MPS. CUDA, FlashAttention and an NVIDIA GPU are not required.
The shared launcher may report zero configured GPUs because this stage does not
use CUDA placement; its factory explicitly loads the model onto MPS.

For compressed reference audio, follow the linked Apple instructions for
`ffmpeg@7` and `DYLD_LIBRARY_PATH`. WAV reference decoding can use the existing
SoundFile fallback. Optional SoX and FlashAttention import warnings from the
codec package do not prevent the validated WAV workflow.

## Generate speech

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "breeze-tts-2",
    "input": "Hello, this is Breeze speaking on a Mac.",
    "instructions": "A warm female voice speaking clearly.",
    "cfg_scale": 4,
    "seed": 42,
    "response_format": "wav"
  }' --output breeze.wav
```

For Chinese, use Chinese text and voice instructions. To stream, set
`"stream": true` and `"response_format": "pcm"`; audio is mono, 24 kHz signed
16-bit little-endian PCM. The default codec chunk contains two frames (160 ms of
output audio), and a final partial chunk is retained. Change the server chunk
size with `--tts.factory.chunk_frames 4`.

For voice cloning, supply `ref_audio` (an audio data URI or a reference accepted
by the server's media policy) and its exact `ref_text`. Add `instructions` for
voice direction. Use your own recording, a consented recording, or a synthetic
reference you may use. Local file references require the standard
`--allowed-local-media-path` server setting. Uploaded voices use the shared Omni
voice API. There are no built-in named speaker presets.

Generation defaults are temperature 0.9, top-k 50, top-p 1, repetition penalty
1.1, CFG scale 1 and at most 750 audio frames. Explicit API sampling values
replace these defaults; implicit defaults belonging to other models do not.
Temperature zero selects greedy decoding. A seed controls a request-local CPU
sampling generator; model computation remains on MPS. This does not promise
bitwise equality across devices, dependency versions, or the CUDA reference.

The 2,048-position checkpoint context includes text and reference frames. An
oversized prompt is rejected instead of silently truncated; generation is
bounded by the remaining space. `speed` must be 1; use instructions to request
pace changes. Supported language hints are English, Chinese and auto.

## Implementation and tests

```text
OpenAI speech request
  -> validated Breeze request
  -> independently encoded T5Gemma2 text segments + reference codec frames
  -> Qwen3 backbone: first codebook / EOS
       -> Llama depth decoder: remaining 15 codebooks
       -> complete frame feeds the next backbone step
  -> shared Qwen3-TTS incremental codec
  -> streamed PCM or accumulated WAV
```

Each request owns its guidance-branch KV caches, depth cache, RNG and codec
state. The scheduler reuses the common inbox/outbox lifecycle and checks a
cancellation event between depth steps. Generator cleanup releases state when a
client disconnects, generation fails, or the server shuts down. The codec and
its existing Transformers compatibility adapter are shared with Qwen3-TTS;
Breeze does not add another global patch implementation.

Run the tests without downloading weights:

```bash
.venv-apple/bin/python -m pytest tests/unit_test/breeze_tts \
  tests/unit_test/qwen3_tts/test_compat.py \
  tests/unit_test/qwen3_tts/test_incremental_codec.py -q
```

The model tests instantiate small real Transformer modules. Scheduler tests use
one controlled audio producer at the model boundary, with bounded waits and
cleanup; they do not patch the scheduler, queues, cancellation or global imports.

After starting the real server, run:

```bash
BREEZE_TEST_BASE_URL=http://127.0.0.1:8000 .venv-apple/bin/python \
  -m pytest tests/test_model/test_breeze_tts_apple.py -q
```

The opt-in suite covers bilingual WAV/streamed PCM equality, cloning and voice
direction using a generated synthetic reference, client errors and disconnect
recovery. It is skipped when the server URL is absent. It does not represent
upstream GPU CI or a full-corpus quality benchmark.

Initial real-weight checks on the tested Mac produced the English sentence
above and “今天天气很好，我们一起去公园散步吧。” Independent Whisper-tiny transcription
recovered both texts (Chinese used traditional characters). This is a small
intelligibility check, not human listening, speaker-similarity evaluation or a
quality ranking. Cold-start timings are not presented as warmed performance.

A separate three-frame FP32 greedy check matched all 48 codec tokens from the
pinned official reference on the same Mac. The BF16 paths were not bit-exact;
they use different attention/head arithmetic and dependency versions. This
small FP32 comparison supports the component mapping, not full-corpus numerical
or perceptual equivalence.

## Reference and license

The inference layout follows the
[official reference](https://github.com/breezeblue-ai/breeze-tts/tree/58ec70ce5fa4cc361bdebf77ec40d1365da00ab2)
and the component mapping in
[the CUDA integration PR](https://github.com/sgl-project/sglang-omni/pull/1974).
The checkpoint is pinned to `3e28c5151381a722f1d8661b4118c298caa77aa4`.
The unused legacy Mimi weights are not instantiated; the released runtime uses
the bundled Qwen3-TTS tokenizer instead.

Reference source code and Qwen3-TTS tokenizer code are Apache-2.0. Breeze weights,
converted weights and self-hosted outputs have separate research/non-commercial
terms; see the [model license](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/3e28c5151381a722f1d8661b4118c298caa77aa4/LICENSE).
No model weights or generated audio are included in the PR.
