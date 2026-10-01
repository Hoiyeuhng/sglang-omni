# Breeze TTS 2

The integration described in this guide currently supports only Apple Silicon
with the MPS backend.

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
CFG values other than 1 require voice instructions; requests without instructions
are rejected instead of silently ignoring guidance. Temperature zero selects greedy decoding. A seed controls a request-local CPU
sampling generator; model computation remains on MPS. This does not promise
bitwise equality across devices, dependency versions, or the CUDA reference.

The 2,048-position checkpoint context includes text and reference frames. An
oversized prompt is rejected instead of silently truncated; generation is
bounded by the remaining space. `speed` must be 1; use instructions to request
pace changes. Supported language hints are `English` / `en`, `Chinese` / `zh`
and `auto`.

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
client disconnects, generation fails, or the server shuts down. Cancellation is
checked between prompt-preparation operations and generation steps; a currently
running encoder or GPU operation is allowed to return before cancellation takes effect. The codec and
its existing Transformers compatibility adapter are shared with Qwen3-TTS;
Breeze does not add another global patch implementation.

Run the tests without downloading weights:

```bash
.venv-apple/bin/python -m pytest tests/unit_test/breeze_tts \
  tests/unit_test/audio/test_qwen3_tts_compat.py \
  tests/unit_test/audio/test_qwen3_tts_codec.py -q
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

## Reproduce reference parity

The reference exporter runs in a separate environment because the pinned reference
uses Torch 2.9.1 and Transformers 4.57.3. The native implementation uses the project
pins (Torch 2.13.0 and Transformers 5.12.1). Both use MPS with CPU fallback disabled,
FP32 and greedy decoding. The reference uses eager attention; native uses SDPA.
The acceptance criterion is exact equality of every generated codec token and the
same termination behavior, with no numerical tolerance on token IDs.

The reference's legacy non-streaming generation raises a tensor-rank error when
its standard repetition-penalty processor receives multi-codebook history. The
comparison explicitly sets repetition penalty to 1 in both implementations.
Production keeps its default 1.1, covered by the sampler unit tests; this comparison
does not claim default stochastic sampling or CUDA fast-runtime parity.

From the repository root, prepare the isolated reference environment once:

```bash
breeze_workspace="$PWD"
git clone https://github.com/breezeblue-ai/breeze-tts.git .venv/breeze-reference
git -C .venv/breeze-reference checkout 58ec70ce5fa4cc361bdebf77ec40d1365da00ab2
uv venv --python 3.12 .venv/breeze-tts
uv pip install --python .venv/breeze-tts/bin/python \
  -r .venv/breeze-reference/requirements.txt typer
```

Reuse an existing checkout/environment when present. Generate the English WAV
from the earlier API example as `breeze.wav`; its transcript must be exactly
"Hello, this is Breeze speaking on a Mac." Then export reference results:

```bash
breeze_workspace="$PWD"
(
  cd .venv/breeze-reference
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
    "$breeze_workspace/.venv/breeze-tts/bin/python" \
    "$breeze_workspace/scripts/apple/breeze_reference.py" \
    "$breeze_workspace/.venv/breeze-checkpoint" \
    "$breeze_workspace/breeze.wav" \
    "$breeze_workspace/results/breeze-tts/parity"
)

PYTORCH_ENABLE_MPS_FALLBACK=0 \
BREEZE_CHECKPOINT=.venv/breeze-checkpoint \
BREEZE_REFERENCE_DIRECTORY=results/breeze-tts/parity \
  .venv-apple/bin/python -m pytest tests/test_model/test_breeze_tts_reference.py -q
```

Replace the checkpoint path with your existing local model directory. The exporter
checks the reference Git revision and records dependency versions, generation
settings, checkpoint-config and reference-audio SHA-256 digests. It copies the
reference audio beside the token arrays; keep these artifacts outside Git.

Observed on the tested Mac with the generated English reference:

| Case (target: Hello.) | CFG | Audio frames | Termination |
| --- | --- | --- | --- |
| Plain | 1 | 120 | Frame limit in both implementations |
| Voice instruction | 4 | 7 | EOS in both implementations |
| Reference cloning | 1 | 10 | EOS in both implementations |
| Reference and instruction | 4 | 10 | EOS in both implementations |

All 2,352 audio tokens matched exactly. EOS rows in the reference are stored as
padding and are checked separately, not passed to the codec. The plain greedy
case does not reach EOS within this limit; it is a bounded-generation check, not
an example of successful natural termination. These four cases do not measure
speaker similarity or establish BF16/stochastic equivalence across versions.

## Accuracy Test

The paired BF16 run tested Omni commit
`62c99c8c7d080b0d592adf8d5d7892e74dec466c` against reference commit
`58ec70ce5fa4cc361bdebf77ec40d1365da00ab2`, using checkpoint
`3e28c5151381a722f1d8661b4118c298caa77aa4` on the M5 Pro described above.
Both implementations generated and scored all 50 English and 50 Chinese inputs.
Every measured generation reached EOS; there were no failures, timeouts,
frame-limit terminations or context-limit terminations.

| Runtime | English WER (errors / words) | Chinese CER (errors / characters) |
| --- | --- | --- |
| Omni | 1.95% (11 / 564) | 1.72% (16 / 931) |
| Reference | 1.06% (6 / 564) | 2.26% (21 / 931) |

These are corpus error rates from the same Whisper large-v3-turbo scorer and
normalization, not subjective quality scores. The scorer used Torch 2.13.0,
Transformers 5.12.1, jiwer 4.0.0, openai-whisper 20250625 and
opencc-python-reimplemented 0.1.7. No sample exceeded 50% error. Errors include
shared scoring effects: Chinese numerals transcribed as digits in `zh-034`,
English contraction expansion in `en-010`, and a source-text typo in `zh-001`.
The original targets and these errors remain in the reported totals.
Omni had five more English word errors and five fewer Chinese character errors;
this small, single-seed subset does not establish a general quality ranking.

The supplemental manifest uses the first target per language in plain
no-reference and instruction-only modes. All four requests generated to EOS
and scored successfully in each of three runs: Omni/common penalty 1.0,
Omni/default penalty 1.1, and reference/common penalty 1.0. Both English cases
had 0% WER in all runs. Chinese CER was 4.55% / 9.09% for Omni/common,
13.64% / 9.09% for Omni/default, and 9.09% / 0% for reference/common
(plain / instruction). These one-sentence checks establish execution and
transcription behavior, not whether the requested voice style was followed.

## Benchmark & Profiling

The following measurements are direct runtime calls, with one excluded warmup
per language. The two implementations ran sequentially. Values are means unless
marked p95; seconds include prompt preparation through CPU waveform availability.

| Runtime / language | Complete seconds | p95 seconds | Audio seconds | First audio seconds | RTF | Requests / compute second |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Omni / EN | 5.909 | 10.670 | 4.166 | 0.388 | 1.423 | 0.1692 |
| Reference / EN | 15.945 | 25.814 | 4.216 | 15.945 | 3.799 | 0.0627 |
| Omni / ZH | 7.865 | 11.255 | 5.782 | 0.336 | 1.361 | 0.1272 |
| Reference / ZH | 21.723 | 30.653 | 5.912 | 21.723 | 3.664 | 0.0460 |

The ratio of reference to Omni mean completion time is 2.70x for English and
2.76x for Chinese. Generated durations differ, so RTF is also reported as the
mean of each request's elapsed seconds divided by its audio seconds. Mean RTF
remains above 1. These results do not establish real-time playback.
The reference returns audio only after full decoding; its first-audio column
is completion latency, not a streaming measurement. Omni uses two-frame
incremental codec chunks. These numbers exclude HTTP transport and queueing.

| Runtime / language | Mean live MPS tensor GiB | Mean MPS driver GiB |
| --- | ---: | ---: |
| Omni / EN | 6.170 | 6.940 |
| Omni / ZH | 6.170 | 6.944 |
| Reference / EN | 7.277 | 9.532 |
| Reference / ZH | 7.277 | 10.094 |

GiB means bytes divided by 2^30. These are post-request snapshots from
`current_allocated_memory` and `driver_allocated_memory`, respectively.
Driver allocation includes allocator/runtime resources beyond live tensors.
Neither column measures peak allocation, process RSS, total system memory,
or a minimum-memory requirement.

### HTTP end-to-end measurements

The same native runtime served the first four inputs per language, twice at
each concurrency level, with a separate excluded warmup before each level.
All 48 measured requests succeeded. Each input's PCM SHA-256 was identical
across repetitions and concurrency levels. HTTP timings include server queueing;
percentiles pool the 16 measured requests at each concurrency level.

| Client concurrency | Requests / second | First audio mean / p95 seconds | Complete mean / p95 seconds | Mean RTF |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.1332 | 0.330 / 0.446 | 7.508 / 11.999 | 1.400 |
| 2 | 0.1345 | 6.841 / 11.946 | 13.979 / 21.401 | 2.929 |
| 4 | 0.1357 | 16.724 / 28.463 | 23.798 / 36.369 | 4.798 |

Throughput remains nearly flat while queueing raises latency. Client concurrency
does not enable model batching in this implementation. This eight-input HTTP
subset is separate from the 100-input direct-runtime comparison; it is not a
full-corpus load test. Raw requests, group wall times and configuration are in
the `http` result directory, with `http-comparison.json` / `.csv` beside the
direct-runtime summaries.

## Validation and Limitations

The final Breeze, shared audio, Qwen3-TTS and speech error/protocol unit suites
passed 520 tests, with 176 hardware/optional-dependency skips. The full
`pre-commit run --all-files` check passed. The real
HTTP suite passed all 10 tests, including `en` / `zh` language hints, exact
streamed PCM versus complete WAV equality, reference cloning, voice direction,
client errors and recovery after disconnect. These runs used the same native
runtime revision as the paired evaluation.

The main paired run uses the common repetition penalty 1.0 because the reference
fails its 1.1 preflight. Other settings and reproduction commands appear below.
The model remains BF16 and the codec FP32. Different Torch/Transformers versions,
attention kernels, samplers and codec execution are part of the runtime-stack
comparison; the timing ratio cannot be attributed solely to the scheduler.
Equal seeds do not imply equal stochastic token sequences. The separate FP32
greedy token comparison above is the exact-parity evidence.

The fixed 100-input subset covers reference-audio cloning. It does not establish
speaker similarity, human-rated naturalness, robustness on the full corpus,
CUDA performance or statistical significance. WER/CER includes ASR and text
normalization effects. Expanding the corpus should follow review of these
results rather than being inferred from this run.

Raw request records, scorer configuration, per-sample transcriptions and
`comparison.json` / `comparison.csv` are retained under
`results/breeze-tts/bf16-eval`. The report can be regenerated from those records.
The manifest and generation metadata record input and checkpoint hashes.
Model weights, generated audio and large evaluation assets stay outside Git.

## Reproduce the paired BF16 evaluation

The evaluation scripts use the first 50 rows of each English and Chinese split
from SeedTTS-Eval revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`, without
filtering. The manifest records source IDs, row indices, texts and reference-audio
SHA-256 hashes. Both implementations receive the same manifest. This fixed
100-request subset is not the full corpus.

The shared configuration uses BF16 model weights, an FP32 codec, temperature
0.9, top-k 50, top-p 1, seed 42 and at most 750 frames. Repetition penalty is
explicitly set to 1: the standard reference raises a tensor-rank error with
1.1 on the paired preflight inputs. This is a common-configuration comparison,
not a claim that the unmodified defaults work in both implementations. Equal
seeds do not produce identical draws in the CPU and MPS samplers.

The reference environment uses Torch 2.9.1, Transformers 4.57.3 and eager
attention, matching its entry point. Omni uses Torch 2.13.0, Transformers 5.12.1
and SDPA. This compares complete runtime stacks, including their different
dependency versions, attention implementations and codec execution.

Reuse the environments and checkpoint from the setup above. Set the FFmpeg
library path before starting either process. Run them sequentially on an idle
Apple GPU:

```bash
breeze_workspace="$PWD"
breeze_evaluation="$PWD/results/breeze-tts/bf16-eval"
breeze_checkpoint="$PWD/.venv/breeze-checkpoint"
export PYTORCH_ENABLE_MPS_FALLBACK=0
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

.venv-apple/bin/python -m scripts.apple.breeze_eval_prepare \
  "$breeze_evaluation" zhaochenyang20/seed-tts-eval-arrow \
  27f4c1adee83b5b29b7c4b375f6b976324bda308

.venv-apple/bin/python -m scripts.apple.breeze_eval_native \
  "$breeze_checkpoint" "$breeze_evaluation/manifest.json" \
  "$breeze_evaluation/native-bf16" --repetition-penalty 1

(
  cd .venv/breeze-reference
  PYTHONPATH="$breeze_workspace:$PWD" \
    "$breeze_workspace/.venv/breeze-tts/bin/python" \
    -m scripts.apple.breeze_eval_reference \
    "$breeze_checkpoint" "$breeze_evaluation/manifest.json" \
    "$breeze_evaluation/reference-bf16" --repetition-penalty 1
)
```

Use `--limit-per-language 2` and new output directories for a preflight. The
first selected input in each language is a separate, excluded warmup. Each
request has a 300-second deadline. Output records retain failures, timeouts and
frame/context-limit termination. Existing request logs are not overwritten.
The preparation script also creates `smoke.json` with no-reference and
instruction-only variants of the first target text in each language.
For the supplemental comparison, repeat both generation commands with
`smoke.json` and fresh `native-smoke-common` / `reference-smoke-common`
directories. Also run the native command with `--repetition-penalty 1.1` into
`native-smoke-default`. Score all three directories with the same scorer.
Report the two common-configuration directories together, using `smoke.json`
as the manifest; retain the default run separately because its sampling differs.

Runtime latency includes prompt preparation, autoregressive generation, codec
decoding and CPU waveform availability. It excludes model loading, weight
hashing, audio-file writes and HTTP transport. The reference decodes the complete
waveform before returning audio; its first-audio latency equals completion
latency. Native decoding emits two-frame chunks. MPS tensor and driver memory
are sampled after each request; neither counter is a peak-memory measurement.

Score both directories with the same locally downloaded Whisper large-v3-turbo
checkpoint, pinned to `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. Place its files
under `assets/whisper-large-v3-turbo` and save an adjacent `assets/asr-model.json`
containing its `model_id` and `revision`:

```bash
.venv-apple/bin/python - "$breeze_evaluation/assets" <<'PY'
import json
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

assets = Path(sys.argv[1])
assets.mkdir(parents=True, exist_ok=True)
model_id = "openai/whisper-large-v3-turbo"
revision = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
snapshot_download(model_id, revision=revision,
                  local_dir=assets / "whisper-large-v3-turbo",
                  allow_patterns=["*.json", "*.txt", "*.safetensors"])
(assets / "asr-model.json").write_text(
    json.dumps({"model_id": model_id, "revision": revision})
)
PY
```

Install the normalization dependency and run the shared scorer:

```bash
uv pip install --python .venv-apple/bin/python opencc-python-reimplemented==0.1.7
.venv-apple/bin/python -m scripts.apple.breeze_eval_score \
  "$breeze_evaluation/assets/whisper-large-v3-turbo" \
  "$breeze_evaluation/native-bf16" "$breeze_evaluation/reference-bf16"
.venv-apple/bin/python -m scripts.apple.breeze_eval_report \
  "$breeze_evaluation/manifest.json" "$breeze_evaluation" \
  "$breeze_evaluation/native-bf16" "$breeze_evaluation/reference-bf16"
```

Scoring uses the repository's English normalization and Chinese character
normalization. OpenCC converts both Chinese reference and hypothesis to
simplified characters before scoring. The scorer uses FP32 MPS inference,
greedy transcription, explicit language hints and overlapping 30-second chunks.
The summary reports corpus WER/CER, not the mean of individual error rates, and
keeps generation and scoring failure counts visible.
Latency and RTF distributions cover successful generations. Direct-runtime
throughput divides successful requests by the summed measured time of all
attempts, including failures; it excludes file-writing and between-request
overhead. Quality scores cover successfully scored outputs, alongside explicit
requested/generated/scored counts.
The reporter rejects mismatched input manifests, weight hashes, sampling
settings and ASR configurations instead of combining incompatible runs.

For HTTP measurements, stop direct inference processes and start the server
from the launch section. Run:

```bash
.venv-apple/bin/python -m scripts.apple.breeze_eval_http \
  http://127.0.0.1:8000 "$breeze_evaluation/manifest.json" \
  "$breeze_evaluation/http"
.venv-apple/bin/python -m scripts.apple.breeze_eval_report \
  "$breeze_evaluation/manifest.json" "$breeze_evaluation" \
  "$breeze_evaluation/native-bf16" "$breeze_evaluation/reference-bf16" \
  --http-directory "$breeze_evaluation/http"
```

This uses four inputs per language, two repetitions, and client concurrency
1/2/4. A separate warmup precedes each concurrency level. Measurements include
server queueing; the model still executes requests serially. Request logs record
first-audio and completion latency, duration and PCM hashes; group logs record
wall time and successful requests per second. Keep model files, reference audio
and generated waveforms outside Git. HTTP percentiles pool the measured requests
across repetitions. The report also checks PCM hash consistency across the
concurrency levels and repetitions.

## CI coverage and lifecycle checks

The existing `.github/workflows/test.yaml` unit job runs
`pytest tests/ -v -m "not benchmark and not accelerator" -x` with CUDA hidden.
It collects the Breeze CPU tests and shared `tests/unit_test/audio/` tests.
This job still uses an H100-labelled runner and is gated by the parent Omni CI;
the separate Intel CPU workflow only runs `tests/unit_test/cpu/`.
Real MPS HTTP and reference-parity tests skip unless their environment variables
are supplied. A missing `run-ci` label in a fork prevents the gated jobs from
running; local results are not a substitute for a claimed GitHub run.

The controlled-producer scheduler test requires a stream message to arrive while
the producer is still blocked, before completion. Other checks cover active
cancellation, shutdown, failure after a streamed chunk and a subsequent healthy
request. A tiny real text encoder verifies cancellation prevents the next CFG
branch from starting. HTTP tests verify streamed/complete byte equality, client
errors and disconnect recovery; network chunk counts alone are not evidence of
real-time generation. Shared codec tests live under `audio/`; Qwen-specific arena,
CUDA graph and installation tests remain under `qwen3_tts/`.

The existing shared compatibility adapter must run before importing the external
`qwen_tts` package with Transformers 5.12. The factory's explicit dynamic import
preserves that order without import-time global patching. This is a constrained
third-party compatibility exception, not a new Breeze patch implementation.

## Reference and license

The inference layout follows the
[official reference](https://github.com/breezeblue-ai/breeze-tts/tree/58ec70ce5fa4cc361bdebf77ec40d1365da00ab2).
The checkpoint is pinned to `3e28c5151381a722f1d8661b4118c298caa77aa4`.
The unused legacy Mimi weights are not instantiated; the released runtime uses
the bundled Qwen3-TTS tokenizer instead.

Reference source code and Qwen3-TTS tokenizer code are Apache-2.0. Breeze weights,
converted weights and self-hosted outputs have separate research/non-commercial
terms; see the [model license](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/3e28c5151381a722f1d8661b4118c298caa77aa4/LICENSE).
No model weights or generated audio are included in the PR.
