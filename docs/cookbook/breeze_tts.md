# Breeze TTS 2

**Currently supported: Apple Silicon (MPS) only.** The `/v1/audio/speech` API
supports English/Chinese synthesis, voice design, reference cloning, voice
direction, seeded sampling and incremental PCM streaming. Requests execute
serially; concurrent clients queue. This uses the English/Chinese open-weight
checkpoint; BreezeBlue's hosted Multilingual model is outside this integration.

## Install and launch

Requires macOS 14 or newer on Apple Silicon and Homebrew. From this repository's
root, create or reuse the Python 3.12 environment and install the codec:

```bash
./install.sh
source .venv-apple/bin/activate
uv pip install --python .venv-apple/bin/python --no-deps qwen-tts==0.1.1

export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-apple/bin/python -m sglang_omni.cli serve \
  --config examples/configs/breeze_tts_apple.yaml \
  --host 127.0.0.1 --port 8000 --model-name breeze-tts-2
```

The installer supplies `uv`, `ffmpeg@7`, the pinned SGLang `all_mps` dependencies
and this checkout. `--no-deps` keeps the codec from replacing those pins.
For installer options, see the shared
[Apple Silicon installation guide](../get_started/installation.md#macos-apple-silicon).

The config pins a public, ungated checkpoint (several GB; no token required).
For an existing download, add `--model-path /path/to/breeze-checkpoint`.
The model uses BF16 and the codec FP32 on MPS; CUDA, an NVIDIA GPU and
FlashAttention are not required. The launcher may report zero configured GPUs
because this stage loads MPS directly rather than using CUDA placement.

Keep the FFmpeg library export when starting the server: compressed references
need it, while WAV decoding has a SoundFile fallback. Use `ffmpeg@7` for the
pinned TorchCodec; optional SoX/FlashAttention warnings do not prevent the WAV
workflow. If a macOS launcher strips `DYLD_*`, set the variable on the final
server process.

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

| Option | Behavior |
| --- | --- |
| Language | `English` / `en`, `Chinese` / `zh`, or `auto`; use Chinese text and instructions for Chinese synthesis. |
| Cloning / direction | Supply `ref_audio` and its exact `ref_text`; add `instructions` for voice direction. Use an audio data URI or another reference accepted by the media policy. Local files require `--allowed-local-media-path`. |
| Voices | Uploaded voices use the shared Omni voice API; no built-in named speaker presets. Use your own, consented or permitted synthetic recordings. |
| Streaming | Set `stream: true`, `response_format: "pcm"`: mono 24 kHz signed 16-bit little-endian PCM. Default chunks contain two frames (160 ms); final partial chunks are retained. Override with `--tts.factory.chunk_frames 4`. |
| Sampling defaults | Temperature **0.9**, top-k **50**, top-p **1**, repetition penalty **1.1**, CFG scale **1**, maximum **750 audio frames (about 60 seconds)**. Explicit API values override these; implicit defaults from other models do not. Temperature **0** selects greedy decoding. |
| Guidance / speed | CFG other than 1 requires instructions or returns a client error. `speed` must be 1; request pace changes through instructions. |
| Unsupported controls | `stream_codec_output` and `initial_codec_chunk_frames` return client errors; configure chunk size with `--tts.factory.chunk_frames`. Non-finite reference audio is rejected. |
| Context | The **2,048 positions** include text and reference frames. Oversized prompts are rejected; generation is bounded by remaining space. Text is not automatically split; reaching the frame/context limit can truncate output. |
| Seed | Controls a request-local CPU sampler while model computation stays on MPS; it does not promise bitwise equality across devices or dependency versions. |

## Accuracy Test

### Tested configuration

| Component | Configuration |
| --- | --- |
| Hardware | M5 Pro, 48 GB unified memory, macOS 26.6; MPS CPU fallback disabled |
| Omni | `62c99c8c7d080b0d592adf8d5d7892e74dec466c`; Torch 2.13.0, Transformers 5.12.1, SDPA |
| Reference | `58ec70ce5fa4cc361bdebf77ec40d1365da00ab2`; Torch 2.9.1, Transformers 4.57.3, eager attention (as in its entry point) |
| Checkpoint | `3e28c5151381a722f1d8661b4118c298caa77aa4`; BF16 model, FP32 codec |
| Inputs | First 50 EN + 50 ZH rows, unfiltered, from SeedTTS-Eval revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`; identical texts, cloning audio and transcripts |
| Shared sampling | Temperature 0.9, top-k 50, top-p 1, seed 42, cap 750 frames; **repetition penalty 1.0** |
| Scorer | Whisper large-v3-turbo `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`; FP32 MPS, greedy transcription, explicit language, overlapping 30-second chunks |
| Normalization | Repository English/Chinese normalization; OpenCC t2s on both Chinese texts before character scoring. Torch 2.13.0, Transformers 5.12.1, jiwer 4.0.0, openai-whisper 20250625, opencc-python-reimplemented 0.1.7 |

**Sampling exception:** the reference's legacy repetition-penalty processor fails
with a tensor-rank error on multi-codebook history at its default 1.1. Paired
runs explicitly use 1.0 on both sides; serving retains 1.1. This compares a
common configuration, not unmodified defaults. Equal CPU/MPS seeds need not
produce equal random draws.

### Results

Both runtimes generated and scored all 100 inputs, all reaching EOS with **zero
failures, timeouts, frame-limit or context-limit terminations**.

| Runtime | English WER (errors / words) | Chinese CER (errors / characters) |
| --- | --- | --- |
| Omni | 1.95% (11 / 564) | 1.72% (16 / 931) |
| Reference | 1.06% (6 / 564) | 2.26% (21 / 931) |

These are corpus rates, not mean per-request rates. No sample exceeded 50%
error. Totals retain numeral-format differences (`zh-034`), contraction
expansion (`en-010`) and a source typo (`zh-001`). Omni has five more English
word errors and five fewer Chinese character errors; this single-seed subset
does not establish a general quality ranking.

Supplemental cases use the first target per language in plain no-reference and
instruction-only modes. All four cases in each run reached EOS and scored:

| Run | EN WER, plain / instruction | ZH CER, plain / instruction |
| --- | --- | --- |
| Omni, common penalty 1.0 | 0% / 0% | 4.55% / 9.09% |
| Omni, default penalty 1.1 | 0% / 0% | 13.64% / 9.09% |
| Reference, common penalty 1.0 | 0% / 0% | 9.09% / 0% |

These check execution and transcription, not adherence to voice style. Earlier
Whisper-tiny checks recovered the example English text and
“今天天气很好，我们一起去公园散步吧。” (Chinese transcription used traditional
characters); these were only small intelligibility checks, not listening tests.

### Independent FP32 parity

With the same environments, MPS fallback disabled, greedy FP32 decoding and
repetition penalty 1.0, all **2,352 codec tokens matched exactly**:

| Case (target: Hello.) | CFG | Audio frames | Termination |
| --- | --- | --- | --- |
| Plain | 1 | 120 | Frame limit in both implementations |
| Voice instruction | 4 | 7 | EOS in both implementations |
| Reference cloning | 1 | 10 | EOS in both implementations |
| Reference and instruction | 4 | 10 | EOS in both implementations |

The target is `Hello.` and the reference is the generated English example WAV.
EOS padding rows are checked separately and excluded from decoding. Plain
generation hitting the shared cap is a bounded-generation check, not successful
natural termination. This does not establish BF16/stochastic or CUDA fast-runtime
parity. [Reproduction commands](#fp32-parity).

The real FP32 codec also matches full decoding within `rtol=2e-4, atol=2e-5`
when the 120-frame reference sequence is split into 1, 2, 7 or 32-frame chunks,
including final partial chunks and sliding-window boundaries.

## Benchmark & Profiling

### Direct runtime

Runtimes ran sequentially with one excluded warmup per language. Values are
means unless marked p95. Timing includes prompt preparation, autoregressive
generation, codec decoding and CPU waveform availability; it excludes loading,
weight hashing, file writes, HTTP and queueing.

| Runtime / language | Complete seconds | p95 seconds | Audio seconds | First audio seconds | RTF | Requests / compute second |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Omni / EN | 5.909 | 10.670 | 4.166 | 0.388 | 1.423 | 0.1692 |
| Reference / EN | 15.945 | 25.814 | 4.216 | 15.945 | 3.799 | 0.0627 |
| Omni / ZH | 7.865 | 11.255 | 5.782 | 0.336 | 1.361 | 0.1272 |
| Reference / ZH | 21.723 | 30.653 | 5.912 | 21.723 | 3.664 | 0.0460 |

Reference/Omni mean completion-time ratios are **2.70x EN / 2.76x ZH**. RTF is
mean per-request elapsed/audio seconds, accounting for different generated
lengths. Mean RTF remains **above 1**, so these results do not establish real-time
playback. Reference audio arrives only after full decoding (first audio equals
completion); Omni emits two-frame chunks.

| Runtime / language | Mean live MPS tensor GiB | Mean MPS driver GiB |
| --- | ---: | ---: |
| Omni / EN | 6.170 | 6.940 |
| Omni / ZH | 6.170 | 6.944 |
| Reference / EN | 7.277 | 9.532 |
| Reference / ZH | 7.277 | 10.094 |

Memory values are **post-request snapshots**, in GiB (bytes / 2^30), from
`current_allocated_memory` and `driver_allocated_memory`. Driver allocation
includes allocator/runtime resources beyond live tensors. Neither measures
peak allocation, RSS, total system memory or minimum required memory.

### HTTP end to end

The same native runtime served the first four inputs per language, twice per
concurrency level, with one excluded warmup per level. **48/48 succeeded**;
each input's PCM SHA-256 matched across repetitions and concurrency levels.
Timing includes server queueing; p95 pools 16 measured requests per level.

| Client concurrency | Requests / second | First audio mean / p95 seconds | Complete mean / p95 seconds | Mean RTF |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.1332 | 0.330 / 0.446 | 7.508 / 11.999 | 1.400 |
| 2 | 0.1345 | 6.841 / 11.946 | 13.979 / 21.401 | 2.929 |
| 4 | 0.1357 | 16.724 / 28.463 | 23.798 / 36.369 | 4.798 |

Throughput stays nearly flat as queueing raises latency: client concurrency
is not model batching. This eight-input load test is separate from the
100-input direct-runtime evaluation.

### Length boundary test

Each runtime receives **24 fixed cloning inputs**: three stories per language at
four cumulative lengths, retaining all actual durations. Omni uses `7efa6788`
with timing instrumentation; other pins/common sampling are as above, with one
excluded warmup per language and a 600-second deadline. Both use **750 frames**
(about 60 seconds); the reference entry point normally defaults to 1,500.

All 48 outputs generated and scored, without runtime errors, timeouts or context
limits. Each runtime has **16 EOS and 8 frame-limit terminations**: six deliberate
over-limit inputs and two nominal long inputs. All capped outputs have tail
deletions in ASR; EOS outputs have no ASR deletions. Paired columns are **Omni / reference**; errors are corpus
EN WER or ZH CER.

| Language / nominal target | Actual audio seconds, range | EOS | WER / CER | Mean RTF |
| --- | --- | --- | --- | --- |
| EN / 15s | 11.7–15.4 / 12.4–15.0 | 3/3 / 3/3 | 0.00% / 0.00% | 1.41 / 3.92 |
| EN / 30s | 26.3–28.6 / 26.1–30.6 | 3/3 / 3/3 | 0.00% / 0.42% | 1.41 / 4.50 |
| EN / 45-55s | 57.0–60.0 / 48.6–52.9 | 2/3 / 3/3 | 4.34% / 0.24% | 1.39 / 3.96 |
| EN / over60s | 60.0 / 60.0 | 0/3 / 0/3 | 36.35% / 35.33% | 1.38 / 3.92 |
| ZH / 15s | 15.8–17.4 / 16.0–19.4 | 3/3 / 3/3 | 2.86% / 6.29% | 1.39 / 3.72 |
| ZH / 30s | 27.8–35.1 / 24.0–41.2 | 3/3 / 3/3 | 2.47% / 2.78% | 1.40 / 3.87 |
| ZH / 45-55s | 48.9–60.0 / 56.9–60.0 | 2/3 / 1/3 | 6.57% / 17.05% | 1.42 / 3.73 |
| ZH / over60s | 60.0 / 60.0 | 0/3 / 0/3 | 54.74% / 50.77% | 1.43 / 3.72 |

Scoring uses the pinned Whisper model's **native long-form transcription with
timestamps**. All 48 waveforms were rescored after overlapping-chunk transcription
omitted middle text and hallucinated repetition; initial results remain in
`scores-chunked.jsonl`. Chinese orthographic/name differences are retained.
This small ASR-scored subset does not establish a general quality ranking.

Starting playback one second after the first chunk produces **21.3–24.8 seconds
of mean simulated stalls** in Omni's long/over-limit groups. Maximum chunk gap:
**0.77 seconds**; maximum required startup delay: **26.42 seconds**. These traces
exclude HTTP/audio-device buffering. The report retains final-20%-of-text
deletions as a diagnostic; neither EOS nor that metric guarantees completeness.

Maximum post-request MPS driver snapshots are **27.56 / 17.25 GiB**; live tensors
remain about **6.17 / 7.28 GiB**. These are not peak or minimum-memory measurements.

At serving-default repetition penalty **1.1**, all **18/18 HTTP requests**
completed and all post-request health checks returned 200. Each of six 60-second
capped requests was bracketed by identical short requests; all **6/6 recovery
responses matched the baseline PCM SHA-256**. Client chunk timings are retained
in `http-recovery.jsonl`.

## Validation and Limitations

| Check | Result / scope |
| --- | --- |
| Related unit suites | **1,108 passed, 176 skipped**, with one existing NVIDIA launcher failure also reproduced on unchanged `main` (`nvidia-smi` unavailable on macOS); Breeze, shared audio, Qwen3-TTS and the full `serve/` suite |
| Real HTTP suite | **14 passed**: `en`/`zh`, exact streamed PCM/complete WAV equality, cloning, direction with a generated reference, unsupported controls, non-finite reference rejection and disconnect recovery |
| Real checkpoint suite | **8 passed**: four FP32 token-parity cases and four full/incremental codec comparisons |
| Repository checks | Full `pre-commit run --all-files` passed |
| Interpretation | Different dependency versions, attention kernels, samplers and codec execution make this a runtime-stack comparison, not a scheduler-only speedup. |
| Coverage limits | No full-corpus, speaker-similarity, human-rated naturalness, statistical-significance, CUDA performance or minimum-memory claim. ASR/normalization affect WER/CER; cold-start timings are not presented as warmed performance. Review this subset before expanding it. |

Run unit checks without downloading weights:

```bash
.venv-apple/bin/python -m pytest tests/unit_test/breeze_tts \
  tests/unit_test/audio/test_qwen3_tts_compat.py \
  tests/unit_test/audio/test_qwen3_tts_codec.py -q
```

Run the real-server suite after launch (skipped without `BREEZE_TEST_BASE_URL`):

```bash
BREEZE_TEST_BASE_URL=http://127.0.0.1:8000 .venv-apple/bin/python \
  -m pytest tests/test_model/test_breeze_tts_apple.py -q
```

The existing `.github/workflows/test.yaml` unit job collects Breeze and shared
`audio/` tests with `pytest tests/ -v -m "not benchmark and not accelerator" -x`
and CUDA hidden. It still uses an H100-labelled runner under the parent Omni CI;
the separate Intel CPU workflow runs only `tests/unit_test/cpu/`. MPS HTTP and
parity tests need their environment variables. Missing `run-ci` labels prevent
gated fork jobs from executing; local counts do not imply hosted CI passed.

## Reproduce the evaluations

### Reference environment

Reuse existing environments/checkpoints where available; otherwise prepare the
isolated reference environment from the repository root:

```bash
breeze_workspace="$PWD"
git clone https://github.com/breezeblue-ai/breeze-tts.git .venv/breeze-reference
git -C .venv/breeze-reference checkout 58ec70ce5fa4cc361bdebf77ec40d1365da00ab2
uv venv --python 3.12 .venv/breeze-tts
uv pip install --python .venv/breeze-tts/bin/python \
  -r .venv/breeze-reference/requirements.txt typer
```

### FP32 parity

Create `breeze.wav` with the API example above, with exact transcript
`Hello, this is Breeze speaking on a Mac.`, then run:

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

Replace the local checkpoint path as needed. The exporter verifies the reference
revision and saves versions, settings, checkpoint-config/reference-audio hashes
and a copy of the reference alongside token arrays. Acceptance requires exact
token equality and matching termination, with no token-ID tolerance.

### Paired BF16 generation

Run sequentially on an idle Apple GPU using the configuration above:

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

Preflight with `--limit-per-language 2` and fresh output directories. Each run
adds one excluded warmup per language and a **300-second request deadline**.
Logs retain failures, timeouts and frame/context-limit termination and refuse
to overwrite existing request records.

For supplemental cases, replace `manifest.json` with generated `smoke.json` and
use fresh `native-smoke-common` / `reference-smoke-common` directories. Run
native again with `--repetition-penalty 1.1` into `native-smoke-default`. Score
all three; report only the two common runs together with `smoke.json`, keeping
the differently configured default run separate.

### Shared scoring and report

Download the pinned scorer and write its adjacent model metadata:

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

Install normalization support, score both outputs and aggregate:

```bash
uv pip install --python .venv-apple/bin/python opencc-python-reimplemented==0.1.7
.venv-apple/bin/python -m scripts.apple.breeze_eval_score \
  "$breeze_evaluation/assets/whisper-large-v3-turbo" \
  "$breeze_evaluation/native-bf16" "$breeze_evaluation/reference-bf16"
.venv-apple/bin/python -m scripts.apple.breeze_eval_report \
  "$breeze_evaluation/manifest.json" "$breeze_evaluation" \
  "$breeze_evaluation/native-bf16" "$breeze_evaluation/reference-bf16"
```

### HTTP benchmark

Stop direct inference, launch the server, then run:

```bash
.venv-apple/bin/python -m scripts.apple.breeze_eval_http \
  http://127.0.0.1:8000 "$breeze_evaluation/manifest.json" \
  "$breeze_evaluation/http"
.venv-apple/bin/python -m scripts.apple.breeze_eval_report \
  "$breeze_evaluation/manifest.json" "$breeze_evaluation" \
  "$breeze_evaluation/native-bf16" "$breeze_evaluation/reference-bf16" \
  --http-directory "$breeze_evaluation/http"
```

### Length and recovery probes

Prepare the length inputs from the frozen manifest, using the variables above:

```bash
breeze_length="$PWD/results/breeze-tts/length-eval"
.venv-apple/bin/python -m scripts.apple.breeze_eval_length prepare \
  "$breeze_evaluation/manifest.json" \
  scripts/apple/fixtures/breeze_length_passages.json "$breeze_length"
```

Repeat the paired generation commands with `$breeze_length/manifest.json` and
fresh `$breeze_length/native` / `$breeze_length/reference` outputs. Set
`--repetition-penalty 1 --max-new-tokens 750 --timeout-seconds 600` on both.
Score with Whisper's native long-form transcription and summarize:

```bash
.venv-apple/bin/python -m scripts.apple.breeze_eval_score \
  "$breeze_evaluation/assets/whisper-large-v3-turbo" \
  "$breeze_length/native" "$breeze_length/reference" --long-form
.venv-apple/bin/python -m scripts.apple.breeze_eval_length report \
  "$breeze_length/native" "$breeze_length/reference" \
  --output "$breeze_length/summary.json"
```

After direct inference finishes, launch the server and run the six
baseline/over-limit/recovery groups at serving-default repetition penalty **1.1**:

```bash
.venv-apple/bin/python -m scripts.apple.breeze_eval_length http \
  "$breeze_length/manifest.json" http://127.0.0.1:8000 \
  "$breeze_length/http-recovery.jsonl"
```

### Artifacts and metric definitions

Results stay under `results/breeze-tts/bf16-eval` or `length-eval`; keep weights, reference
audio, generated waveforms and large assets outside Git.

| Artifact / statistic | Contents / definition |
| --- | --- |
| Manifest / generation metadata | Source IDs, row indices, texts, reference-audio hashes, input/checkpoint hashes and configuration |
| Request / score logs | Per-request timing, termination, waveform metadata, scorer versions and transcriptions; explicit requested/generated/scored counts |
| `comparison.json` / `.csv` | Recomputable direct-runtime summaries; latency/RTF cover successful generations, quality covers scored outputs |
| Direct throughput | Successful requests / summed measured time of **all attempts**, including failures; excludes writes and between-request overhead |
| `http/` | Request first-audio/completion latency, duration and PCM hashes; group wall times, successful requests/second and configuration |
| `http-comparison.json` / `.csv` | HTTP summaries beside direct summaries; percentiles pool repetitions; verifies PCM hash consistency |
| Report integrity | Rejects mismatched manifests, weights, sampling or scorer configurations |

## Implementation notes

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

Each request owns guidance-branch KV caches, depth cache, RNG and codec state.
The scheduler reuses the shared inbox/outbox lifecycle. Cancellation is checked
between prompt-preparation operations, generation and depth steps; in-flight
encoder/GPU operations finish first. Generator cleanup releases state on
disconnect, failure or shutdown.

The shared Qwen3-TTS codec and Transformers compatibility adapter are reused.
The adapter must run before importing external `qwen_tts` with Transformers
5.12; the factory's explicit dynamic import preserves that order without
import-time global patching or a new Breeze patch implementation. The released
runtime uses the bundled Qwen3-TTS tokenizer, not the unused legacy Mimi weights.

Model tests use small real Transformers. Scheduler tests use a bounded,
controlled audio producer without patching scheduler/queues/cancellation/global
imports, and verify a chunk arrives before the blocked producer completes.
They cover active cancellation, shutdown, failure after a chunk and subsequent
recovery; a tiny text encoder checks cancellation before the next CFG branch.
HTTP chunk counts alone do not prove real-time generation. Shared codec tests
live in `audio/`; Qwen-specific arena, CUDA graph and installation tests remain
in `qwen3_tts/`.

## Reference and license

The layout follows the [pinned official reference](https://github.com/breezeblue-ai/breeze-tts/tree/58ec70ce5fa4cc361bdebf77ec40d1365da00ab2).
Reference and Qwen3-TTS tokenizer code are Apache-2.0. Breeze weights, converted
weights and self-hosted outputs have separate research/non-commercial terms;
see the [model license](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/3e28c5151381a722f1d8661b4118c298caa77aa4/LICENSE).
No weights or generated audio are included in the PR.
