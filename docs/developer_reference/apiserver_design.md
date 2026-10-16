# API Server Design

This page explains the API server at the level that is most useful for maintenance: where it sits in the system, which files matter, and how requests are mapped into the runtime.

If you only want to launch the server and call it, start with [API Server Quickstart](../get_started/apiserver_quickstart.md).

## Role in the System

The API server is the outer protocol layer on top of the `sglang-omni` pipeline runtime.

At a high level, the built-in server startup path is:

`CLI / Python entrypoint` → `PipelineConfig` → `Pipeline Startup` → `Coordinator` → `Client` → `FastAPI`

After startup, the request path is:

`HTTP request` → `FastAPI route` → `Client` → `Coordinator` → `Stage pipeline` → `Client aggregation` → `HTTP/SSE response`

That split keeps responsibilities clean:

- the pipeline runtime handles orchestration and execution
- the `Client` layer submits requests and assembles results
- the API server translates between HTTP/OpenAI-style payloads and those internal abstractions

## Key Files

For the current server implementation, these are the files that matter most.

| File | Role |
| --- | --- |
| `sglang_omni/serve/openai_api.py` | Defines the FastAPI app, routes, request conversion, and response formatting |
| `sglang_omni/serve/protocol.py` | Defines request and response schemas |
| `sglang_omni/serve/launcher.py` | Compiles the pipeline, starts the runtime, mounts the app, and runs Uvicorn |
| `sglang_omni/client/client.py` | Submits requests to the coordinator and aggregates text, audio, and stream results |
| `sglang_omni/cli/serve.py` | Defines the current CLI surface for `sgl-omni serve` |

If you are tracing endpoint behavior, `openai_api.py` and `client.py` are usually the best places to start.

## `create_app()` vs `launch_server()`

This is the most important distinction in the serving code.

### `create_app(client, model_name=...)`

`create_app()` only builds the FastAPI app and registers the core routes.

It does **not**:

- compile the pipeline
- start the runtime
- create the coordinator
- mount profiling routes
- run Uvicorn

Use it when you already have a live `Client` and want to embed the HTTP layer yourself.

### `launch_server(pipeline_config, ...)`

`launch_server()` is the full built-in server lifecycle.

It:

- compiles the pipeline config
- starts the pipeline runtime
- creates the `Client`
- creates the FastAPI app
- mounts profiling routes on the single-process path
- runs Uvicorn
- stops the runtime on shutdown

Use it when you want the standard out-of-the-box server path.

### Timeouts for shared sessions

A shared session has an admission timeout and an input-idle timeout. The admission timeout sets a time limit to start session admission. The input-idle timeout sets a time limit without accepted input.

| Timeout | Runtime limit | Default |
| --- | --- | --- |
| Admission timeout | `session_update_timeout_s` | 60 seconds |
| Input-idle timeout | `idle_input_timeout_s` | 60 seconds |

To use these options, set `--enable-realtime`. The pipeline must have `realtime_deployment_factory`. `MiniCPMODuplexPipelineConfig` has this factory.

| Server option | Python argument | Example value |
| --- | --- | --- |
| `--realtime-admission-timeout-s` | `realtime_admission_timeout_s` | `20` seconds |
| `--realtime-idle-input-timeout-s` | `realtime_idle_input_timeout_s` | `300` seconds |

Set each timeout to a finite number greater than zero. If you do not set an option, the server uses the deployment limit. Give the Python arguments to `launch_server` or `run_server`. Do not put these options in the pipeline YAML.

The admission timer starts when the connection starts. An admission error does not start a new timer. After the deadline, a new `session.update` cannot start admission.

An admission operation can complete after the deadline. The timer waits for this operation and its cleanup. Each operation uses `cleanup_timeout_s` as its time limit.

A new input-idle period starts when the session opens. A new period also starts after each of these events:

- The runtime accepts audio input or an image.
- An input clear or input end command completes.
- The adapter completes an input unit.

The input-idle timeout cannot close a session while the adapter processes an input unit. A command that holds the command lock also prevents this closure. WebSocket Pong frames and `session.update` messages do not start a new input-idle period. Rejected images do not start a new period.

The runtime does an input activity check each input-idle period. An input timeout can occur after approximately two periods. Cleanup and the event loop can add more time.

`CoordinatorAdapter` also sets a pipeline idle timeout. This timeout gives a second time limit. Its value is the maximum of these values:

- The default pipeline idle timeout of 300 seconds.
- The pipeline idle timeout in the configuration.
- Two input-idle periods, plus the runtime cleanup timeout and the pipeline operation timeout.

The pipeline receives full input units. The runtime can accept audio before an input unit is full. Audio, images, and clear commands can keep the runtime active without a new pipeline input unit. The pipeline idle timeout can close that session.

Other pipeline sessions keep the idle timeout from their configuration. This change does not change operation timeouts or cleanup timeouts.

### Release note: input-idle timeout

Shared sessions have a default input-idle timeout of 60 seconds. Before this change, the runtime did not have this timeout. If a client sends no input during a long pause, the server can close its connection. WebSocket Pong frames do not prevent this closure.

To let a pause continue for more time, increase `--realtime-idle-input-timeout-s`. For example, use `--realtime-idle-input-timeout-s 300` for a timeout of five minutes.

### Transport timeouts for shared sessions

The transport timeout sets a time limit for WebSocket write and close operations. `RuntimeLimits.transport_timeout_s` has a default value of 30 seconds. Set this deployment limit to a finite number greater than zero. The `/v1/realtime/capabilities` response gives this limit.

Each text write has a timeout. If a write does not complete in time, the runtime starts cleanup with the reason `transport_timeout`. If a close task controls cleanup, that task keeps its first reason.

After native cleanup, the session waits for the last JSON events and the Close frame. This wait has a limit of one transport timeout period. The session then cancels local tasks that did not stop. Each of the two cancellation rounds has a limit of one transport timeout period.

If local tasks stop after cancellation, the connection slot is available in three transport timeout periods or less after native cleanup. The event loop can add more time. If a task ignores the two cancellation rounds, the connection reports a cleanup error. That task can stay active.

The connection keeps its first `CancelledError`, including the message. More cancellation requests do not replace this error. The connection task waits for local cleanup before it stops with this error. If local cleanup has an error, the cleanup error becomes the cause of the cancellation error.

A different task controls native cleanup. A transport timeout does not cancel that task. Native cleanup keeps its time limits. After a transport timeout, the last JSON events or the Close frame might not get to the peer.

The ASGI backend controls TCP shutdown. The Close handshake timer can continue after the connection slot is available.

### WebSocket close codes for shared sessions

A shared session sends the last JSON events before the WebSocket Close frame. The frame gives a close code and a close reason.

| Close code | Close reason |
| --- | --- |
| `1000` | `client_closed` or `disconnect` |
| `1008` | `session_update_timeout`, `idle_timeout`, or `context_exhausted` |
| `1011` | An internal error, adapter failure, cleanup failure, or transport timeout |

The server uses code `1008` for an admission timeout, an input-idle timeout, or the context limit. These conditions are application policies. [RFC 6455, section 7.4.1](https://www.rfc-editor.org/rfc/rfc6455.html#section-7.4.1) gives the WebSocket close codes.

The Close frame has a limit of 123 bytes for the close reason. The server uses UTF-8 for this text. If the text is too long, the server removes whole characters from the end. The JSON events keep the full reason.

The first close request sets the reason. A cleanup failure changes the reason to `cleanup_timeout`. An unknown reason uses code `1011`. A missing reason uses code `1011` and the reason `internal`.

A request error closes the session only if it is a fatal error.

## Route Surface

The current server exposes these main routes:

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Health status from `client.health()` |
| `GET` | `/v1/models` | Single-model listing for the active pipeline |
| `POST` | `/v1/chat/completions` | Chat completions, including streaming and optional audio |
| `POST` | `/v1/audio/speech` | Text-to-speech, raw audio response or raw PCM chunks when `stream=true` |
| `POST` | `/start_profile` | Torch trace + (optional) request-level events. Added by the built-in launcher |
| `POST` | `/stop_profile` | Stops both torch trace and request-level events |
| `POST` | `/start_request_profile` | Request-level event recorder only (no torch trace) |
| `POST` | `/stop_request_profile` | Stops the request-level event recorder |

The profiling routes are mounted by the single-process `launch_server()` path. The
current multi-process launcher path does not mount them.

`/start_profile` accepts:

```jsonc
{
  "run_id": "demo-run",
  "trace_path_template": "/tmp/profiles/demo-run/trace",  // torch trace template
  "event_dir": "/tmp/profiles/demo-run/events",            // request-event JSONL dir (optional)
  "enable_torch": true                                     // set false to skip torch trace
}
```

`/stop_profile` and `/stop_request_profile` both accept an optional
`run_id`. Omitting it is a wildcard: every stage stops whatever profiler
session is currently active.

Request-level events are emitted as JSON lines under
`<event_dir>/events_<stage>_<pid>.jsonl`. Use `python -m sglang_omni.profiler
<event_dir>` to derive the timeline / stage / hop reports described in
`docs/developer_reference/profiler.md`.

## Request Mapping

The server does not pass OpenAI-style request bodies straight into the runtime. It first converts them into internal request objects.

### Chat requests

`ChatCompletionRequest` includes standard OpenAI-style fields such as:

- `model`
- `messages`
- `temperature`
- `top_p`
- `max_tokens`
- `stop`
- `seed`
- `stream`

It also includes `sglang-omni` extensions such as:

- `images`
- `audios`
- `videos`
- video processing overrides such as `video_fps` and frame/pixel limits
- `modalities`
- `audio`
- `stage_sampling`
- `stage_params`
- talker-specific generation overrides
- `request_id`

### Conversion into `GenerateRequest`

`_build_chat_generate_request()` in `openai_api.py` is the key translation point. It:

- normalizes stop sequences
- builds `SamplingParams`
- converts chat messages into internal `Message` objects
- maps per-stage sampling overrides
- passes per-stage runtime params through `stage_params`
- stores media input, audio config, and video processing overrides in request metadata
- stores talker-specific generation overrides in `extra_params`
- copies `modalities` into `output_modalities`

The route hands that `GenerateRequest` to `Client`. The client then converts it
to an `OmniRequest` before submitting it to the coordinator.

## Response Paths

### Non-streaming chat

For non-streaming chat, the path is roughly:

`chat request` → `Client.completion()` → OpenAI-style JSON response

`Client.completion()` aggregates:

- text fragments
- audio chunks
- final usage
- final finish reason

If audio is present, it is base64-encoded before being returned to the API layer.

### Streaming chat

For streaming chat, the server emits SSE events.

The important current semantics are:

- the first chunk may contain only `role="assistant"`
- text and audio are emitted as separate deltas
- the final completion chunk includes the `finish_reason`
- the stream ends with `data: [DONE]`
- `usage` is attached to the final completion chunk

### Speech / TTS

The speech route reuses the same internal request path rather than introducing a separate serving stack.

`CreateSpeechRequest` is converted into a `GenerateRequest` with:

- `output_modalities=["audio"]`
- `task="tts"` in metadata
- TTS-specific parameters stored under `tts_params`

For non-streaming requests, `Client.speech()` collects audio chunks, encodes
them, and returns raw audio bytes to the HTTP layer.

For `stream=true`, the route emits `audio/pcm` bytes directly. The HTTP response
headers are derived from the first audio chunk and subsequent chunks must keep
the same sample rate. TTS chunk-timing knobs such as
`initial_codec_chunk_frames` are forwarded as request params only when the
client provides them, including an explicit `0`, so model schedulers can consume
them without changing Stage, Coordinator, or Relay. When the field is omitted,
the speech route leaves it unset and each model applies its own streaming
default.
