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

### Timeouts for shared duplex sessions

The runtime has an admission timeout and an input-idle timeout.

| Timeout | Runtime limit | Default | Function |
| --- | --- | --- | --- |
| Admission timeout | `session_update_timeout_s` | 60 seconds | The server waits for `session.update` before the session opens |
| Input-idle timeout | `idle_input_timeout_s` | 60 seconds | The server closes an open session after a period without input |

To use these options, set `--enable-realtime`.
The pipeline must have `realtime_deployment_factory`.
`MiniCPMODuplexPipelineConfig` has this factory.

Use these server options to change the timeouts:

| Server option | Python keyword argument | Example value |
| --- | --- | --- |
| `--realtime-admission-timeout-s` | `realtime_admission_timeout_s` | `20` seconds |
| `--realtime-idle-input-timeout-s` | `realtime_idle_input_timeout_s` | `300` seconds |

Set each timeout to a finite number greater than zero.
If you do not set an option, the server uses the deployment limit.
Give the Python keyword arguments to `launch_server` or `run_server`.
Do not add these options to the pipeline YAML.

The admission timeout period starts when the connection starts.
An admission failure does not start a new period.
The runtime waits for an admission operation and its cleanup before it can close the session for this timeout.
Each operation has a time limit.
After the end of this period, another `session.update` cannot start admission.

The runtime starts a new input-idle timeout period after each of these events:

- The runtime accepts audio input or an image.
- An input clear or input end command completes.
- The adapter completes an input unit.

The runtime does not use the input-idle timeout while the adapter processes an input unit.
The runtime does not use this timeout while a command holds the command lock.
WebSocket pings and session configuration updates do not start a new input-idle timeout period.
Images that the runtime rejects do not start a new input-idle timeout period.

For shared realtime sessions, `CoordinatorAdapter` keeps a finite pipeline idle timeout as a second protection.
It uses the maximum of these values:

- The default pipeline idle timeout of 300 seconds.
- The pipeline idle timeout in the configuration.
- Two runtime input-idle periods, plus the runtime cleanup timeout and the pipeline operation timeout.

The runtime can accept input before it has a full pipeline input unit.
The pipeline idle timer operates while the pipeline waits for another input unit.
Inputs that do not make a full unit can keep the runtime active until this pipeline timeout occurs.
Image input and clear commands also do not give the pipeline an input unit.
The pipeline timeout also gives a time limit if runtime cleanup fails.

Other pipeline sessions keep their idle timeout.
The default pipeline idle timeout is 300 seconds.
The operation timeout limits and cleanup timeout limits do not change.

### Release note: input-idle timeout

This change adds a default input-idle timeout of 60 seconds to shared duplex sessions.
Before this change, an open session stayed open after application input stopped.
A client that sends audio only when its user speaks can now disconnect during a long period without input.
WebSocket Pong frames do not prevent this closure.

For long periods without audio input, increase `--realtime-idle-input-timeout-s`.
For example, use `--realtime-idle-input-timeout-s 300` for a five-minute timeout period.

### Transport timeouts for shared duplex sessions

`RuntimeLimits.transport_timeout_s` has a default value of 30 seconds.
Set this deployment limit to a finite number greater than zero.
The capabilities endpoint gives this value with the admission timeout and the input-idle timeout.

Each WebSocket text write has a transport timeout.
If a write timeout occurs while the runtime is active, the runtime starts cleanup with the reason `transport_timeout`.
If there is already a close task, that task keeps its initial reason.

After native cleanup, the connection waits a maximum of one transport timeout period for the last events and the Close frame.
If the sender does not complete, the connection cancels the local transport tasks.
The connection gives these tasks a maximum of two more transport timeout periods to stop.
If the connection receives more cancellation requests, it continues to wait for this local cleanup.
The connection raises the initial `CancelledError` after local cleanup.
If local cleanup has an error, that error is the cause of the cancellation exception.

The local transport tasks must stop after cancellation for these time limits to be correct.
With this condition, the connection slot becomes available in three transport timeout periods or less after native cleanup.
The event loop can make this total period longer.

Transport timeouts do not cancel the native cleanup task.
Native cleanup uses the cleanup timeout limits.
After a transport timeout, the peer does not always receive the last JSON events or the Close frame.

The ASGI backend controls TCP shutdown.
Its Close handshake timer can continue after the connection slot becomes available.

### WebSocket close codes for shared duplex sessions

The server sends the last application events before it sends the WebSocket Close frame.
The Close frame gives the code and reason for session closure.

| Close code | Reason for session closure |
| --- | --- |
| `1000` | `client_closed` or `disconnect` |
| `1008` | `session_update_timeout`, `idle_timeout`, or `context_exhausted` |
| `1011` | An internal error, adapter failure, or cleanup failure |

[RFC 6455 section 7.4.1](https://www.rfc-editor.org/rfc/rfc6455.html#section-7.4.1) gives these codes.
The server uses `1008` for the admission timeout, the input-idle timeout, and the context limit.
The RFC does not set the timeout periods or the JSON error codes.
The application sets the JSON error codes.

The reason in the Close frame has a maximum length of 123 UTF-8 bytes.
If the reason is too long, the server removes characters from its end.
The server does not keep part of a UTF-8 character.
The JSON events keep the full reason.

A non-fatal request error does not close the connection.

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
