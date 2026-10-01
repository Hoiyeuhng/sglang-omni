"""Qwen-specific codec arena, CUDA graphs and fused-kernel checks."""

import os

import pytest
import torch

from sglang_omni.audio.qwen3_tts_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)
from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.utils import snake_beta
from tests.utils.qwen3_tts_codec import Decoder


def arena_decode(
    incremental: Qwen3TTSIncrementalDecoder,
    arena: Qwen3TTSCodecStateArena,
    slots: list[int],
    positions: list[int],
    codes: torch.Tensor,
) -> torch.Tensor:
    """Run one cohort decode the way the vocoder does: gather, decode, scatter."""
    state = arena.gather(slots)
    state.frame_positions = torch.tensor(positions, dtype=torch.long)
    waveform = incremental.decode(codes, state)
    arena.scatter(slots, state)
    return waveform


def make_arena(
    decoder: Decoder, slots: int = 4
) -> tuple[Qwen3TTSIncrementalDecoder, Qwen3TTSCodecStateArena]:
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(
        incremental,
        num_slots=slots,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return incremental, arena


def test_state_spec_covers_every_key_the_decode_creates() -> None:
    """The arena preallocates from the spec, so it must match what decode uses."""
    torch.manual_seed(12)
    decoder = Decoder()
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    spec = incremental.state_spec()

    state = Qwen3TTSIncrementalCodecState()
    incremental.decode(torch.randint(0, 16, (1, 2, 5)), state)

    assert {key for key, _, _ in spec.conv_histories} == set(state.conv_histories)
    assert {key for key, _, _ in spec.transconv_overlaps} == set(
        state.transconv_overlaps
    )
    assert spec.num_layers == len(state.transformer_keys)
    for key, channels, length in spec.conv_histories:
        assert tuple(state.conv_histories[key].shape) == (1, channels, length)
    for key, channels, length in spec.transconv_overlaps:
        assert tuple(state.transconv_overlaps[key].shape) == (1, channels, length)
    assert spec.bytes_per_stream(torch.float32) > 0


def test_arena_backed_decode_matches_the_lazy_state() -> None:
    """Full-width buffers plus negative-position masking must be exact.

    An arena slot always carries the full retained window, so a stream that has
    not filled it reads zeros at negative nominal positions. If the mask did not
    exclude them the early chunks would drift from the lazily grown state, which
    the whole-sequence parity tests above already pin to the reference decoder.
    """
    torch.manual_seed(13)
    decoder = Decoder()
    incremental, arena = make_arena(decoder)
    codes = torch.randint(0, 16, (1, 2, 11))
    partitions = [2, 1, 5, 3]

    lazy_state = Qwen3TTSIncrementalCodecState()
    slot = arena.acquire()
    assert slot is not None

    offset = 0
    position = 0
    for length in partitions:
        chunk = codes[..., offset : offset + length]
        expected = incremental.decode(chunk, lazy_state)
        actual = arena_decode(incremental, arena, [slot], [position], chunk)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        offset += length
        position += length


def test_arena_cohort_matches_per_stream_decodes() -> None:
    """Streams at different playback positions must batch without changing output.

    This is the property the per-row position work exists for: the cohort holds
    one cold stream and two warm streams whose absolute positions differ, and
    every row must match what it produces decoded on its own.
    """
    torch.manual_seed(14)
    decoder = Decoder()
    incremental, arena = make_arena(decoder, slots=8)

    warmups = [0, 3, 9]
    fresh = 2
    streams = [torch.randint(0, 16, (1, 2, warmup + fresh)) for warmup in warmups]

    cohort_slots: list[int] = []
    solo_slots: list[int] = []
    for stream, warmup in zip(streams, warmups):
        cohort_slot = arena.acquire()
        solo_slot = arena.acquire()
        assert cohort_slot is not None and solo_slot is not None
        cohort_slots.append(cohort_slot)
        solo_slots.append(solo_slot)
        for index in range(warmup):
            chunk = stream[..., index : index + 1]
            for slot in (cohort_slot, solo_slot):
                arena_decode(incremental, arena, [slot], [index], chunk)

    expected = [
        arena_decode(incremental, arena, [slot], [warmup], stream[..., warmup:])
        for slot, warmup, stream in zip(solo_slots, warmups, streams)
    ]
    batched = arena_decode(
        incremental,
        arena,
        cohort_slots,
        warmups,
        torch.cat([stream[..., warmup:] for stream, warmup in zip(streams, warmups)]),
    )

    assert batched.shape[0] == 3
    for row, single in enumerate(expected):
        torch.testing.assert_close(batched[row : row + 1], single, rtol=2e-5, atol=2e-6)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("mode", "batch_size", "batch_bucket"),
    [("cold", 1, 1), ("warm", 2, 2), ("warm", 3, 4)],
)
def test_incremental_codec_cuda_graph_matches_eager_state(
    mode: str, batch_size: int, batch_bucket: int
) -> None:
    from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
    from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
        Qwen3TTSIncrementalCodecCudaGraphRunner,
    )

    torch.manual_seed(17)
    device = torch.device("cuda", torch.cuda.current_device())
    decoder = Decoder().to(device).eval()
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(
        incremental, num_slots=batch_bucket + 1, device=device, dtype=torch.float32
    )
    # note (luojiaxuan): rows start at different positions so the replay is
    # checked against a ragged cohort, not only fresh slots.
    slots = []
    for row in range(batch_size):
        slot = arena.acquire()
        slots.append(slot)
        warmup_frames = row * 3
        if warmup_frames == 0:
            continue
        warm_state = arena.gather([slot])
        incremental.decode(
            torch.randint(0, 16, (1, 2, warmup_frames), device=device), warm_state
        )
        arena.scatter([slot], warm_state)
    eager_state = arena.gather(slots)
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        incremental,
        device=device,
        dtype=torch.float32,
        num_quantizers=2,
        mode=mode,
        fresh_frames=(2,),
        batch_sizes=(batch_bucket,),
        min_free_gb=0,
        arena=arena,
    )
    runner.capture()
    stats = runner.stats()
    assert stats["enabled"] is True
    assert stats["binding"]["mode"] == mode
    assert stats["build"]["captured_keys"] == [
        {"fresh_frames": 2, "batch_bucket": batch_bucket}
    ]
    for step in range(10):
        codes = (
            torch.arange(batch_size * 4, device=device)
            .view(batch_size, 2, 2)
            .add(step)
            .remainder(16)
        )
        expected_waveform = incremental.decode(codes, eager_state)
        waveform = runner.decode_slots(codes, slots)
        assert waveform is not None
        torch.cuda.synchronize(device)
        torch.testing.assert_close(waveform, expected_waveform, rtol=2e-4, atol=2e-5)
        graph_state = arena.gather(slots)
        assert graph_state.frame_positions.tolist() == [
            row * 3 + 2 * (step + 1) for row in range(batch_size)
        ]
        torch.testing.assert_close(
            graph_state.frame_positions, eager_state.frame_positions
        )
        for graph_mapping, eager_mapping in (
            (graph_state.transformer_keys, eager_state.transformer_keys),
            (graph_state.transformer_values, eager_state.transformer_values),
            (graph_state.conv_histories, eager_state.conv_histories),
            (graph_state.transconv_overlaps, eager_state.transconv_overlaps),
        ):
            assert graph_mapping.keys() == eager_mapping.keys()
            for key in graph_mapping:
                torch.testing.assert_close(
                    graph_mapping[key], eager_mapping[key], rtol=2e-4, atol=2e-5
                )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_incremental_codec_cuda_graph_alternates_shared_pool_keys() -> None:
    from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
    from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
        Qwen3TTSIncrementalCodecCudaGraphRunner,
    )

    torch.manual_seed(18)
    device = torch.device("cuda", torch.cuda.current_device())
    decoder = Decoder().to(device).eval()
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(
        incremental, num_slots=5, device=device, dtype=torch.float32
    )
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        incremental,
        device=device,
        dtype=torch.float32,
        num_quantizers=2,
        mode="warm",
        fresh_frames=(2,),
        batch_sizes=(1, 4),
        min_free_gb=0,
        arena=arena,
    )
    runner.capture()
    assert len(runner.graphs) == 2
    slots = {1: [arena.acquire()], 4: [arena.acquire() for _ in range(4)]}
    eager_states = {
        batch_size: arena.gather(rows) for batch_size, rows in slots.items()
    }
    for step, batch_size in enumerate((1, 4, 1, 4)):
        codes = (
            torch.arange(batch_size * 4, device=device)
            .view(batch_size, 2, 2)
            .add(step)
            .remainder(16)
        )
        expected = incremental.decode(codes, eager_states[batch_size])
        # note (luojiaxuan): the two keys share one graph pool, so the borrowed
        # waveform is compared before the other key replays over it.
        waveform = runner.decode_slots(codes, slots[batch_size])
        assert waveform is not None
        torch.cuda.synchronize(device)
        torch.testing.assert_close(waveform, expected, rtol=2e-4, atol=2e-5)
        graph_state = arena.gather(slots[batch_size])
        torch.testing.assert_close(
            graph_state.frame_positions, eager_states[batch_size].frame_positions
        )
        for graph_mapping, eager_mapping in (
            (graph_state.transformer_keys, eager_states[batch_size].transformer_keys),
            (
                graph_state.transformer_values,
                eager_states[batch_size].transformer_values,
            ),
            (graph_state.conv_histories, eager_states[batch_size].conv_histories),
            (
                graph_state.transconv_overlaps,
                eager_states[batch_size].transconv_overlaps,
            ),
        ):
            for key in graph_mapping:
                torch.testing.assert_close(
                    graph_mapping[key], eager_mapping[key], rtol=2e-4, atol=2e-5
                )


def test_arena_slot_reuse_starts_from_a_cold_state() -> None:
    torch.manual_seed(15)
    decoder = Decoder()
    incremental, arena = make_arena(decoder, slots=1)
    codes = torch.randint(0, 16, (1, 2, 3))

    slot = arena.acquire()
    assert slot == 0
    cold = arena_decode(incremental, arena, [slot], [0], codes)
    arena_decode(incremental, arena, [slot], [3], codes)
    assert arena.active_slots() == 1

    arena.release(slot)
    assert arena.active_slots() == 0
    reused = arena.acquire()
    assert reused == slot
    again = arena_decode(incremental, arena, [reused], [0], codes)

    torch.testing.assert_close(again, cold)


def test_arena_reports_exhaustion_and_retirement() -> None:
    decoder = Decoder()
    _, arena = make_arena(decoder, slots=1)

    slot = arena.acquire()
    assert slot is not None
    assert arena.acquire() is None
    assert arena.exhausted_count == 1

    arena.release(slot)
    reused = arena.acquire()
    assert reused is not None
    arena.retire(reused)
    assert arena.acquire() is None
    assert arena.active_slots() == 0
    # Note (Qihao Liu): a retired slot stays withdrawn even if its owner
    # releases it later.
    arena.release(reused)
    assert arena.acquire() is None
    assert arena.describe()["bytes_per_slot"] == arena.bytes_per_slot


def fresh_state(rows: int) -> Qwen3TTSIncrementalCodecState:
    state = Qwen3TTSIncrementalCodecState()
    state.frame_positions = torch.zeros(rows, dtype=torch.long)
    return state


def test_incremental_decoder_routes_only_precompiled_shapes_to_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(6)
    decoder = Decoder()
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    calls: list[tuple[int, int]] = []

    def fake_compile(fn, **kwargs):
        def wrapped(codes, state):
            calls.append((int(codes.shape[0]), int(codes.shape[-1])))
            return fn(codes, state)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)
    trace_codes = torch.randint(0, 16, (2, 2, 3))
    with torch.inference_mode():
        incremental.precompile(trace_codes, fresh_state(2))
        incremental.precompile(trace_codes, fresh_state(2))
    assert calls == [(2, 3)], "precompile traces a shape once, on the given tensors"

    codes = torch.randint(0, 16, (2, 2, 9))
    expected = decoder(codes)
    state = Qwen3TTSIncrementalCodecState()
    state.frame_positions = torch.zeros(2, dtype=torch.long)
    actual = torch.cat(
        [
            incremental.decode(codes[..., i : i + 3], state, compiled=True)
            for i in range(0, 9, 3)
        ],
        dim=-1,
    )
    assert calls == [(2, 3)] * 4, "every 3-frame step runs through the kernel"
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert state.frame_position == 9
    assert state.transformer_context_length == decoder.pre_transformer.window_size - 1

    other = Qwen3TTSIncrementalCodecState()
    other.frame_positions = torch.zeros(2, dtype=torch.long)
    incremental.decode(codes[..., :3], other)
    assert calls == [(2, 3)] * 4, "eager decodes never enter the kernel"
    with pytest.raises(RuntimeError, match="not precompiled"):
        incremental.decode(codes[..., :4], other, compiled=True)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_arena_bound_graph_replays_match_eager_and_advance_the_arena() -> None:
    """A captured graph gathers, decodes and scatters arena rows exactly like the eager path."""
    from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
    from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
        Qwen3TTSIncrementalCodecCudaGraphRunner,
    )

    torch.manual_seed(7)
    device = torch.device("cuda", torch.cuda.current_device())
    decoder = Decoder().to(device)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(
        incremental, num_slots=3, device=device, dtype=torch.float32
    )
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        incremental,
        device=device,
        dtype=torch.float32,
        num_quantizers=2,
        mode="warm",
        fresh_frames=(2,),
        batch_sizes=(1, 2),
        min_free_gb=0.0,
        arena=arena,
    )
    runner.capture()
    assert runner.stats()["build"]["capture_complete"]

    slots = [arena.acquire(), arena.acquire()]
    bystander = arena.acquire()
    codes = torch.randint(0, 16, (2, 2, 6), device=device)

    # note (luojiaxuan): eager reference on private copies of the same zeroed rows
    reference = Qwen3TTSIncrementalCodecState()
    reference.frame_positions = torch.zeros(2, dtype=torch.long, device=device)
    expected = torch.cat(
        [incremental.decode(codes[..., i : i + 2], reference) for i in range(0, 6, 2)],
        dim=-1,
    )

    actual = []
    for i in range(0, 6, 2):
        waveform = runner.decode_slots(codes[..., i : i + 2], slots)
        assert waveform is not None
        actual.append(waveform.clone())
    torch.cuda.synchronize()
    torch.testing.assert_close(
        torch.cat(actual, dim=-1), expected, rtol=2e-4, atol=2e-5
    )

    advanced = arena.gather(slots)
    assert advanced.frame_positions.tolist() == [6, 6]
    torch.testing.assert_close(
        advanced.transformer_keys[0],
        reference.transformer_keys[0],
        rtol=2e-4,
        atol=2e-5,
    )
    untouched = arena.gather([bystander])
    assert untouched.frame_positions.tolist() == [0]
    assert torch.count_nonzero(untouched.transformer_keys[0]).item() == 0

    # note (luojiaxuan): a one-row cohort replays the two-row bucket; the padded row
    # lands on scratch
    single = runner.decode_slots(codes[:1, :, :2], slots[:1])
    assert single is not None and single.shape[0] == 1
    torch.cuda.synchronize()
    assert arena.gather(slots).frame_positions.tolist() == [8, 6]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_windowed_replays_match_one_eager_decode_and_its_arena_state() -> None:
    """A bootstrap split across captured windows, with one window replayed twice, leaves
    the waveform and every arena row where a single eager decode of the width would."""
    from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
        Qwen3TTSIncrementalCodecCudaGraphRunner,
    )
    from sglang_omni.models.qwen3_tts.streaming_vocoder import (
        IncrementalDecodeBatch,
        IncrementalDecodePlan,
        Qwen3TTSStreamingVocoderScheduler,
    )

    torch.manual_seed(23)
    device = torch.device("cuda", torch.cuda.current_device())
    decoder = Decoder().to(device).eval()
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(
        incremental, num_slots=4, device=device, dtype=torch.float32
    )
    slots = [arena.acquire(), arena.acquire()]
    bystander = arena.acquire()
    warm_state = arena.gather(slots[1:])
    incremental.decode(torch.randint(0, 16, (1, 2, 3), device=device), warm_state)
    arena.scatter(slots[1:], warm_state)

    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        incremental,
        device=device,
        dtype=torch.float32,
        num_quantizers=2,
        mode="window",
        fresh_frames=(1, 2, 4, 8),
        batch_sizes=(1, 2),
        min_free_gb=0.0,
        arena=arena,
    )
    runner.capture()
    assert len(runner.stats()["build"]["captured_keys"]) == 8

    width = 21
    split = runner.split_frames(width)
    assert split == (8, 8, 4, 1)
    codes = torch.randint(0, 16, (2, 2, width), device=device)
    eager_state = arena.gather(slots)
    eager_waveform = incremental.decode(codes, eager_state)

    scheduler = Qwen3TTSStreamingVocoderScheduler.__new__(
        Qwen3TTSStreamingVocoderScheduler
    )
    scheduler.samples_per_frame = decoder.total_upsample
    plans = [
        IncrementalDecodePlan(
            decoder_input=codes[0:1],
            slot=slots[0],
            fresh_frames=width,
            reference_trim_frames=9,
            generated_frames=12,
            emitted_generated_frames=0,
        ),
        IncrementalDecodePlan(
            decoder_input=codes[1:2],
            slot=slots[1],
            fresh_frames=width,
            reference_trim_frames=15,
            generated_frames=6,
            emitted_generated_frames=0,
        ),
    ]
    batch = IncrementalDecodeBatch(decoder=incremental, arena=arena, slots=slots)
    deltas, waveform = scheduler.decode_incremental_windows(
        codes, plans, batch, runner, split
    )
    torch.cuda.synchronize(device)

    assert runner.stats()["runtime"]["replays"] == 4
    torch.testing.assert_close(waveform, eager_waveform[:, 0], rtol=2e-4, atol=2e-5)
    samples = decoder.total_upsample
    for plan, delta, row in zip(plans, deltas, eager_waveform[:, 0]):
        start = plan.reference_trim_frames * samples
        end = start + plan.generated_frames * samples
        torch.testing.assert_close(delta, row[start:end], rtol=2e-4, atol=2e-5)
        assert (
            delta.untyped_storage().data_ptr() == waveform.untyped_storage().data_ptr()
        )

    graph_state = arena.gather(slots)
    assert graph_state.frame_positions.tolist() == [width, 3 + width]
    torch.testing.assert_close(graph_state.frame_positions, eager_state.frame_positions)
    for graph_mapping, eager_mapping in (
        (graph_state.transformer_keys, eager_state.transformer_keys),
        (graph_state.transformer_values, eager_state.transformer_values),
        (graph_state.conv_histories, eager_state.conv_histories),
        (graph_state.transconv_overlaps, eager_state.transconv_overlaps),
    ):
        assert graph_mapping.keys() == eager_mapping.keys()
        for key in graph_mapping:
            torch.testing.assert_close(
                graph_mapping[key], eager_mapping[key], rtol=2e-4, atol=2e-5
            )
    assert arena.gather([bystander]).frame_positions.tolist() == [0]


@pytest.mark.benchmark
@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_tts_decoder_and_incremental_pcm_equal() -> None:
    checkpoint = os.environ.get("QWEN3_TTS_TOKENIZER_PATH")
    if checkpoint is None:
        pytest.skip("Set QWEN3_TTS_TOKENIZER_PATH to run the real checkpoint gate")
    from sglang_omni.audio.qwen3_tts_codec import (
        Qwen3TTSIncrementalCodecState,
        Qwen3TTSIncrementalDecoder,
    )
    from sglang_omni.audio.qwen3_tts_compat import (
        apply_qwen_tts_transformers_compatibility_patches,
    )

    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts import Qwen3TTSTokenizer

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        checkpoint,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    decoder = tokenizer.model.decoder.eval()
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    codes = [
        torch.randint(
            decoder.config.codebook_size,
            (batch, decoder.config.num_quantizers, frames),
            device="cuda:0",
            generator=generator,
        )
        for batch, frames in ((1, 2), (1, 24), (1, 35), (8, 24))
    ]
    with torch.inference_mode():
        expected = [decoder(value).clone() for value in codes]
        incremental = Qwen3TTSIncrementalDecoder(decoder)
        state = Qwen3TTSIncrementalCodecState()
        parts = codes[1].split((2, 6, 8, 8), dim=-1)
        incremental_expected = [
            incremental.decode(part, state).clone() for part in parts
        ]

        assert snake_beta.fuse_vocoder_decoder(decoder) == 29
        for value, pcm in zip(codes, expected):
            assert torch.equal(decoder(value), pcm), tuple(value.shape)
        incremental = Qwen3TTSIncrementalDecoder(decoder)
        state = Qwen3TTSIncrementalCodecState()
        for part, pcm in zip(parts, incremental_expected):
            assert torch.equal(incremental.decode(part, state), pcm)
