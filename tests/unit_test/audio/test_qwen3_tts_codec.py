# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.audio.qwen3_tts_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
    incremental_causal_conv1d,
    incremental_causal_transconv1d,
    incremental_transformer,
)
from tests.utils.qwen3_tts_codec import (
    CausalConv,
    CausalTransConv,
    Decoder,
    Transformer,
    full_transformer,
    random_partitions,
)


@pytest.mark.parametrize("partitions", [[9], [1] * 9, [1, 8], [8, 1], [3, 2, 4]])
def test_incremental_causal_conv_matches_whole(
    partitions: list[int],
) -> None:
    torch.manual_seed(1)
    module = CausalConv(2, 3, 7, dilation=3)
    inputs = torch.randn(1, 2, sum(partitions))
    expected = module(inputs)
    state = Qwen3TTSIncrementalCodecState()
    actual = []
    offset = 0
    for length in partitions:
        actual.append(
            incremental_causal_conv1d(
                module, inputs[..., offset : offset + length], state, "conv"
            )
        )
        offset += length

    torch.testing.assert_close(torch.cat(actual, dim=-1), expected)
    assert state.conv_histories["conv"].shape[-1] == module.padding


@pytest.mark.parametrize("partitions", [[9], [1] * 9, [1, 8], [8, 1], [3, 2, 4]])
def test_incremental_causal_transconv_matches_whole(
    partitions: list[int],
) -> None:
    torch.manual_seed(2)
    module = CausalTransConv(2, 3, 8, 4)
    inputs = torch.randn(1, 2, sum(partitions))
    expected = module(inputs)
    state = Qwen3TTSIncrementalCodecState()
    actual = []
    offset = 0
    for length in partitions:
        actual.append(
            incremental_causal_transconv1d(
                module, inputs[..., offset : offset + length], state, "transconv"
            )
        )
        offset += length

    torch.testing.assert_close(torch.cat(actual, dim=-1), expected)
    assert state.transconv_overlaps["transconv"].shape[-1] == module.right_pad


@pytest.mark.parametrize("partitions", [[11], [1] * 11, [1, 10], [10, 1], [3, 2, 6]])
def test_incremental_transformer_matches_whole_across_window(
    partitions: list[int],
) -> None:
    torch.manual_seed(3)
    transformer = Transformer()
    inputs = torch.randn(1, sum(partitions), 2)
    expected = full_transformer(transformer, inputs)
    state = Qwen3TTSIncrementalCodecState()
    actual = []
    offset = 0
    for length in partitions:
        actual.append(
            incremental_transformer(
                transformer, inputs[:, offset : offset + length], state
            )
        )
        state.frame_position += length
        state.transformer_context_length = min(
            transformer.window_size - 1, state.transformer_context_length + length
        )
        offset += length

    torch.testing.assert_close(torch.cat(actual, dim=1), expected)
    assert state.transformer_context_length == transformer.window_size - 1
    assert all(
        item.shape[-2] == transformer.window_size - 1
        for item in state.transformer_keys.values()
    )


def test_incremental_codec_state_clone_owns_tensor_storage() -> None:
    state = Qwen3TTSIncrementalCodecState(
        frame_position=3,
        transformer_context_length=2,
        transformer_keys={0: torch.ones(1, 1, 2, 2)},
        transformer_values={0: torch.ones(1, 1, 2, 2)},
        conv_histories={"conv": torch.ones(1, 1, 2)},
        transconv_overlaps={"transconv": torch.ones(1, 1, 2)},
    )

    cloned = state.clone()
    cloned.transformer_keys[0].zero_()
    cloned.transformer_values[0].zero_()
    cloned.conv_histories["conv"].zero_()
    cloned.transconv_overlaps["transconv"].zero_()

    assert state.transformer_keys[0].count_nonzero() == 4
    assert state.transformer_values[0].count_nonzero() == 4
    assert state.conv_histories["conv"].count_nonzero() == 2
    assert state.transconv_overlaps["transconv"].count_nonzero() == 2


@pytest.mark.parametrize(
    "partitions",
    [[11], [1] * 11, [1, 10], [10, 1], [3, 2, 6], random_partitions(96, 5)],
)
def test_incremental_decoder_matches_whole_for_arbitrary_partitions(
    partitions: list[int],
) -> None:
    torch.manual_seed(4)
    decoder = Decoder()
    codes = torch.randint(0, 16, (1, 2, sum(partitions)))
    expected = decoder(codes)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    state = Qwen3TTSIncrementalCodecState()
    actual = []
    offset = 0
    for length in partitions:
        actual.append(incremental.decode(codes[..., offset : offset + length], state))
        offset += length

    actual_waveform = torch.cat(actual, dim=-1)
    torch.testing.assert_close(actual_waveform, expected, rtol=2e-5, atol=2e-6)
    assert actual_waveform.shape[-1] == sum(partitions) * decoder.total_upsample
    assert state.frame_position == sum(partitions)
    assert state.transformer_context_length == decoder.pre_transformer.window_size - 1


def test_incremental_decoder_reference_prefix_matches_generated_waveform() -> None:
    torch.manual_seed(5)
    decoder = Decoder()
    codes = torch.randint(0, 16, (1, 2, 9))
    reference_frames = 4
    expected = decoder(codes)[..., reference_frames * decoder.total_upsample :]
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    state = Qwen3TTSIncrementalCodecState()

    initial = incremental.decode(codes[..., :7], state)
    initial = initial[..., reference_frames * decoder.total_upsample :]
    final = incremental.decode(codes[..., 7:], state)
    actual = torch.cat((initial, final), dim=-1)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert actual.shape[-1] == (9 - reference_frames) * decoder.total_upsample


def test_incremental_decoder_rejects_malformed_codes() -> None:
    decoder = Decoder()
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    state = Qwen3TTSIncrementalCodecState()

    with pytest.raises(ValueError, match=r"\[B, Q, T\]"):
        incremental.decode(torch.ones(2, 1, dtype=torch.long), state)
    with pytest.raises(ValueError, match="fresh frames"):
        incremental.decode(torch.ones(1, 2, 0, dtype=torch.long), state)


def test_incremental_decoder_batches_rows_sharing_a_position() -> None:
    """A batch of identical-position rows must equal the same rows decoded alone."""
    torch.manual_seed(11)
    decoder = Decoder()
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    codes = torch.randint(0, 16, (3, 2, 6))

    batched_state = Qwen3TTSIncrementalCodecState(
        frame_positions=torch.zeros(3, dtype=torch.long)
    )
    batched = incremental.decode(codes, batched_state)

    for row in range(3):
        single = incremental.decode(
            codes[row : row + 1], Qwen3TTSIncrementalCodecState()
        )
        torch.testing.assert_close(batched[row : row + 1], single, rtol=2e-5, atol=2e-6)
    assert batched_state.frame_positions.tolist() == [6, 6, 6]
