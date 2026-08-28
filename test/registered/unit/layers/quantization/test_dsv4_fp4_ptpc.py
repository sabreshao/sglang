import torch

from sglang.srt.layers.quantization.fp8 import (
    DSV4_DEQUANT_FP4_TABLE,
    cast_e2m1fn_to_e4m3fn_per_channel,
)


def test_dsv4_fp4_to_ptpc_preserves_dequantized_values():
    packed = torch.full((2, 32), 0x21, dtype=torch.int8)
    packed[1].fill_(0x76)
    group_scale = torch.tensor(
        [[2.0, 4.0], [1.0, 0.5]], dtype=torch.float32
    )

    quantized, channel_scale = cast_e2m1fn_to_e4m3fn_per_channel(
        packed, group_scale
    )

    low = packed.to(torch.uint8) & 0x0F
    high = (packed.to(torch.uint8) >> 4) & 0x0F
    expected = torch.stack(
        [DSV4_DEQUANT_FP4_TABLE[low.long()], DSV4_DEQUANT_FP4_TABLE[high.long()]],
        dim=-1,
    ).flatten(1)
    expected = expected * group_scale.repeat_interleave(32, dim=1)
    reconstructed = quantized.float() * channel_scale

    assert quantized.dtype == torch.float8_e4m3fn
    assert channel_scale.shape == (2, 1)
    torch.testing.assert_close(reconstructed, expected, rtol=0.05, atol=0.02)
