"""Backend selection tests for AMD FP8 block GEMM."""

import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization import fp8_utils
from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_amd_ci(est_time=5, suite="stage-a-test-1-gpu-small-amd")


class TestAiterFp8Dispatch(unittest.TestCase):
    def test_gfx942_uses_aiter_ck(self):
        with patch.object(fp8_utils, "_use_aiter_bpreshuffle_gfx95", False), patch.object(
            fp8_utils, "_use_aiter_gfx95", False
        ):
            self.assertFalse(fp8_utils._aiter_w8a8_use_triton(4096, 7168, 1))

    def test_gfx95_tuned_shape_keeps_triton_fallback(self):
        with patch.object(fp8_utils, "_use_aiter_bpreshuffle_gfx95", True), patch.object(
            fp8_utils, "_use_aiter_gfx95", True
        ), patch.object(fp8_utils, "_FORCE_CK_W8A8", False):
            self.assertTrue(fp8_utils._aiter_w8a8_use_triton(4096, 7168, 1))

    def test_gfx95_force_ck_overrides_tuned_fallback(self):
        with patch.object(fp8_utils, "_use_aiter_bpreshuffle_gfx95", True), patch.object(
            fp8_utils, "_use_aiter_gfx95", True
        ), patch.object(fp8_utils, "_FORCE_CK_W8A8", True):
            self.assertFalse(fp8_utils._aiter_w8a8_use_triton(4096, 7168, 1))

    def test_gfx95_unsafe_large_m_keeps_triton_fallback(self):
        with patch.object(fp8_utils, "_use_aiter_bpreshuffle_gfx95", False), patch.object(
            fp8_utils, "_use_aiter_gfx95", True
        ), patch.object(fp8_utils, "_FORCE_CK_W8A8", True):
            self.assertTrue(fp8_utils._aiter_w8a8_use_triton(2560, 4096, 2049))


if __name__ == "__main__":
    unittest.main()
