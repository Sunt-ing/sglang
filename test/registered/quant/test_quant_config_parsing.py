import unittest
from unittest.mock import MagicMock

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestQuantLogString(CustomTestCase):
    def test_qwen_fp8_config(self):
        # Example from Qwen/Qwen3-4B-Thinking-2507-FP8
        quant_config = {
            "activation_scheme": "dynamic",
            "modules_to_not_convert": ["lm_head"],
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }

        # Create a raw instance
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=fp8, fmt=e4m3"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test Qwen FP8] Result: {result}")
        self.assertEqual(result, expected)

    def test_llama_gptq_int4_config(self):
        # Example from hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
        quant_config = {"bits": 4, "quant_method": "gptq", "group_size": 128}
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=gptq, bits=4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test Llama GPTQ] Result: {result}")
        self.assertEqual(result, expected)

    def test_awq_config(self):
        quant_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
        }
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=awq, bits=4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test AWQ] Result: {result}")
        self.assertEqual(result, expected)

    def test_modelopt_nvfp4(self):
        quant_config = {"quant_method": "modelopt_fp4", "quant_algo": "NVFP4"}
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=modelopt_fp4, quant_algo=NVFP4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test ModelOpt] Result: {result}")
        self.assertEqual(result, expected)

    def test_no_quant_config(self):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=None)

        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test No Quant] Result: {result}")
        self.assertIsNone(result)


class TestSupportsCutlass24(CustomTestCase):
    @staticmethod
    def _scheme(fmt, structure="2:4"):
        scheme = MagicMock()
        scheme.sparsity_structure = structure
        scheme.format = fmt
        return scheme

    def test_current_2of4_sparse_formats_are_recognized(self):
        # These formats must reach the existing 2:4 reject path instead of
        # silently loading as dense and dropping sparse packed weights.
        for fmt in ("sparse-bitmask", "sparse-24"):
            with self.subTest(format=fmt):
                self.assertTrue(
                    CompressedTensorsConfig.supports_cutlass_24(
                        weight_quant=None,
                        input_quant=None,
                        sparsity_scheme=self._scheme(fmt),
                    )
                )

    def test_non_2of4_structure_not_recognized(self):
        self.assertFalse(
            CompressedTensorsConfig.supports_cutlass_24(
                weight_quant=None,
                input_quant=None,
                sparsity_scheme=self._scheme(
                    "sparse-bitmask", structure="unstructured"
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
