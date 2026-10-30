"""计时边界与真实 HF 生成的最小验证，不下载模型。"""

import unittest
from unittest.mock import patch

from bench.baseline import TokenTimer, run_once, timing_metrics


class FakeToken:
    def numel(self):
        return 1


class TimingTests(unittest.TestCase):
    def test_prompt_is_not_first_generated_token(self):
        timer = TokenTimer()
        with patch("bench.baseline.time.perf_counter", side_effect=[12.0, 12.5]):
            timer.put(FakeToken())  # prompt：不记时
            timer.put(FakeToken())
            timer.put(FakeToken())
        self.assertEqual(timer.timestamps, [12.0, 12.5])

    def test_finalization_does_not_inflate_decode_interval(self):
        result = timing_metrics(10, 15, [12, 12.5, 13])
        self.assertEqual(result["ttft_ms"], 2000)
        self.assertEqual(result["tpot_ms"], 500)
        self.assertEqual(result["finalization_ms"], 2000)
        self.assertAlmostEqual(result["output_tokens_per_s"], 0.6)

    def test_one_token_has_no_decode_interval(self):
        self.assertIsNone(timing_metrics(1, 3, [2])["tpot_ms"])

    def test_no_tokens_and_bad_clock_are_errors(self):
        with self.assertRaises(ValueError):
            timing_metrics(1, 3, [])
        with self.assertRaises(ValueError):
            timing_metrics(1, 3, [2.5, 2])


class TinyQwenIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

        cls.torch = torch
        cls.GenerationConfig = GenerationConfig
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        config._attn_implementation = "eager"
        cls.model = Qwen3ForCausalLM(config).eval()
        # 真实 Qwen checkpoint 默认启用采样，不能让它覆盖本基线的 greedy。
        cls.model.generation_config.do_sample = True
        cls.model.generation_config.temperature = 0.6
        cls.inputs = {"input_ids": torch.tensor([[1, 4, 5]]),
                      "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def run_config(self, **kwargs):
        return self.GenerationConfig(do_sample=False, use_cache=True,
                                     pad_token_id=0, bos_token_id=1, **kwargs)

    def test_streamer_matches_direct_generate_and_fixed_length(self):
        config = self.run_config(max_new_tokens=4, min_new_tokens=4, eos_token_id=2)
        with patch.object(self.torch, "multinomial", side_effect=AssertionError("意外启用采样")):
            result = run_once(self.model, self.inputs, config, self.torch, self.torch.device("cpu"))
        with self.torch.inference_mode():
            expected = self.model.generate(**self.inputs, generation_config=config,
                                           logits_to_keep=1, use_model_defaults=False)
        self.assertEqual(result["output_ids"], expected[0, 3:].tolist())
        self.assertEqual(result["output_tokens"], 4)
        self.assertEqual(len(result["itl_ms"]), 3)

    def test_early_eos_counts_actual_tokens(self):
        # 将模型的第一个贪心预测设为 EOS，验证不使用 max_new_tokens 冒充输出长度。
        with self.torch.inference_mode():
            eos = self.model(**self.inputs).logits[0, -1].argmax().item()
        config = self.run_config(max_new_tokens=4, eos_token_id=eos)
        result = run_once(self.model, self.inputs, config, self.torch, self.torch.device("cpu"))
        self.assertEqual(result["output_tokens"], 1)
        self.assertEqual(result["output_ids"], [eos])
        self.assertIsNone(result["tpot_ms"])


if __name__ == "__main__":
    unittest.main()
