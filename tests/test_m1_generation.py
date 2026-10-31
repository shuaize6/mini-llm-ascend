"""对照真实 HF generate，验证 EOS 策略和请求缓存隔离。"""

import unittest
from unittest.mock import patch

import torch
from transformers import GenerationConfig

from mini_llm_ascend.engine.model_runner import ModelRunner
from mini_llm_ascend.engine.sampler import greedy_select
from test_model import tiny_models


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.hf, self.model = tiny_models()
        self.runner = ModelRunner(self.model, capacity=16)
        self.ids = torch.tensor([[1, 4, 7]])

    def test_fixed_generation_matches_hf_and_masks_eos(self):
        eos = greedy_select(self.model(self.ids))  # 刻意让最高分成为 EOS。
        config = GenerationConfig(do_sample=False, max_new_tokens=4, min_new_tokens=4,
                                  eos_token_id=eos, pad_token_id=0, use_cache=True)
        with torch.inference_mode(), patch.object(torch, "multinomial", side_effect=AssertionError("sampling")):
            expected = self.hf.generate(self.ids, generation_config=config, use_model_defaults=False)
            actual = self.runner.generate(self.ids, 4, eos_token_id=[eos], fixed_length=True)
        self.assertEqual(actual, expected[0, 3:].tolist())
        self.assertNotIn(eos, actual)
        self.assertEqual(self.runner.cache.length, 0)

    def test_one_token_and_early_eos_only_do_prefill(self):
        eos = greedy_select(self.model(self.ids))
        for budget in (1, 5):
            with patch.object(self.model, "forward", wraps=self.model.forward) as call:
                self.assertEqual(self.runner.generate(self.ids, budget, eos_token_id=eos), [eos])
                self.assertEqual(call.call_count, 1)

    def test_second_request_and_callback_failure_reset_cache(self):
        expected = self.runner.generate(self.ids, 3)
        self.runner.generate(torch.tensor([[6, 8]]), 2)
        self.assertEqual(self.runner.generate(self.ids, 3), expected)
        def fail(_):
            raise RuntimeError("callback failed")
        with self.assertRaises(RuntimeError):
            self.runner.generate(self.ids, 3, on_token=fail)
        self.assertEqual(self.runner.cache.length, 0)
        self.assertEqual(self.runner.generate(self.ids, 3), expected)

    def test_teacher_forcing_matches_hf_at_every_step(self):
        cache = self.model.new_cache(16)
        hf_cache = None
        for ids in (self.ids, torch.tensor([[5]]), torch.tensor([[2]])):
            with torch.inference_mode():
                ref = self.hf(ids, past_key_values=hf_cache, use_cache=True)
            hf_cache = ref.past_key_values
            torch.testing.assert_close(self.model(ids, cache), ref.logits[:, -1:], atol=1e-5, rtol=1e-4)

    def test_reject_budget_and_nonfinite_logits(self):
        for budget in (0, 20):
            with self.assertRaises(ValueError):
                self.runner.generate(self.ids, budget)
        with self.assertRaises(ValueError):
            greedy_select(torch.full((1, 1, 4), float("nan")))
        with self.assertRaises(ValueError):
            greedy_select(torch.zeros(1, 1, 2), [0, 1], suppress_eos=True)


if __name__ == "__main__":
    unittest.main()
