"""验证跨层长度、容量、重置与失败状态，不要求 NPU。"""

import unittest
from unittest.mock import patch

import torch

from mini_llm_ascend.cache.contiguous_kv import ContiguousKVCache
from test_model import tiny_models


class CacheTests(unittest.TestCase):
    def new_cache(self):
        return ContiguousKVCache(2, 2, 3, 4, device="cpu", dtype=torch.float32)

    def test_commit_once_and_exact_capacity(self):
        cache = self.new_cache()
        cache.begin_step(3)
        value = torch.ones(1, 2, 3, 4)
        cache.write_layer(0, value, value * 2)
        self.assertEqual(cache.length, 0)
        with self.assertRaises(RuntimeError):
            cache.commit_step()
        with self.assertRaises(ValueError):
            cache.write_layer(0, value, value)
        cache.write_layer(1, value * 3, value * 4)
        torch.testing.assert_close(cache.view_layer(1)[0], value * 3)
        cache.commit_step()
        self.assertEqual(cache.length, 3)
        self.assertEqual(cache.nbytes, 2*2*2*3*4*4)
        with self.assertRaises(ValueError):
            cache.begin_step(1)
        self.assertEqual(cache.length, 3)

    def test_bad_shape_dtype_and_unwritten_read(self):
        cache = self.new_cache()
        cache.begin_step(1)
        with self.assertRaises(RuntimeError):
            cache.view_layer(0)
        for value in (torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 1, 4, dtype=torch.float16)):
            with self.assertRaises(ValueError):
                cache.write_layer(0, value, value)

    def test_abort_requires_reset_and_old_values_are_invisible(self):
        cache = self.new_cache()
        cache.begin_step(2)
        cache.write_layer(0, torch.ones(1, 2, 2, 4), torch.ones(1, 2, 2, 4))
        cache.abort_step()
        with self.assertRaises(RuntimeError):
            cache.begin_step(1)
        cache.reset()
        cache.begin_step(1)
        fresh = torch.full((1, 2, 1, 4), 9.0)
        for layer in range(2):
            cache.write_layer(layer, fresh, fresh)
            torch.testing.assert_close(cache.view_layer(layer)[0], fresh)
        cache.commit_step()
        self.assertEqual(cache.length, 1)

    def test_failed_model_forward_poisoned_cache(self):
        _, model = tiny_models()
        cache = model.new_cache(8)
        with patch.object(model.model.layers[1], "forward", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                model(torch.tensor([[1, 4]]), cache)
        self.assertEqual(cache.length, 0)
        with self.assertRaises(RuntimeError):
            model(torch.tensor([[1]]), cache)
        cache.reset()
        actual = model(torch.tensor([[3, 2]]), cache)
        torch.testing.assert_close(actual, model(torch.tensor([[3, 2]])))


if __name__ == "__main__":
    unittest.main()
