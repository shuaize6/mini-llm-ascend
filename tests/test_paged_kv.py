"""M2 必要测试：只覆盖新增存储机制和 M1 生成循环的连接。"""

import unittest
from unittest.mock import patch

import torch

from mini_llm_ascend.cache.block_manager import BlockManager, CacheCapacityError
from mini_llm_ascend.cache.paged_kv import PagedKVPool
from mini_llm_ascend.engine.model import ModelConfig, Qwen3
from mini_llm_ascend.engine.model_runner import ModelRunner


def tiny_model():
    torch.manual_seed(17)
    return Qwen3(ModelConfig(vocab_size=41, hidden_size=24, intermediate_size=40,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=64)).eval()


def append(cache, values):
    """每层不同数值，便于抓住层索引错用；返回最后层可见前缀。"""
    key = torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1).expand(1, 2, -1, 4)
    cache.begin_step(len(values))
    for layer in range(len(cache.keys)):
        cache.write_layer(layer, key + layer * 100, key + layer * 100 + 1000)
    visible = tuple(t.clone() for t in cache.view_layer(len(cache.keys) - 1))
    cache.commit_step()
    return visible


class PagedKVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def small_pool(self, blocks=3):
        return PagedKVPool(2, 2, 4, blocks, 4, device="cpu", dtype=torch.float32)

    def test_manager_atomic_exhaustion_and_idempotent_release(self):
        manager = BlockManager(3)
        self.assertEqual(manager.allocate("A", 2), (0, 1))
        with self.assertRaises(CacheCapacityError):
            manager.allocate("B", 2)
        self.assertEqual(manager.blocks("A"), (0, 1))
        self.assertEqual(manager.blocks("B"), ())
        self.assertEqual(manager.free_count, 1)
        self.assertEqual(manager.allocate("B", 1), (2,))
        manager.release("A")
        manager.release("A")
        self.assertEqual(manager.allocate("C", 2), (0, 1))
        self.assertEqual(manager.free_count, 0)
        self.assertEqual(manager.blocks("B"), (2,))

    def test_noncontiguous_cross_block_tail_and_dirty_reuse(self):
        pool = self.small_pool()
        for t in pool.keys + pool.values:
            t.fill_(float("nan"))  # 错读尾块/别人的块会显式暴露。
        a, b = pool.new_cache(12), pool.new_cache(8)
        append(a, [1, 2, 3])
        append(b, [71, 72, 73, 74])
        key, value = append(a, [4, 5, 6])
        self.assertEqual(a.block_table, (0, 2))
        self.assertEqual(key[0, 0, :, 0].tolist(), [101, 102, 103, 104, 105, 106])
        torch.testing.assert_close(value, key + 1000)
        state = pool.snapshot()
        self.assertEqual(state["committed_tokens"], 10)
        self.assertEqual(state["unused_allocated_slots"], 2)
        self.assertEqual(state["pool_reserved_bytes"], 2 * 2 * 3 * 4 * 2 * 4 * 4)
        a.reset()
        self.assertEqual(a.last_request_stats["peak_valid_tokens"], 6)
        self.assertEqual(a.last_request_stats["peak_gather_pair_bytes"], 2 * 6 * 2 * 4 * 4)
        c = pool.new_cache(4)
        fresh, _ = append(c, [9, 10])
        self.assertEqual(c.block_table, (0,))
        self.assertEqual(fresh[0, 0, :, 0].tolist(), [109, 110])
        # B 的块在 A/C 操作后仍能正确读取与续写。
        b_key, _ = append(b, [75])
        self.assertEqual(b_key[0, 0, :, 0].tolist(), [171, 172, 173, 174, 175])
        for cache in (a, b, c):
            cache.close()
        self.assertEqual(pool.snapshot()["allocated_blocks"], 0)
        self.assertEqual(pool.snapshot()["request_handles"], 0)

    def test_pool_exhaustion_preserves_committed_prefix(self):
        pool = self.small_pool(2)
        a, b = pool.new_cache(8), pool.new_cache(4)
        append(a, [1, 2, 3, 4])
        append(b, [8])
        with self.assertRaises(CacheCapacityError):
            a.begin_step(1)
        self.assertEqual((a.length, a.block_table, pool.manager.free_count), (4, (0,), 0))
        b.close()
        key, _ = append(a, [5])
        self.assertEqual(key[0, 0, :, 0].tolist(), [101, 102, 103, 104, 105])
        a.close()

    def test_transaction_guards_and_closed_handle(self):
        pool = self.small_pool()
        cache = pool.new_cache(4)
        cache.begin_step(1)
        k = torch.zeros(1, 2, 1, 4)
        with self.assertRaises(RuntimeError):
            cache.view_layer(0)
        with self.assertRaises(ValueError):
            cache.write_layer(0, k.double(), k.double())
        cache.write_layer(0, k, k)
        with self.assertRaises(ValueError):
            cache.write_layer(0, k, k)
        with self.assertRaises(RuntimeError):
            cache.commit_step()
        cache.write_layer(1, k, k)
        cache.commit_step()
        self.assertEqual(cache.length, 1)
        with self.assertRaises(ValueError):
            cache.begin_step(4)
        cache.close()
        cache.close()
        with self.assertRaises(RuntimeError):
            cache.begin_step(1)
        self.assertEqual(pool.manager.free_count, 3)

    def test_interleaved_requests_match_m1_logits(self):
        model = tiny_model()
        pool = PagedKVPool.for_model(model, num_blocks=5, block_size=4)
        caches = {name: pool.new_cache(16) for name in ("A", "B")}
        refs = {name: model.new_cache(16) for name in ("A", "B")}
        for name, tokens in (("A", [1, 4, 6]), ("B", [7, 3, 2, 5, 1]),
                             ("A", [8, 9, 10]), ("B", [4])):
            ids = torch.tensor([tokens])
            actual = model(ids, caches[name], last_only=False)
            expected = model(ids, refs[name], last_only=False)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        self.assertEqual(caches["A"].block_table, (0, 3))
        caches["A"].close()
        ids = torch.tensor([[8]])
        torch.testing.assert_close(model(ids, caches["B"]), model(ids, refs["B"]), atol=1e-5, rtol=1e-4)
        caches["B"].close()
        self.assertEqual(pool.snapshot()["allocated_blocks"], 0)

    def test_layer_exception_poison_and_owner_only_cleanup(self):
        model = tiny_model()
        pool = PagedKVPool.for_model(model, 4, 4)
        a, b = pool.new_cache(12), pool.new_cache(12)
        model(torch.tensor([[1, 2]]), b)
        with patch.object(model.model.layers[1], "forward", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                model(torch.tensor([[3, 4]]), a)
        with self.assertRaises(RuntimeError):
            a.begin_step(1)
        a.close()
        self.assertEqual(pool.snapshot()["allocated_blocks"], 1)
        torch.testing.assert_close(model(torch.tensor([[5]]), b), model(torch.tensor([[1, 2, 5]])), atol=1e-5, rtol=1e-4)
        b.close()

    def test_runner_generation_eos_callback_and_exhaustion_cleanup(self):
        model = tiny_model()
        pool = PagedKVPool.for_model(model, 3, 4)
        other = pool.new_cache(12)
        model(torch.tensor([[7]]), other)
        cache = pool.new_cache(12)
        runner = ModelRunner(model, cache=cache)
        ids = torch.tensor([[1, 4, 6]])
        expected = ModelRunner(model, 12).generate(ids, 4, fixed_length=True)
        self.assertEqual(runner.generate(ids, 4, fixed_length=True), expected)
        self.assertEqual(pool.snapshot()["allocated_blocks"], 1)
        self.assertEqual(cache.last_request_stats["peak_blocks"], 2)
        eos = int(model(ids).argmax(-1).item())
        self.assertEqual(runner.generate(ids, 4, eos_token_id=eos), [eos])
        def fail(_):
            raise RuntimeError("callback")
        with self.assertRaisesRegex(RuntimeError, "callback"):
            runner.generate(ids, 4, on_token=fail)
        with self.assertRaises(CacheCapacityError):
            runner.generate(ids, 9, fixed_length=True)  # 只有两块供本请求，超过 8 个 KV token 时失败。
        self.assertEqual(pool.snapshot()["allocated_blocks"], 1)
        self.assertEqual(runner.generate(ids, 4, fixed_length=True), expected)
        cache.close()
        other.close()
        self.assertEqual(pool.snapshot()["allocated_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
