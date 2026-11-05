"""M3 必要验证：合批数学、调度生命周期与失败回收。"""

import unittest
from unittest.mock import patch
import torch

from tests.test_paged_kv import tiny_model
from mini_llm_ascend.cache.paged_kv import PagedKVPool
from mini_llm_ascend.engine.batch_runner import forward_batch
from mini_llm_ascend.engine.model_runner import ModelRunner
from mini_llm_ascend.engine.scheduler import Scheduler


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = tiny_model()
        self.pool = PagedKVPool.for_model(self.model, 12, 4)
        self.scheduler = Scheduler(self.model, self.pool, 2)

    def tearDown(self):
        self.scheduler.close()
        self.assertEqual(self.pool.snapshot()['allocated_blocks'], 0)
        self.assertEqual(self.pool.snapshot()['request_handles'], 0)

    def test_packed_prefill_and_mixed_decode_logits(self):
        caches = [self.pool.new_cache(16) for _ in range(2)]
        refs = [self.model.new_cache(16) for _ in range(2)]
        try:
            for inputs in ([torch.tensor([[1,2,3]]), torch.tensor([[5,6]])],
                           [torch.tensor([[7]]), torch.tensor([[8,9,10]])]):
                expected = torch.cat([self.model(x,c) for x,c in zip(inputs,refs)])
                actual = forward_batch(self.model, inputs, caches)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        finally:
            for cache in caches: cache.close()

    def test_dynamic_join_and_refill_matches_serial(self):
        s = self.scheduler
        a = s.submit('a', torch.tensor([[1,2,3]]), 4, fixed_length=True)
        self.assertEqual([e.request_id for e in s.step()], ['a'])
        b = s.submit('b', torch.tensor([[7,8]]), 1, fixed_length=True)
        c = s.submit('c', torch.tensor([[4,5,6]]), 2, fixed_length=True)
        self.assertEqual([e.request_id for e in s.step()], ['a','b'])
        self.assertEqual(c.status, 'waiting')
        self.assertEqual([e.request_id for e in s.step()], ['a','c'])
        while s.has_pending: s.step()
        for r in (a,b,c):
            expected = ModelRunner(self.model,16).generate(r.input_ids,r.max_new_tokens,fixed_length=True)
            self.assertEqual(r.output_ids, expected)
            self.assertEqual(r.status,'finished')

    def test_cancel_waiting_running_and_close(self):
        s = self.scheduler
        for name in ('a','b','c'): s.submit(name,torch.tensor([[1,2]]),3)
        s.step()
        self.assertTrue(s.cancel('a'))
        self.assertTrue(s.cancel('c'))
        self.assertFalse(s.cancel('a'))
        s.close()
        s.close()
        self.assertFalse(s.has_pending)
        with self.assertRaises(RuntimeError): s.step()

    def test_eos_and_input_validation(self):
        ids = torch.tensor([[1,2]])
        token = int(self.model(ids).argmax(-1).item())
        r = self.scheduler.submit('a', ids, 4, eos_token_id=token)
        event = self.scheduler.step()[0]
        self.assertTrue(event.finished)
        self.assertEqual(r.finish_reason,'eos')
        for name,x,budget in [('a',ids,2),('b',ids,0),('c',torch.tensor([[999]]),2),('d',ids,100)]:
            with self.assertRaises(ValueError): self.scheduler.submit(name,x,budget)

    def test_credit_prevents_exhaustion_and_fifo_progress(self):
        pool = PagedKVPool.for_model(self.model,2,4)
        s = Scheduler(self.model,pool,2)
        try:
            a = s.submit('a',torch.tensor([[1,2,3]]),4,fixed_length=True)
            b = s.submit('b',torch.tensor([[7]]),2,fixed_length=True)
            s.step()
            self.assertEqual(b.status,'waiting')
            while s.has_pending: s.step()
            self.assertEqual((a.status,b.status),('finished','finished'))
        finally: s.close()
        self.assertEqual(pool.snapshot()['request_handles'],0)

    def test_layer_failure_releases_batch_and_preserves_queue(self):
        s = self.scheduler
        for name in ('a','b','c'): s.submit(name,torch.tensor([[1,2]]),2)
        with patch.object(self.model.model.layers[1].mlp,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'): s.step()
        self.assertEqual(s.requests['a'].status,'failed')
        self.assertEqual(s.requests['b'].output_ids,[])
        self.assertEqual(s.requests['c'].status,'waiting')
        self.assertEqual(self.pool.snapshot()['allocated_blocks'],0)
        while s.has_pending: s.step()
        self.assertEqual(s.requests['c'].status,'finished')


if __name__ == '__main__':
    unittest.main()
