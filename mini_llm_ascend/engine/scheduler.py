"""单线程 FIFO 连续批处理；缓存按需领取，准入信用按最大预算预留。"""

from collections import deque
from dataclasses import dataclass, field

import torch

from .batch_runner import forward_batch
from .sampler import greedy_select, normalize_eos


@dataclass
class Request:
    request_id: str
    input_ids: torch.Tensor
    max_new_tokens: int
    eos_ids: set
    fixed_length: bool
    credit: int
    output_ids: list = field(default_factory=list)
    status: str = "waiting"
    finish_reason: str = ""
    cache: object = None


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token_id: int
    finished: bool
    finish_reason: str


class Scheduler:
    def __init__(self, model, pool, max_batch_size=4):
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size 必须是正整数。")
        if pool.snapshot()["request_handles"]:
            raise ValueError("调度器需要独占空池。")
        self.model, self.pool = model.eval(), pool
        self.max_batch_size = max_batch_size
        self.requests, self.waiting, self.active = {}, deque(), []
        self.closed = False

    def submit(self, request_id, input_ids, max_new_tokens, *, eos_token_id=None, fixed_length=False):
        if self.closed:
            raise RuntimeError("调度器已关闭。")
        if not isinstance(request_id, str) or not request_id or request_id in self.requests:
            raise ValueError("请求 ID 必须是唯一非空字符串。")
        weight = self.model.model.embed_tokens.weight
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("输入必须是非空 [1,Q]。")
        if input_ids.device != weight.device or input_ids.dtype != torch.long:
            raise ValueError("输入设备或 dtype 错误。")
        if bool(((input_ids < 0) | (input_ids >= self.model.config.vocab_size)).any().item()):
            raise ValueError("token ID 超出词表。")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("输出预算必须是正整数。")
        capacity = input_ids.shape[1] + max_new_tokens - 1
        credit = (capacity + self.pool.block_size - 1) // self.pool.block_size
        if capacity > self.model.config.max_position_embeddings or credit > self.pool.manager.num_blocks:
            raise ValueError("单请求预算超出位置上限或整个分页池容量。")
        eos = normalize_eos(eos_token_id, self.model.config.vocab_size)
        if fixed_length and len(eos) == self.model.config.vocab_size:
            raise ValueError("屏蔽 EOS 后没有可选 token。")
        request = Request(request_id, input_ids.clone(), max_new_tokens, eos, fixed_length, credit)
        self.requests[request_id] = request
        self.waiting.append(request)
        return request

    def _finish(self, request, status, reason):
        if request.cache is not None:
            request.cache.close()
            request.cache = None
        request.status, request.finish_reason = status, reason

    def cancel(self, request_id):
        request = self.requests[request_id]
        if request.status not in ("waiting", "running"):
            return False
        self._finish(request, "cancelled", "cancelled")
        self.active = [r for r in self.active if r is not request]
        self.waiting = deque(r for r in self.waiting if r is not request)
        return True

    @property
    def has_pending(self):
        return bool(self.active or self.waiting)

    def step(self):
        if self.closed:
            raise RuntimeError("调度器已关闭。")
        try:
            credit = sum(r.credit for r in self.active)
            while self.waiting and len(self.active) < self.max_batch_size:
                request = self.waiting[0]
                if credit + request.credit > self.pool.manager.num_blocks:
                    break
                request.cache = self.pool.new_cache(request.input_ids.shape[1] + request.max_new_tokens - 1)
                request.status = "running"
                self.active.append(self.waiting.popleft())
                credit += request.credit
            if not self.active:
                return []
            inputs = [r.input_ids if not r.output_ids else
                      torch.tensor([[r.output_ids[-1]]], device=r.input_ids.device, dtype=torch.long)
                      for r in self.active]
            logits = forward_batch(self.model, inputs, [r.cache for r in self.active])
            tokens = [greedy_select(logits[i:i+1], r.eos_ids, suppress_eos=r.fixed_length)
                      for i, r in enumerate(self.active)]
            events = []
            for request, token in zip(self.active, tokens):
                request.output_ids.append(token)
                reason = ""
                if not request.fixed_length and token in request.eos_ids:
                    reason = "eos"
                elif len(request.output_ids) == request.max_new_tokens:
                    reason = "length"
                if reason:
                    self._finish(request, "finished", reason)
                events.append(TokenEvent(request.request_id, token, bool(reason), reason))
            self.active = [r for r in self.active if r.status == "running"]
            return events
        except BaseException:
            for request in self.active:
                self._finish(request, "failed", "batch_error")
            self.active.clear()
            raise

    def close(self):
        for request in list(self.active) + list(self.waiting):
            self._finish(request, "cancelled", "closed")
        self.active.clear()
        self.waiting.clear()
        self.closed = True
