"""共享分页池 + 每请求块表；gather 后复用 M1 的 dense 注意力。

这不是直接读取分页块的原生 attention kernel。池预留、请求占用和
gather 临时输出分开统计，避免把可复用空间说成实际内存已释放。
"""

import torch

from .block_manager import BlockManager


class PagedKVPool:
    def __init__(self, num_layers, num_kv_heads, head_dim, num_blocks, block_size, *, device, dtype):
        if any(type(n) is not int or n <= 0 for n in
               (num_layers, num_kv_heads, head_dim, num_blocks, block_size)):
            raise ValueError("分页池维度必须为正整数。")
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.manager = BlockManager(num_blocks)
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.keys = [torch.empty(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.values = [torch.empty_like(k) for k in self.keys]
        self.bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * self.keys[0].element_size()
        self._next_owner = 0
        self._lengths = {}

    @classmethod
    def for_model(cls, model, num_blocks, block_size):
        c = model.config
        weight = model.model.embed_tokens.weight
        return cls(c.num_hidden_layers, c.num_key_value_heads, c.head_dim,
                   num_blocks, block_size, device=weight.device, dtype=weight.dtype)

    def new_cache(self, capacity):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("请求 capacity 必须为正整数。")
        owner = self._next_owner
        self._next_owner += 1
        self._lengths[owner] = 0
        return PagedKVCache(self, owner, capacity)

    @property
    def nbytes(self):
        return self.manager.num_blocks * self.block_size * self.bytes_per_token

    def snapshot(self):
        allocated = self.manager.num_blocks - self.manager.free_count
        valid = sum(self._lengths.values())
        return {"num_blocks": self.manager.num_blocks, "block_size": self.block_size,
                "free_blocks": self.manager.free_count, "allocated_blocks": allocated,
                "peak_allocated_blocks": self.manager.peak_allocated_blocks,
                "pool_reserved_bytes": self.nbytes,
                "allocated_capacity_bytes": allocated * self.block_size * self.bytes_per_token,
                "committed_tokens": valid, "logical_valid_bytes": valid * self.bytes_per_token,
                "unused_allocated_slots": allocated * self.block_size - valid,
                "request_handles": len(self._lengths)}


class PagedKVCache:
    """一个请求的逻辑视图；不能共享给两个同时执行的 forward。"""

    def __init__(self, pool, owner, capacity):
        self.pool, self.owner, self.capacity = pool, owner, capacity
        self._closed = False
        self.last_request_stats = None
        self.reset()

    @property
    def keys(self):
        # M1 forward 用 len(cache.keys) 校验层数，不依赖具体存储布局。
        return self.pool.keys

    @property
    def block_table(self):
        return self.pool.manager.blocks(self.owner)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("缓存句柄已 close，请创建新句柄。")

    def begin_step(self, q_len):
        self._check_open()
        if self._failed or self._step is not None:
            raise RuntimeError("缓存失败或仍有未提交步骤，请先 reset。")
        if type(q_len) is not int or q_len <= 0 or self.length + q_len > self.capacity:
            raise ValueError("输入长度无效或超过请求 capacity。")
        end = self.length + q_len
        needed = (end + self.pool.block_size - 1) // self.pool.block_size
        # allocate 先检查容量；失败时此前已提交的前缀和块表不变。
        self.pool.manager.allocate(self.owner, needed - len(self.block_table))
        self._step = (self.length, end)
        self._written = set()
        self.peak_blocks = max(self.peak_blocks, len(self.block_table))
        try:
            # 只传有效槽，末块的旧数据不参与 attention；索引在层间复用。
            slots = [block * self.pool.block_size + offset
                     for block in self.block_table for offset in range(self.pool.block_size)][:end]
            self._slots = torch.tensor(slots, device=self.keys[0].device, dtype=torch.long)
            self.peak_slot_index_bytes = max(self.peak_slot_index_bytes, self._slots.numel() * self._slots.element_size())
        except BaseException:
            self.abort_step()
            raise

    def write_layer(self, layer_idx, key, value):
        self._check_open()
        if self._failed or self._step is None:
            raise RuntimeError("先 begin_step，再写入缓存。")
        if type(layer_idx) is not int or not 0 <= layer_idx < len(self.keys) or layer_idx in self._written:
            raise ValueError("层号无效或本层重复写入。")
        start, end = self._step
        expected = (1, self.pool.num_kv_heads, end - start, self.pool.head_dim)
        for tensor in (key, value):
            if tuple(tensor.shape) != expected or tensor.dtype != self.keys[0].dtype or tensor.device != self.keys[0].device:
                raise ValueError("K/V 的 shape、dtype 或设备与池不一致。")
        position = start
        while position < end:
            logical, offset = divmod(position, self.pool.block_size)
            count = min(end - position, self.pool.block_size - offset)
            physical = self.block_table[logical]
            source_start = position - start
            # 一个片段写一块；由 [H,Q,D] 转成物理布局 [Q,H,D]。
            for source, target in ((key, self.keys[layer_idx]), (value, self.pool.values[layer_idx])):
                part = source[0, :, source_start:source_start + count, :].transpose(0, 1).contiguous()
                target[physical, offset:offset + count].copy_(part)
            position += count
        self._written.add(layer_idx)

    def view_layer(self, layer_idx):
        self._check_open()
        if self._failed or self._step is None or layer_idx not in self._written:
            raise RuntimeError("只能读取本步已写入的层。")
        gathered = []
        for tensor in (self.keys[layer_idx], self.pool.values[layer_idx]):
            flat = tensor.view(-1, self.pool.num_kv_heads, self.pool.head_dim)
            selected = flat.index_select(0, self._slots)
            gathered.append(selected.transpose(0, 1).unsqueeze(0))
        pair_bytes = sum(t.numel() * t.element_size() for t in gathered)
        self.peak_gather_pair_bytes = max(self.peak_gather_pair_bytes, pair_bytes)
        return tuple(gathered)

    def commit_step(self):
        self._check_open()
        if self._failed or self._step is None or len(self._written) != len(self.keys):
            raise RuntimeError("全部层写入成功后才能提交。")
        self.length = self._step[1]
        self.pool._lengths[self.owner] = self.length
        self.peak_valid_tokens = max(self.peak_valid_tokens, self.length)
        self._step, self._slots = None, None
        self._written = set()

    def abort_step(self):
        self._check_open()
        self._failed = True
        self.pool._lengths[self.owner] = 0
        self._step, self._slots = None, None
        self._written = set()

    def reset(self):
        self._check_open()
        if self.block_table:
            self.last_request_stats = {
                "block_table": list(self.block_table), "peak_blocks": self.peak_blocks,
                "peak_allocated_capacity_bytes": self.peak_blocks * self.pool.block_size * self.pool.bytes_per_token,
                "peak_valid_tokens": self.peak_valid_tokens,
                "peak_logical_valid_bytes": self.peak_valid_tokens * self.pool.bytes_per_token,
                "peak_gather_pair_bytes": self.peak_gather_pair_bytes,
                "peak_slot_index_bytes": self.peak_slot_index_bytes,
                "poisoned": self._failed,
            }
        self.pool.manager.release(self.owner)
        self.pool._lengths[self.owner] = 0
        self.length = 0
        self._failed = False
        self._step, self._slots = None, None
        self._written = set()
        self.peak_blocks = self.peak_valid_tokens = 0
        self.peak_gather_pair_bytes = self.peak_slot_index_bytes = 0

    def close(self):
        if not self._closed:
            self.reset()
            self.pool._lengths.pop(self.owner, None)
            self._closed = True
