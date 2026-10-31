"""单请求连续 KV：层内写入，整次 forward 成功后统一提交长度。"""

import torch


class ContiguousKVCache:
    def __init__(self, num_layers, num_kv_heads, capacity, head_dim, *, device, dtype):
        if any(type(n) is not int or n <= 0 for n in
               (num_layers, num_kv_heads, capacity, head_dim)):
            raise ValueError("缓存维度必须为正整数。")
        self.capacity = capacity
        shape = (1, num_kv_heads, capacity, head_dim)
        self.keys = [torch.empty(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.values = [torch.empty_like(k) for k in self.keys]
        self.reset()

    def reset(self):
        """旧槽位无需清零：读者只能访问本次重新写入的有效区。"""
        self.length = 0
        self._step = None
        self._written = set()
        self._failed = False

    def begin_step(self, q_len):
        if self._failed or self._step is not None:
            raise RuntimeError("缓存失败或仍有未提交步骤，请先 reset。")
        if type(q_len) is not int or q_len <= 0 or self.length + q_len > self.capacity:
            raise ValueError("输入长度无效或超过 KV 容量。")
        self._step = (self.length, self.length + q_len)
        self._written = set()

    def write_layer(self, layer_idx, key, value):
        if self._failed or self._step is None:
            raise RuntimeError("先 begin_step，再写入缓存。")
        if not 0 <= layer_idx < len(self.keys) or layer_idx in self._written:
            raise ValueError("层号无效或本层重复写入。")
        start, end = self._step
        target = self.keys[layer_idx]
        expected = (1, target.shape[1], end - start, target.shape[3])
        for tensor in (key, value):
            if tuple(tensor.shape) != expected or tensor.dtype != target.dtype or tensor.device != target.device:
                raise ValueError("K/V 的 shape、dtype 或设备与缓存不一致。")
        self.keys[layer_idx][:, :, start:end, :].copy_(key)
        self.values[layer_idx][:, :, start:end, :].copy_(value)
        self._written.add(layer_idx)

    def view_layer(self, layer_idx):
        if self._failed or self._step is None or layer_idx not in self._written:
            raise RuntimeError("只能读取本步已写入的层。")
        end = self._step[1]
        return self.keys[layer_idx][:, :, :end, :], self.values[layer_idx][:, :, :end, :]

    def commit_step(self):
        if self._failed or self._step is None or len(self._written) != len(self.keys):
            raise RuntimeError("全部层写入成功后才能提交。")
        self.length = self._step[1]
        self._step = None
        self._written = set()

    def abort_step(self):
        """不尝试恢复半写入缓存，调用者必须 reset 或丢弃。"""
        self._failed = True
        self._step = None
        self._written = set()

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)
