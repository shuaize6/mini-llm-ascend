"""纯 Python 块所有权：单线程使用，申请不足时不修改已有状态。"""

from collections import deque


class CacheCapacityError(RuntimeError):
    """共享 KV 池的空闲块不足；排队/拒绝策略由上层决定。"""


class BlockManager:
    def __init__(self, num_blocks):
        if type(num_blocks) is not int or num_blocks <= 0:
            raise ValueError("num_blocks 必须为正整数。")
        self.num_blocks = num_blocks
        self._free = deque(range(num_blocks))
        self._owned = {}
        self.peak_allocated_blocks = 0

    @property
    def free_count(self):
        return len(self._free)

    def blocks(self, owner):
        # 返回不可变副本，调用者不能直接改动所有权表。
        return tuple(self._owned.get(owner, ()))

    def allocate(self, owner, count):
        hash(owner)
        if type(count) is not int or count < 0:
            raise ValueError("申请块数必须为非负整数。")
        if count > self.free_count:
            raise CacheCapacityError("需要 {} 块，但只有 {} 块空闲。".format(count, self.free_count))
        if count == 0:
            return ()
        acquired = tuple(self._free.popleft() for _ in range(count))
        self._owned.setdefault(owner, []).extend(acquired)
        self.peak_allocated_blocks = max(self.peak_allocated_blocks, self.num_blocks - self.free_count)
        return acquired

    def release(self, owner):
        # 按所有者一次释放，不接受任意块编号，重复释放不会制造重复空闲块。
        released = tuple(self._owned.pop(owner, ()))
        # 优先复用刚归还的块；串行请求可重复使用同一批物理位置。
        self._free.extendleft(reversed(released))
        return released
