"""block_size=4 的块表演示：不导入 torch、不需要模型或 NPU。"""

import json

from mini_llm_ascend.cache.block_manager import BlockManager


def demo():
    manager = BlockManager(3)
    events = []
    for owner, count in (("A", 1), ("B", 1), ("A", 1)):
        manager.allocate(owner, count)
        events.append({"action": "allocate " + owner, "A": list(manager.blocks("A")),
                       "B": list(manager.blocks("B")), "free_blocks": manager.free_count})
    events.append({"A_logical_positions": list(range(6)),
                   "A_physical_slots": [manager.blocks("A")[p // 4] * 4 + p % 4 for p in range(6)]})
    manager.release("A")
    manager.allocate("C", 1)
    events.append({"action": "release A, allocate C", "B": list(manager.blocks("B")),
                   "C": list(manager.blocks("C")), "free_blocks": manager.free_count})
    return events


if __name__ == "__main__":
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
