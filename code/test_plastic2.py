# -*- coding: utf-8 -*-
"""常用 vs 少用 vs 不用：在"周期整体降权 + 用进"下的权重。

每一拍**只让一对变化**：
  常用对 (0灭 → 1亮)：每拍都出现
  少用对 (2灭 → 3亮)：每 100 拍出现一次
  不用对 (4常亮 → 5常灭)：永远不满足"刚变化"
"""
import numpy as np

from plastic2 import Plastic2

N, TICKS = 8, 4000
pl = Plastic2()
pl.enabled = True

for t in range(TICKS):
    prev = np.zeros(N)
    cur = np.zeros(N)
    prev[0] = 1.0; cur[1] = 1.0                  # 常用：灭0 → 亮1
    if t % 100 == 0:
        prev[2] = 1.0; cur[3] = 1.0              # 少用
    prev[4] = 1.0; cur[4] = 1.0                  # 常亮（不参与）
    cur[5] = 0.0                                 # 常灭（不参与）
    pl.tick(prev, cur)

for name, key in (("常用（每拍）", (0, 1)), ("少用（每100拍）", (2, 3)),
                  ("不用（常亮→常灭）", (4, 5))):
    print(f"  {name:20s} 权重 {pl.edges.get(key, 0.0):8.4f}")
print(f"可塑边总数 {len(pl.edges)}；最大权重 "
      f"{max(pl.edges.values()) if pl.edges else 0.0:.3f}（无上限）")
print(f"降权 {TICKS // pl.cfg.T} 次，每次 ×{pl.cfg.gamma}")
