# -*- coding: utf-8 -*-
"""把可塑边的真实键与权重直接打出来（不再猜键）。"""
import numpy as np

from plastic2 import Plastic2

N, TICKS = 8, 4000
pl = Plastic2()
pl.enabled = True
for t in range(TICKS):
    prev = np.zeros(N); cur = np.zeros(N)
    prev[0] = 1.0; cur[1] = 1.0
    if t % 100 == 0:
        prev[2] = 1.0; cur[3] = 1.0
    prev[4] = 1.0; cur[4] = 1.0
    pl.tick(prev, cur)

print("真实的可塑边（键 → 权重），按权重降序：")
for k, w in sorted(pl.edges.items(), key=lambda kv: -kv[1]):
    print(f"   {k} → {w:.4f}")
print(f"共 {len(pl.edges)} 条")
