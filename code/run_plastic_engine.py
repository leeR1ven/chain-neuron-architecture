# -*- coding: utf-8 -*-
"""可塑连接接进真引擎：喂一串随机画面，一边跑一边长连接。
检查：① 长出了多少条边 ② 有没有边连到输入层（那些"总是亮"的细胞）——应为 0
     ③ 权重分布（无上限）④ 每步耗时（实时性）"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Config  # noqa: E402
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402
from plastic2 import Plastic2  # noqa: E402

sp = Spec(px=10, n_rgb=3, n_level=10, n_audio=20, n_touch=40, n_balance=6,
          n_muscle=8, n_force=10, k_near=8, layer_ratio=(2.0, 4.0, 6.0),
          layer_floor=(128, 192, 256))
B, ids, sp, geo = build(sp, Config(), verbose=True)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)
pl = Plastic2()
pl.enabled = True
n0 = len(B.src)
rng = np.random.default_rng(7)
prev = B.act.copy()
t0 = time.perf_counter()
TICKS = 40
for t in range(TICKS):
    ui.set_image(rng.random((sp.px, sp.px, sp.n_rgb)).astype(np.float32))
    ui.set_signals(audio=rng.random(sp.n_audio).astype(np.float32),
                   touch=rng.random(sp.n_touch).astype(np.float32),
                   balance=rng.random(sp.n_balance).astype(np.float32))
    for _ in range(3):
        B.step(external=ui.drive())
        pl.tick(prev, B.act)
        prev = B.act.copy()
    pl.inject(B)
ms = (time.perf_counter() - t0) / (TICKS * 3) * 1000
added = len(B.src) - n0
# 输入层的细胞（pos/neg）：那些"总是亮"的
inp = ids["pos"]["vision"].ravel()
m = np.isin(B.src, inp) | np.isin(B.dst, inp)
print(f"\n可塑边新增 {added} 条（引擎连接 {n0:,} → {len(B.src):,}）")
print(f"其中连到输入层细胞的：{int(m.sum())} 条 —— 应为 0（输入层总是亮，不许参与）")
w = pl.edges.values()
if len(pl.edges):
    wa = np.array(list(w))
    print(f"可塑边权重：中位 {np.median(wa):.3f}  最大 {wa.max():.3f}（无上限）")
print(f"每步耗时 {ms:.2f} 毫秒（20 毫秒一拍内）")
