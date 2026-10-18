# -*- coding: utf-8 -*-
"""测"顺序有没有被连接记住"——比较法（不加任何机制）：

  做法：先给 A 几拍，再切到 B，记录切换之后前几拍"图案有多像 B"。
  对比：学习之前 vs 反复看 A→B→C→D→E 之后。
  若学习后"像 B"爬得更快/更高，说明 A→B 的连接把顺序记住了。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Config  # noqa: E402
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402
from plastic2 import Plastic2  # noqa: E402

P, C, L, NF, Q = 10, 3, 10, 5, 3
sp = Spec(px=P, n_rgb=3, n_level=L, n_audio=20, n_touch=40, n_balance=6,
          n_muscle=8, n_force=10, k_near=8, layer_ratio=(2.0, 4.0, 6.0),
          layer_floor=(128, 192, 256))
B, ids, sp, geo = build(sp, Config(), verbose=False)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)
SPOTS = [(0, 0), (0, P - Q), (P // 2 - 1, P // 2 - 1), (P - Q, 0), (P - Q, P - Q)]
COLS = [0, 1, 2, 0, 1]
BANDS = [2, 6, 10, 14, 18]


def ext_of(k):
    im = np.full((P, P, C), 0.45, np.float32)
    r, c = SPOTS[k]
    im[r:r + Q, c:c + Q, :] = 0.15
    im[r:r + Q, c:c + Q, COLS[k]] = 0.95
    ui.set_image(im)
    au = np.full(sp.n_audio, 0.3); au[BANDS[k]] = 0.9
    ui.set_signals(audio=au, touch=np.full(sp.n_touch, 0.34),
                   balance=np.full(sp.n_balance, 0.34))
    return ui.drive()


def run_frame(k, steps=16):
    ext = ext_of(k)
    B.act[:] = 0.0
    for _ in range(steps):
        B.step(external=ext)
    return B.act.copy()


G = B.ids_of("vision_L2")
ref = np.array([run_frame(k, 24)[G].copy() for k in range(NF)])
M = ref.mean(axis=0)
Cn = (ref - M)
Cn = Cn / (np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-9)


def priming(a_idx, b_idx, a_ticks=6, b_ticks=6):
    """先给 A 若干拍，再切到 B，返回切换后各拍'像 B'的相似度"""
    ea, eb = ext_of(a_idx), ext_of(b_idx)
    B.act[:] = 0.0
    for _ in range(a_ticks):
        B.step(external=ea)
    out = []
    for _ in range(b_ticks):
        B.step(external=eb)
        v = B.act[G].copy() - M
        sim = Cn @ (v / (np.linalg.norm(v) + 1e-9))
        out.append(float(sim[b_idx]))
    return out, float(sim[b_idx])


before, _ = priming(0, 1)
print("学习前：先看第1帧再看第2帧 → 切换后每拍'像第2帧'的相似度")
print("   " + "  ".join(f"{v:.2f}" for v in before))

pl = Plastic2()
pl.enabled = True
pl.cfg.eta = 0.05
for rep in range(40):
    prev = B.act.copy()
    for k in range(NF):
        a = run_frame(k)
        pl.tick(prev, a)
        prev = a.copy()
    pl.inject(B)
print(f"看完 40 遍序列：可塑边表 {len(pl.edges)} 条，引擎连接 {len(B.src):,}")

after, _ = priming(0, 1)
print("学习后：先看第1帧再看第2帧 → 切换后每拍'像第2帧'的相似度")
print("   " + "  ".join(f"{v:.2f}" for v in after))
print(f"\n对比：切换后第 1 拍 {before[0]:.2f} → {after[0]:.2f}；"
      f"第 3 拍 {before[2]:.2f} → {after[2]:.2f}（学习后更高=顺序被记住了）")
