# -*- coding: utf-8 -*-
"""诊断"路上中继 + 前额叶组"这一段：活动是否饱和、驱动/门槛差多少、变化还剩多少。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402

sp = Spec(px=2, n_rgb=3, n_level=3, n_audio=2, n_touch=3, n_balance=2,
          n_muscle=2, n_force=3, k_near=8, layer_ratio=(2.0, 4.0, 6.0),
          layer_floor=(16, 24, 32))
B, ids, sp, geo = build(sp, verbose=False)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)
P, C = sp.px, sp.n_rgb
fwd = np.array([not B.region[int(x)].startswith("ret_") for x in B.src])
sw = np.bincount(B.dst[fwd], weights=B.w[fwd], minlength=B.n)


def run(v):
    ui.set_image(v)
    ui.set_signals(audio=np.zeros(sp.n_audio), touch=np.zeros(sp.n_touch),
                   balance=np.zeros(sp.n_balance))
    B.act[:] = 0.0
    for _ in range(120):
        B.step(external=ui.drive())
    return B.act.copy()


base = np.full((P, P, C), 0.34, np.float32)
chg = base.copy(); chg[0, 0, 0] = 0.67
a0, a1 = run(base), run(chg)
d = np.bincount(B.dst, weights=B.sgn * B.w * B.act[B.src], minlength=B.n)
print(f"gain={B.gain:.3f}")
for nm in ("vision_L4", "hop_vision_L4_0", "hop_vision_L4_1", "pfc_vision_L4"):
    g = B.ids_of(nm)
    if len(g) == 0:
        continue
    thr = np.asarray(B.thr)[g]
    sat = float((a0[g] > 0.98).mean()); dead = float((a0[g] < 0.02).mean())
    dd = np.abs(a1[g] - a0[g])
    print(f"{nm:16s} n={len(g):3d} 活动 {a0[g].mean():.3f}（饱和 {sat:.2f} 死 {dead:.2f}）"
          f" 驱动/权重和 {np.median(d[g] / np.maximum(sw[g], 1e-9)):.3f}"
          f" 门槛/权重和 {np.median(thr / np.maximum(sw[g], 1e-9)):.3f}"
          f" 变化最大 {dd.max():.3f} 变化>0.1 的 {int((dd > 0.1).sum())}")
