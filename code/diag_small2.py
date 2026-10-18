    # -*- coding: utf-8 -*-
"""看小区的链条到底卡在哪一级：逐级打印 密度 / 入边条数 / 门槛比例 / 驱动比例。"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402

sp = Spec()
B, ids, sp, geo = build(sp, verbose=False)
ui = Interface(B, ids, sp)
from calib import ref_calibrate
ref_calibrate(B, ids, ui)
fwd = np.array([not B.region[int(x)].startswith("ret_") for x in B.src])
sw = np.bincount(B.dst[fwd], weights=B.w[fwd], minlength=B.n)
deg = np.bincount(B.dst[fwd], minlength=B.n)

ui.set_values(0.5)
ui.set_signals(audio=np.full(sp.n_audio, 0.5), touch=np.full(sp.n_touch, 0.5),
               balance=np.full(sp.n_balance, 0.5))
B.act[:] = 0.0
for _ in range(140):
    B.step(external=ui.drive())
d = np.bincount(B.dst, weights=B.sgn * B.w * B.act[B.src], minlength=B.n)
print("（参考输入 全部 0.5）")
for nm in ("audio_L1on", "audio_L2", "audio_L3", "audio_L4",
           "hop_audio_L2_0", "hop_audio_L3_0"):
    g = B.ids_of(nm)
    if len(g) == 0:
        continue
    thr = np.asarray(B.thr)[g]
    print(f"{nm:16s} n={len(g):4d} 活动均值 {B.act[g].mean():.3f} "
          f"活跃(>0.1) {float((B.act[g] > 0.1).mean()):.2f} "
          f"入边 {int(np.median(deg[g])):3d} "
          f"门槛/权重和 {np.median(thr/np.maximum(sw[g],1e-9)):.3f} "
          f"驱动/权重和 {np.median(d[g]/np.maximum(sw[g],1e-9)):.3f}")
