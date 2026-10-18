# -*- coding: utf-8 -*-
"""小模型：查视觉第3→4层为什么死（逐级打印数值）。"""
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
fwd = np.array([not B.region[int(x)].startswith("ret_") for x in B.src])
sw = np.bincount(B.dst[fwd], weights=B.w[fwd], minlength=B.n)
deg = np.bincount(B.dst[fwd], minlength=B.n)

P, C = sp.px, sp.n_rgb
ui.set_image(np.full((P, P, C), 0.33, np.float32))
ui.set_signals(audio=np.zeros(sp.n_audio), touch=np.zeros(sp.n_touch),
               balance=np.zeros(sp.n_balance))
B.act[:] = 0.0
for _ in range(120):
    B.step(external=ui.drive())
d = np.bincount(B.dst, weights=B.sgn * B.w * B.act[B.src], minlength=B.n)
print(f"增益 gain={B.gain:.3f}")
for nm in ("vision_L1on", "vision_L2", "hop_vision_L2_0", "hop_vision_L2_1",
           "vision_L3", "hop_vision_L3_0", "vision_L4", "pfc_vision_L4"):
    g = B.ids_of(nm)
    if len(g) == 0:
        continue
    thr = np.asarray(B.thr)[g]
    print(f"{nm:16s} n={len(g):3d} 入边{int(np.median(deg[g])):3d} "
          f"活动 {B.act[g].mean():.3f} 活跃{float((B.act[g] > 0.1).mean()):.2f} "
          f"门槛/权重和 {np.median(thr / np.maximum(sw[g], 1e-9)):.3f} "
          f"驱动/权重和 {np.median(d[g] / np.maximum(sw[g], 1e-9)):.3f}")
