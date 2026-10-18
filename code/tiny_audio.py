# -*- coding: utf-8 -*-
"""查听觉那一条不过的通道：逐级打印路上各级与前额叶组的数值。"""
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
BASE = {"audio": np.full(sp.n_audio, 0.34), "touch": np.full(sp.n_touch, 0.34),
        "balance": np.full(sp.n_balance, 0.34)}


def run(sig):
    ui.set_image(np.zeros((sp.px, sp.px, sp.n_rgb), np.float32))
    ui.set_signals(**sig)
    B.act[:] = 0.0
    for _ in range(120):
        B.step(external=ui.drive())
    return B.act.copy()


a0 = run(BASE)
print(f"gain={B.gain:.3f}")
for c in range(sp.n_audio):
    s1 = {k: v.copy() for k, v in BASE.items()}
    s1["audio"] = s1["audio"].copy(); s1["audio"][c] = 0.67
    a1 = run(s1)
    print(f"\n听觉 第 {c} 号通道动一档：")
    for nm in ("audio_L4", "hop_audio_L4_0", "hop_audio_L4_1", "pfc_audio_L4"):
        g = B.ids_of(nm)
        if len(g) == 0:
            continue
        thr = np.asarray(B.thr)[g]
        dd = np.abs(a1[g] - a0[g]).max()
        print(f"   {nm:16s} n={len(g):3d} 活动 {a0[g].mean():.3f} "
              f"驱动/权重和 {np.median(np.bincount(B.dst, weights=B.sgn*B.w*B.act[B.src], minlength=B.n)[g] / np.maximum(sw[g],1e-9)):.3f}"
              f" 门槛/权重和 {np.median(thr / np.maximum(sw[g], 1e-9)):.3f}"
              f" 变化 {dd:.3f}")
