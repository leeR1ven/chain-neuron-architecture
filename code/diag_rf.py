# -*- coding: utf-8 -*-
"""只改感受野大小（门槛仍由连线比例定，不标定），看"色块位置"能不能被分辨。

预期：感受野越小，局部图案在感受野里占的比例越大 → 越容易分辨出"哪里有东西"。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402

P, C = 10, 3


def probe(k):
    sp = Spec(k_near=k)
    B, ids, _, _ = build(sp, verbose=False)
    ui = Interface(B, ids, sp)
    q = max(2, sp.px // 3)

    def run(im):
        ui.set_image(im)
        ui.set_signals(audio=np.zeros(sp.n_audio), touch=np.zeros(sp.n_touch),
                       balance=np.zeros(sp.n_balance))
        B.act[:] = 0.0
        for _ in range(140):
            B.step(external=ui.drive())
        return B.act.copy()

    z = np.zeros((sp.px, sp.px, C), np.float32)
    a1 = z.copy(); a1[0:q, 0:q, 0] = 0.8
    a2 = z.copy(); a2[sp.px - q:, sp.px - q:, 0] = 0.8
    r1, r2 = run(a1), run(a2)
    out = []
    for key in ("vision_L2", "vision_L3", "vision_L4", "pfc_vision_L4"):
        g = ids[key]
        d = np.abs(r1[g] - r2[g])
        out.append((key, r1[g].mean(), d.max(), int((d > 0.02).sum()), len(g)))
    return out


for k in (8, 16, 32, 64):
    print(f"\n感受野 k={k}（左上块红 vs 右下块红）")
    for key, m, mx, cnt, n in probe(k):
        print(f"   {key:16s} n={n:5d} 活动均值 {m:.3f}  最大差 {mx:.3f}  "
              f"差>0.02 的 {cnt:5d}")
