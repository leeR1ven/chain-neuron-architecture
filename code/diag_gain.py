# -*- coding: utf-8 -*-
"""同一感受野(k=16)，只改"门槛/带宽比例"，看信号能不能一直传到前额叶。

机制：每层都在稀释"活跃神经元密度"（第2层约 1/4 活着，第3层再从里面取 16 个就凑不满门槛）。
      门槛比例要低于上游的活跃密度，链条才不会一级级死掉。

做法：连线照旧，构建完直接把各层/路上各级/前额叶组/返回中继的门槛重设成
      门槛 = f × 入边权重和，带宽 = g × 入边权重和（就是"纯连线定门槛"）。
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402

PAT = re.compile(r"(_L\d+$|^hop_|^pfc_|^ret_)")
P, C = 10, 3
sp = Spec(k_near=16)
B, ids, _, _ = build(sp, verbose=False)
ui = Interface(B, ids, sp)
sw = np.bincount(B.dst, weights=B.w, minlength=B.n)

gset = [i for i in range(B.n) if PAT.search(B.region[i])]
gset = np.array(gset)
q = max(2, sp.px // 3)
z = np.zeros((sp.px, sp.px, C), np.float32)
img1 = z.copy(); img1[0:q, 0:q, 0] = 0.8
img2 = z.copy(); img2[sp.px - q:, sp.px - q:, 0] = 0.8


def run(im):
    ui.set_image(im)
    ui.set_signals(audio=np.zeros(sp.n_audio), touch=np.zeros(sp.n_touch),
                   balance=np.zeros(sp.n_balance))
    B.act[:] = 0.0
    for _ in range(140):
        B.step(external=ui.drive())
    return B.act.copy()


for f, g in ((0.25, 0.45), (0.15, 0.35), (0.10, 0.30), (0.06, 0.20)):
    thr = np.asarray(B.thr, dtype=np.float64).copy()
    soft = np.asarray(B.soft, dtype=np.float64).copy()
    thr[gset] = f * sw[gset]
    soft[gset] = np.maximum(1e-4, g * sw[gset])
    B.thr = thr.tolist()
    B.soft = soft.tolist()
    r1, r2 = run(img1), run(img2)
    print(f"\n门槛 {f}×权重和、带宽 {g}×权重和")
    for key in ("vision_L2", "vision_L3", "vision_L4", "pfc_vision_L4"):
        gid = ids[key]
        d = np.abs(r1[gid] - r2[gid])
        print(f"   {key:16s} 活动 {r1[gid].mean():.3f}→{r2[gid].mean():.3f}  "
              f"最大差 {d.max():.3f}  差>0.02 的 {int((d > 0.02).sum()):5d}")
