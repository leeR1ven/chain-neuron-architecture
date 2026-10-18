# -*- coding: utf-8 -*-
"""扫描"按深度定的门槛比例"组合，找能让第2/3/4层**同时处于分级状态**的取值。

判据（不再靠反应反推）：
  · 活动均值落在 0.05~0.6 之间 = 分级
  · 0 或 1.0 = 死/饱和
  · 同时看"色块位置"能分辨的神经元个数
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402

sp = Spec()
B, ids, sp, geo = build(sp, verbose=False)
ui = Interface(B, ids, sp)
fwd = np.array([not B.region[int(x)].startswith("ret_") for x in B.src])
sw = np.bincount(B.dst[fwd], weights=B.w[fwd], minlength=B.n)   # 只算前向边
target = []
for i in range(B.n):
    r = B.region[i]
    if r.startswith("ret_"):
        continue
    if r.startswith(("hop_", "pfc_")) or re.search(r"_L\d+$", r):
        m = re.search(r"_L(\d)", r)
        if m:
            target.append((i, int(m.group(1))))
P, C, q = sp.px, sp.n_rgb, max(2, sp.px // 3)
z = np.zeros((P, P, C), np.float32)
i1 = z.copy(); i1[0:q, 0:q, 0] = 0.8
i2 = z.copy(); i2[P - q:, P - q:, 0] = 0.8


def run(im):
    ui.set_image(im)
    ui.set_signals(audio=np.zeros(sp.n_audio), touch=np.zeros(sp.n_touch),
                   balance=np.zeros(sp.n_balance))
    B.act[:] = 0.0
    for _ in range(140):
        B.step(external=ui.drive())
    return B.act.copy()


for cfg in ((0.25, 0.15, 0.10), (0.25, 0.22, 0.18), (0.30, 0.26, 0.22),
            (0.35, 0.30, 0.26)):
    thr = np.asarray(B.thr, dtype=np.float64).copy()
    soft = np.asarray(B.soft, dtype=np.float64).copy()
    for i, d in target:
        f = cfg[d - 1] if 1 <= d <= 3 else cfg[2]
        thr[i] = f * sw[i]
        soft[i] = max(1e-4, 3.0 * f * sw[i])
    B.thr = thr.tolist()
    B.soft = soft.tolist()
    a, b = run(i1), run(i2)
    print(f"\nL2/L3/L4 门槛比例 = {cfg}")
    for key in ("vision_L2", "vision_L3", "vision_L4"):
        g = ids[key]
        d = np.abs(a[g] - b[g])
        state = "死" if a[g].mean() < 0.01 else ("饱和" if a[g].mean() > 0.9 else "分级")
        print(f"   {key:12s} 活动 {a[g].mean():.3f} [{state}]  "
              f"位置差 最大 {d.max():.3f} 个数 {int((d > 0.02).sum()):5d}")
