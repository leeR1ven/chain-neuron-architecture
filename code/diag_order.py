# -*- coding: utf-8 -*-
"""按顺序从输入层往上查：一次输入的小变化，逐层还剩多少。

同时并排比两种感受野（每个神经元接上一层 24 个 vs 64 个），
看"接得越多是不是越好"——不预设结论，用数据说话。

测的输入变化：整幅灰度 0.50 → 0.55（半档）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402

LAYERS = ("pos(输入编码)", "vision_L1on", "vision_L1off", "vision_L2", "vision_L3",
          "vision_L4", "pfc_vision_L4", "audio_L2", "touch_L2", "balance_L2")


def probe(k):
    sp = Spec(k_near=k)
    B, ids, sp, geo = build(sp, verbose=False)
    ui = Interface(B, ids, sp)
    ref_calibrate(B, ids, ui)   # 单轮、所有组同一状态

    def run(v):
        ui.set_values(v)
        ui.set_signals(audio=np.full(sp.n_audio, 0.5), touch=np.full(sp.n_touch, 0.5),
                       balance=np.full(sp.n_balance, 0.5))
        B.act[:] = 0.0
        for _ in range(140):
            B.step(external=ui.drive())
        return B.act.copy()

    a, b = run(0.50), run(0.55)
    out = {}
    for nm in LAYERS:
        g = ids["pos"]["vision"].ravel() if nm.startswith("pos") else ids.get(nm)
        if g is None:
            g = B.ids_of(nm)
        d = np.abs(a[g] - b[g])
        out[nm] = (len(g), a[g].mean(), b[g].mean(), d.max(),
                   int((d > 0.02).sum()))
    return B.n, len(B.src), out


print(f"{'层':18s} {'个数':>6s} | k=24: 0.50→0.55 活动 · 最大差 · 个数"
      f" | k=64: 0.50→0.55 活动 · 最大差 · 个数")
res = {}
for k in (24, 64):
    res[k] = probe(k)
for nm in LAYERS:
    cells = []
    for k in (24, 64):
        n, m5, m6, mx, cnt = res[k][2][nm]
        cells.append(f"{m5:.3f}→{m6:.3f} {mx:.3f} {cnt:>5d}")
    n = res[24][2][nm][0]
    print(f"{nm:18s} {n:6d} | {cells[0]:36s} | {cells[1]}")
for k in (24, 64):
    print(f"k={k}: 总神经元 {res[k][0]}，连接 {res[k][1]:,}")
