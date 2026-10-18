# -*- coding: utf-8 -*-
"""前额叶那组为什么收不到：在三个时刻分别打印它的活动，
   （a）刚建好+标定后（b）看完动画（注入可塑边）之后（c）喂第1帧时。"""
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


def run_frame(k, steps=12):
    ext = ext_of(k)
    B.act[:] = 0.0
    for _ in range(steps):
        B.step(external=ext)
    return B.act.copy()


def show(tag):
    for nm in ("vision_L4", "hop_vision_L4_0", "hop_vision_L4_1", "pfc_vision_L4"):
        g = B.ids_of(nm)
        print(f"   {tag:10s} {nm:18s} 活动均值 {B.act[g].mean():.3f} "
              f"活跃(>0.1) {float((B.act[g] > 0.1).mean()):.2f}")


run_frame(0, 20)
print("(a) 刚标定完，喂第1帧：")
show("a")

pl = Plastic2()
pl.enabled = True
pl.cfg.eta = 0.05
n0 = len(B.src)
for rep in range(30):
    prev = B.act.copy()
    for k in range(NF):
        a = run_frame(k)
        pl.tick(prev, a)
        prev = a.copy()
    pl.inject(B)
run_frame(0, 20)
print(f"(b) 看完 30 遍动画（连接 {n0:,} → {len(B.src):,}），再喂第1帧：")
show("b")

B.act[:] = 0.0
ext0 = ext_of(0)
for t in range(40):
    B.step(external=ext0)
print("(c) 连续喂第1帧 40 拍之后：")
show("c")
