# -*- coding: utf-8 -*-
"""顺序实验（v7）：帧改成"灰底 + 色块"（前额叶能收到），
   测法改成"先减掉五帧的共同部分，再比像谁"（否则共同部分把差别盖住）。"""
import os
import sys
import time

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
AREAS = ("vision_L2", "audio_L2", "pfc_vision_L4")
SPOTS = [(0, 0), (0, P - Q), (P // 2 - 1, P // 2 - 1), (P - Q, 0), (P - Q, P - Q)]
COLS = [0, 1, 2, 0, 1]
BANDS = [2, 6, 10, 14, 18]


def ext_of(k):
    im = np.full((P, P, C), 0.45, np.float32)        # 灰底（前额叶能收到的量级）
    r, c = SPOTS[k]
    im[r:r + Q, c:c + Q, :] = 0.15
    im[r:r + Q, c:c + Q, COLS[k]] = 0.95             # 该帧的颜色块
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


pl = Plastic2()
pl.enabled = True
pl.cfg.eta = 0.60          # 平时 0.05；这里调大，先排除"只是连接太弱"这一条
t0 = time.perf_counter()
for rep in range(150):
    prev = B.act.copy()
    for k in range(NF):
        a = run_frame(k)
        pl.tick(prev, a)
        prev = a.copy()
    pl.inject(B)
print(f"看 30 遍（{time.perf_counter() - t0:.0f} 秒）：可塑边表 {len(pl.edges)} 条")

ref = {nm: np.array([run_frame(k, 16)[B.ids_of(nm)].copy() for k in range(NF)])
       for nm in AREAS}
print("\n每帧各区的活动均值")
for nm in AREAS:
    print(f"   {nm:14s} " + "  ".join(f"{r.mean():.3f}" for r in ref[nm]))
print("\n五帧图案：原始相似度 / 减掉共同部分后的相似度（后者越接近 0 越分得开）")
for nm in AREAS:
    R = ref[nm]
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    C_ = R - R.mean(axis=0, keepdims=True)
    Cn = C_ / (np.linalg.norm(C_, axis=1, keepdims=True) + 1e-9)
    m = ~np.eye(NF, dtype=bool)
    print(f"   {nm:14s} {(Rn @ Rn.T)[m].mean():.2f}   /   {(Cn @ Cn.T)[m].mean():.2f}")

ext0 = ext_of(0)
B.act[:] = 0.0
print("\n只喂第 1 帧 → 每 8 拍：最像哪一帧（相似度），前额叶活动")
for nm in AREAS:
    g = B.ids_of(nm)
    R = ref[nm]
    C_ = R - R.mean(axis=0, keepdims=True)
    Cn = C_ / (np.linalg.norm(C_, axis=1, keepdims=True) + 1e-9)
    out = []
    for t in range(80):
        B.step(external=ext0)
        if t % 8 == 7:
            v = B.act[g].copy()
            sim = Cn @ ((v - R.mean(axis=0)) / (np.linalg.norm(v - R.mean(axis=0)) + 1e-9))
            out.append(f"{int(np.argmax(sim)) + 1}({sim.max():.2f}|{v.mean():.2f})")
    print(f"   {nm:14s} " + " → ".join(out))
