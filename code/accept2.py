# -*- coding: utf-8 -*-
"""换尺子：不再要求"某个神经元差 >0.1"，而是问两件事：
   1. 变化传下去了吗？—— 该层最大差 > 1e-4（网络是确定的，变了就是传下去了）
   2. 送去前额叶的图案分得开吗？—— 不同通道的响应图案两两相似度（越低越分得开）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402

sp = Spec(px=4, n_rgb=3, n_level=5, n_audio=4, n_touch=8, n_balance=3,
          n_muscle=2, n_force=3, k_near=8, layer_ratio=(2.0, 4.0, 6.0),
          layer_floor=(64, 96, 128))
B, ids, sp, geo = build(sp, verbose=False)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)
P, C = sp.px, sp.n_rgb
MODS = (("视觉", "vision", P * P * C), ("听觉", "audio", sp.n_audio),
        ("触觉", "touch", sp.n_touch), ("平衡", "balance", sp.n_balance))
BASE = {"audio": np.full(sp.n_audio, 0.34), "touch": np.full(sp.n_touch, 0.34),
        "balance": np.full(sp.n_balance, 0.34)}
base_img = np.full((P, P, C), 0.34, np.float32)


def run(img, sig):
    ui.set_image(img)
    ui.set_signals(**sig)
    B.act[:] = 0.0
    for _ in range(120):
        B.step(external=ui.drive())
    return B.act.copy()


def sig_of(key, c):
    s = {k: v.copy() for k, v in BASE.items()}
    s[key] = s[key].copy(); s[key][c] = 0.67
    return s


a0 = run(base_img, BASE)
print(f"{'区':6s} {'通道':>4s} | 变化传下去了吗（各层最大差）      | 前额叶图案可分性")
for nm, key, n in MODS:
    n2 = n3 = n4 = npf = 0
    pf_vecs = []
    for c in range(n):
        if key == "vision":
            v = base_img.copy(); v.reshape(-1, C)[c // C, c % C] = 0.67
            a1 = run(v, BASE)
        else:
            a1 = run(base_img, sig_of(key, c))
        g2, g3, g4 = ids[f"{key}_L2"], ids[f"{key}_L3"], ids[f"{key}_L4"]
        d2 = np.abs(a1[g2] - a0[g2]).max()
        d3 = np.abs(a1[g3] - a0[g3]).max()
        d4 = np.abs(a1[g4] - a0[g4]).max()
        n2 += d2 > 1e-4; n3 += d3 > 1e-4; n4 += d4 > 1e-4
        gp = B.ids_of(f"pfc_{key}_L4")
        v = a1[gp] - a0[gp]
        npf += np.abs(v).max() > 1e-4
        pf_vecs.append(v)
    M = np.array(pf_vecs)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = Mn @ Mn.T
    off = S[~np.eye(len(S), dtype=bool)]
    print(f"{nm:6s} {n:4d} | 第2层 {n2}/{n}  第3层 {n3}/{n}  第4层 {n4}/{n} | "
          f"送达 {npf}/{n}，两两相似度 {off.mean():.2f}")
