# -*- coding: utf-8 -*-
"""目标规模验收：10×10 像素、10 档、听觉 20 / 触觉 40 / 平衡 6。
视觉 300 条通道，为控制时间每条隔 10 条抽样（30 条），其余区全测。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402

sp = Spec(px=10, n_rgb=3, n_level=10, n_audio=20, n_touch=40, n_balance=6,
          n_muscle=8, n_force=10, k_near=8, layer_ratio=(2.0, 4.0, 6.0),
          layer_floor=(128, 192, 256))
B, ids, sp, geo = build(sp, verbose=True)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)
P, C = sp.px, sp.n_rgb
BASE = {"audio": np.full(sp.n_audio, 0.34), "touch": np.full(sp.n_touch, 0.34),
        "balance": np.full(sp.n_balance, 0.34)}
base_img = np.full((P, P, C), 0.34, np.float32)
MODS = (("视觉", "vision", P * P * C, 10), ("听觉", "audio", sp.n_audio, 1),
        ("触觉", "touch", sp.n_touch, 1), ("平衡", "balance", sp.n_balance, 1))


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
print(f"{'区':6s} {'抽样':>4s} | 各层变化传导      | 送到前额叶   图案相似度")
for nm, key, n, step in MODS:
    idx = list(range(0, n, step))
    n2 = n3 = n4 = npf = 0
    vecs = []
    for c in idx:
        if key == "vision":
            v = base_img.copy(); v.reshape(-1, C)[c // C, c % C] = 0.67
            a1 = run(v, BASE)
        else:
            a1 = run(base_img, sig_of(key, c))
        g2, g3, g4 = ids[f"{key}_L2"], ids[f"{key}_L3"], ids[f"{key}_L4"]
        n2 += np.abs(a1[g2] - a0[g2]).max() > 1e-4
        n3 += np.abs(a1[g3] - a0[g3]).max() > 1e-4
        n4 += np.abs(a1[g4] - a0[g4]).max() > 1e-4
        gp = B.ids_of(f"pfc_{key}_L4")
        d = a1[gp] - a0[gp]
        npf += np.abs(d).max() > 1e-4
        vecs.append(d)
    M = np.array(vecs)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = Mn @ Mn.T
    off = S[~np.eye(len(S), dtype=bool)]
    print(f"{nm:6s} {len(idx):4d} | {n2}/{len(idx)} {n3}/{len(idx)} {n4}/{len(idx)} | "
          f"{npf}/{len(idx)}   {off.mean():.2f}")
