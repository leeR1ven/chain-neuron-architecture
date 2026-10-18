# -*- coding: utf-8 -*-
"""极小模型端到端自检。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build
from calib import ref_calibrate

sp = Spec(px=2, n_rgb=3, n_level=3, n_audio=2, n_touch=3, n_balance=2,
          n_muscle=2, n_force=3, k_near=8, layer_ratio=(2.0, 4.0, 6.0),
          layer_floor=(16, 24, 32))
B, ids, sp, geo = build(sp, verbose=True)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)


def run(img=None, sig=None):
    if img is not None:
        ui.set_image(img)
    ui.set_signals(audio=np.zeros(sp.n_audio) if not sig else sig.get("audio", np.zeros(sp.n_audio)),
                   touch=np.zeros(sp.n_touch) if not sig else sig.get("touch", np.zeros(sp.n_touch)),
                   balance=np.zeros(sp.n_balance) if not sig else sig.get("balance", np.zeros(sp.n_balance)))
    B.act[:] = 0.0
    for _ in range(120):
        B.step(external=ui.drive())
    return B.act.copy()


P, C = sp.px, sp.n_rgb
base = np.full((P, P, C), 0.33, np.float32)
chg = base.copy(); chg[0, 0, 0] = 0.66
pos1 = np.zeros((P, P, C), np.float32); pos1[0, 0, 0] = 1.0
pos2 = np.zeros((P, P, C), np.float32); pos2[P - 1, P - 1, 0] = 1.0
a, b = run(base), run(chg)
print("\n视觉：一个像素一个通道 0.33→0.66")
for k in ("vision_L2", "vision_L4", "pfc_vision_L4", "pfc"):
    g = ids[k] if k in ids else B.ids_of(k)
    d = np.abs(a[g] - b[g])
    print(f"   {k:16s} n={len(g):4d} 活动 {a[g].mean():.3f}→{b[g].mean():.3f} "
          f"最大差 {d.max():.3f} 差>0.02 的 {int((d > 0.02).sum())}")
a, b = run(pos1), run(pos2)
print("视觉：亮块在左上 vs 右下")
for k in ("vision_L2", "vision_L4", "pfc_vision_L4", "pfc"):
    g = ids[k] if k in ids else B.ids_of(k)
    d = np.abs(a[g] - b[g])
    print(f"   {k:16s} n={len(g):4d} 活动 {a[g].mean():.3f}→{b[g].mean():.3f} "
          f"最大差 {d.max():.3f} 差>0.02 的 {int((d > 0.02).sum())}")
for nm, n, key in (("听觉", sp.n_audio, "audio"), ("触觉", sp.n_touch, "touch"),
                   ("平衡", sp.n_balance, "balance")):
    ui.set_image(np.zeros((P, P, C), np.float32))
    s0 = {key: np.full(n, 0.33)}
    a = run(sig=s0)
    s1 = {key: np.full(n, 0.33)}; s1[key] = np.full(n, 0.33); s1[key][0] = 1.0
    b = run(sig=s1)
    for k in (f"{key}_L2", f"{key}_L4"):
        g = ids[k]
        d = np.abs(a[g] - b[g])
        print(f"{nm}：第0号通道 0.33→1.0  {k:12s} n={len(g):4d} "
              f"最大差 {d.max():.3f} 差>0.02 的 {int((d > 0.02).sum())}")
