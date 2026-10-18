# -*- coding: utf-8 -*-
"""查细微区分能力（你点名的几处）：
 A. 输入只动一点点（整幅灰度 0.50→0.55，半档）→ 各区第2层 / 前额叶有没有反应
 B. 只动一个像素的一个颜色通道（0.5→0.6）→ 有没有反应
 C. 运动力度能不能细分：前额叶运动相关组分别亮 3/5/7/10 个 → 读出力等级
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402

sp = Spec()
B, ids, sp, geo = build(sp, verbose=False)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)


def settle(ext, steps=140):
    B.act[:] = 0.0
    for _ in range(steps):
        B.step(external=ext)
    return B.act.copy()


def vals(v):
    ui.set_values(v)
    ui.set_signals(audio=np.full(sp.n_audio, 0.5), touch=np.full(sp.n_touch, 0.5),
                   balance=np.full(sp.n_balance, 0.5))
    return ui.drive()


a = settle(vals(0.50))
b = settle(vals(0.55))
print("A. 整幅灰度 0.50 → 0.55（半档）")
for k in ("vision_L2", "vision_L3", "vision_L4"):
    d = np.abs(a[ids[k]] - b[ids[k]])
    print(f"   {k:10s} 最大差 {d.max():.3f}  差>0.02 的 {int((d>0.02).sum())}/{len(ids[k])}")
for nm in ("audio", "touch", "balance"):
    k = f"{nm}_L2"
    d = np.abs(a[ids[k]] - b[ids[k]])
    print(f"   {k:10s} 最大差 {d.max():.3f}  差>0.02 的 {int((d>0.02).sum())}/{len(ids[k])}")
dp = np.abs(a[ids["pfc"]] - b[ids["pfc"]])
print(f"   前额叶      最大差 {dp.max():.3f}  差>0.02 的 {int((dp>0.02).sum())}/{len(ids['pfc'])}")

base = np.full((sp.px, sp.px, sp.n_rgb), 0.5, np.float32)
a2 = settle(vals(0.50))
im = base.copy(); im[3, 4, 1] = 0.6          # 只动一个像素的一个通道
ui.set_image(im)
ui.set_signals(audio=np.full(sp.n_audio, 0.5), touch=np.full(sp.n_touch, 0.5),
               balance=np.full(sp.n_balance, 0.5))
b2 = settle(ui.drive())
print("B. 只动一个像素的一个颜色通道（0.5→0.6）")
for k in ("vision_L2", "vision_L3", "vision_L4", "pfc"):
    d = np.abs(a2[ids[k]] - b2[ids[k]])
    print(f"   {k:10s} 最大差 {d.max():.3f}  差>0.02 的 {int((d>0.02).sum())}/{len(ids[k])}")

print("C. 运动力度细分：前额叶运动相关组点亮 n 个 → 力度档")
for n in (2, 3, 5, 7, 8, 10):
    B.act[:] = 0.0
    ext = np.zeros(B.n, dtype=np.float32)
    ext[ids["motor_pfc"][0][:n]] = 5.0
    for _ in range(140):
        B.step(external=ext)
    on = int((B.act[ids["force"][0]] > 0.35).sum())
    print(f"   亮 {n:2d} 个 → 力度档 {on}/10，下降链 {float(B.act[ids['down'][0]].max()):.2f}")
