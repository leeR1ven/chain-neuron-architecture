# -*- coding: utf-8 -*-
"""只改一路信号时，前额叶的**连续活动向量**差多少（不只看"亮了几个"）。

顺便确认反射抑制链现在由触觉"正"细胞触发时是否有效。
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


vec = {}
for mod in ("vision", "audio", "touch", "balance"):
    ui.set_image(np.zeros((sp.px, sp.px, sp.n_rgb), np.float32))
    ui.set_signals(audio=np.zeros(sp.n_audio), touch=np.zeros(sp.n_touch),
                   balance=np.zeros(sp.n_balance))
    if mod == "vision":
        ui.set_image(np.full((sp.px, sp.px, sp.n_rgb), 0.8, np.float32))
    else:
        ui.set_signals(**{mod: np.full(len(ui.sig[mod]), 0.8)})
    vec[mod] = settle(ui.drive())[ids["pfc"]]

ks = list(vec)
print("前额叶活动向量的平均差（连续值，不只是数个数）：")
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        d = float(np.abs(vec[ks[i]] - vec[ks[j]]).mean())
        mx = float(np.abs(vec[ks[i]] - vec[ks[j]]).max())
        print(f"  {ks[i]:8s} vs {ks[j]:8s}: 平均差 {d:.4f}  最大差 {mx:.3f}")

print("\n反射抑制链（现在由触觉'正'细胞触发）：")
m = 0
for label, trig in (("不触发", False), ("触发（触觉有信号）", True)):
    B.act[:] = 0.0
    ext = np.zeros(B.n, dtype=np.float32)
    ext[ids["motor_pfc"][m]] = 1.6
    if trig:
        ext[ids["pos"]["touch"].ravel()] = 1.6
    for _ in range(140):
        B.step(external=ext)
    print(f"  {label:20s} → 抑制细胞 {float(B.act[ids['inh'][m]].max()):.2f}  "
          f"力度档 {int((B.act[ids['force'][m]] > 0.35).sum())}/{sp.n_force}")
