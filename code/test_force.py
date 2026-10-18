# -*- coding: utf-8 -*-
"""力度分档测试：点亮前额叶里属于某块肌肉的运动相关神经元 n 个，
跑 140 步，读"力度档 = 活跃的力度神经元个数 / 总档数"。

用法：python test_force.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402

sp = Spec(px=2, n_rgb=3, n_level=3, n_audio=2, n_touch=3, n_balance=2,
          n_muscle=2, n_force=10, k_near=8, layer_ratio=(2.0, 4.0, 6.0),
          layer_floor=(16, 24, 32))
B, ids, sp, geo = build(sp, verbose=False)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)

M = 0                                   # 第 0 块肌肉
print(f"每块肌肉 {sp.n_force} 档力度，肌肉数 {sp.n_muscle}")
print(f"{'亮的运动相关神经元':>10s} | {'下降链末端':>8s} | {'力度档':>6s} | 各档活动")
for n_on in (0, 1, 2, 3, 4, 5, 7, 8, 10):
    ext = np.asarray(ui.drive(), dtype=np.float32).copy()
    if n_on:
        ext[ids["motor_pfc"][M][:n_on]] = 5.0        # 点亮前 n_on 个
    B.act[:] = 0.0
    for _ in range(140):
        B.step(external=ext)
    fo = B.act[ids["force"][M]]
    lvl = int((fo > 0.35).sum())
    dl = float(B.act[ids["down"][M][-1]])
    gi = float(B.act[ids["gate"][M]])
    gd = float(B.act[ids["dis"][M]])
    print(f"{n_on:>10d} | {dl:>8.2f} | {lvl:>3d}/{sp.n_force:<2d} | "
          f"{np.round(fo, 2)}  I={gi:.2f} D={gd:.2f}")
