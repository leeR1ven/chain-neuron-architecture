# -*- coding: utf-8 -*-
"""查听觉送前额叶：逐通道打印各级的变化量（>1e-4 就算传到了）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Config  # noqa: E402
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402

sp = Spec(px=10, n_rgb=3, n_level=10, n_audio=20, n_touch=40, n_balance=6,
          n_muscle=8, n_force=10, k_near=8, layer_ratio=(2.0, 4.0, 6.0),
          layer_floor=(128, 192, 256))
B, ids, sp, geo = build(sp, Config(), verbose=False)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)
BASE = {"audio": np.full(sp.n_audio, 0.34), "touch": np.full(sp.n_touch, 0.34),
        "balance": np.full(sp.n_balance, 0.34)}
img = np.zeros((sp.px, sp.px, sp.n_rgb), np.float32)
STAGES = ("audio_L4", "hop_audio_L4_0", "hop_audio_L4_1", "pfc_audio_L4")


def run(sig):
    ui.set_image(img)
    ui.set_signals(**sig)
    B.act[:] = 0.0
    for _ in range(120):
        B.step(external=ui.drive())
    return B.act.copy()


a0 = run(BASE)
ok = 0
for c in range(sp.n_audio):
    s = {k: v.copy() for k, v in BASE.items()}
    s["audio"][c] = 0.67
    a1 = run(s)
    vals = [np.abs(a1[B.ids_of(nm)] - a0[B.ids_of(nm)]).max() for nm in STAGES]
    ok += vals[-1] > 1e-4
    print(f"  通道{c:2d}: L4活动 {float(B.act[B.ids_of('audio_L4')].mean()):.2f} | "
          f"L4 {vals[0]:.1e} → 中继1 {vals[1]:.1e} → 中继2 {vals[2]:.1e} → 前额叶 {vals[3]:.1e}")
print(f"送到前额叶 {ok}/{sp.n_audio}  gain={B.gain:.3f}")
