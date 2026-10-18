# -*- coding: utf-8 -*-
"""逐通道验收（四个区 + 前额叶）：每条通道各动最小一档，逐条检查。

检查点：该区第2/3/4层 + 该区送往前额叶的那一组（pfc_<区>_L4）。
判据：至少 1 个神经元 |Δ| > 0.1。
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


def run(img=None, sig=None):
    if img is not None:
        ui.set_image(img)
    ui.set_signals(audio=np.zeros(sp.n_audio) if not sig else sig["audio"],
                   touch=np.zeros(sp.n_touch) if not sig else sig.get("touch", np.zeros(sp.n_touch)),
                   balance=np.zeros(sp.n_balance) if not sig else sig.get("balance", np.zeros(sp.n_balance)))
    B.act[:] = 0.0
    for _ in range(120):
        B.step(external=ui.drive())
    return B.act.copy()


base_img = np.full((P, P, C), 0.34, np.float32)


def make_img(c):
    v = base_img.copy()
    v.reshape(-1, C)[c // C, c % C] = 0.67
    return v


def make_sig(key, n, c):
    s = {k: np.zeros(m) for k, m in (("audio", sp.n_audio), ("touch", sp.n_touch),
                                     ("balance", sp.n_balance))}
    for k in s:
        s[k] = np.full(len(s[k]), 0.34)
    s[key] = s[key].copy(); s[key][c] = 0.67
    return s


BASE = {"audio": np.full(sp.n_audio, 0.34), "touch": np.full(sp.n_touch, 0.34),
        "balance": np.full(sp.n_balance, 0.34)}
a0_all = {key: run(base_img, BASE) for _, key, _ in MODS}

print(f"{'区':6s} {'通道数':>5s} {'第2层':>7s} {'第3层':>7s} {'第4层':>7s} {'前额叶组':>8s}")
for nm, key, n in MODS:
    ok = {"L2": 0, "L3": 0, "L4": 0, "PF": 0}
    for c in range(n):
        if key == "vision":
            a1 = run(make_img(c), BASE)
        else:
            a1 = run(base_img, make_sig(key, n, c))
        a0 = a0_all[key]
        for tag, gid in (("L2", f"{key}_L2"), ("L3", f"{key}_L3"),
                         ("L4", f"{key}_L4")):
            g = ids.get(gid) if gid in ids else B.ids_of(gid)
            if len(g) and np.abs(a1[g] - a0[g]).max() > 0.1:
                ok[tag] += 1
        gp = B.ids_of(f"pfc_{key}_L4")
        if len(gp) and np.abs(a1[gp] - a0[gp]).max() > 0.1:
            ok["PF"] += 1
    print(f"{nm:6s} {n:5d} {ok['L2']:>4d}/{n} {ok['L3']:>4d}/{n} "
          f"{ok['L4']:>4d}/{n} {ok['PF']:>5d}/{n}")
