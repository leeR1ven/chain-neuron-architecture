# -*- coding: utf-8 -*-
"""总验收：一条命令把三块成果全验一遍。
  ① 四区逐通道表（最小一档变化：各层能不能传下去 + 能不能送到前额叶 + 图案分不分得开）
  ② 运动力度表（亮的运动相关神经元个数 → 出力档位）
  ③ 可塑连接自测（若 plastic.py 已存在）
判据用"变化传下去了吗（>1e-4）"，不用"差值>0.1"（会严重低估）。
"""
import importlib
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
B, ids, sp, geo = build(sp, Config(), verbose=True)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)
P, C = sp.px, sp.n_rgb
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


print("\n① 四区逐通道表")
a0 = run(base_img, BASE)
for nm, key, n, step in (("视觉", "vision", P * P * C, 10), ("听觉", "audio", sp.n_audio, 1),
                         ("触觉", "touch", sp.n_touch, 4), ("平衡", "balance", sp.n_balance, 1)):
    idx = list(range(0, n, step))
    c2 = c3 = c4 = cpf = 0
    vecs = []
    for c in idx:
        if key == "vision":
            v = base_img.copy(); v.reshape(-1, C)[c // C, c % C] = 0.67
            a1 = run(v, BASE)
        else:
            s = {k: x.copy() for k, x in BASE.items()}
            s[key][c] = 0.67
            a1 = run(base_img, s)
        for tag, gid in (("2", f"{key}_L2"), ("3", f"{key}_L3"), ("4", f"{key}_L4")):
            g = ids[gid]
            d = np.abs(a1[g] - a0[g]).max() > 1e-4
            c2 += d if tag == "2" else 0
            c3 += d if tag == "3" else 0
            c4 += d if tag == "4" else 0
        gp = B.ids_of(f"pfc_{key}_L4")
        dd = a1[gp] - a0[gp]
        cpf += np.abs(dd).max() > 1e-4
        vecs.append(dd)
    M = np.array(vecs)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = Mn @ Mn.T
    off = S[~np.eye(len(S), dtype=bool)]
    print(f"   {nm:4s} 抽样{len(idx):3d}：各层 {c2}/{len(idx)} {c3}/{len(idx)} {c4}/{len(idx)}"
          f" | 送前额叶 {cpf}/{len(idx)} | 图案相似度 {off.mean():.2f}")

print("\n② 运动力度表（点亮前额叶运动相关组的个数 → 出力档位）")
ui.set_image(np.zeros((P, P, C), np.float32))
ui.set_signals(audio=np.zeros(sp.n_audio), touch=np.zeros(sp.n_touch),
               balance=np.zeros(sp.n_balance))
for k in (1, 2, 3, 5, 7, 8, 10):
    B.act[:] = 0.0
    ext = np.zeros(B.n, dtype=np.float32)
    ext[ids["motor_pfc"][0][:k]] = 5.0
    for _ in range(140):
        B.step(external=ext)
    lvl = int((B.act[ids["force"][0]] > 0.35).sum())
    print(f"   亮 {k:2d} 个 → 力度档 {lvl}/{sp.n_force}")

print("\n③ 可塑连接自测")
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plastic.py")):
    try:
        mod = importlib.import_module("plastic")
        if hasattr(mod, "self_test"):
            mod.self_test()
        else:
            print("   plastic.py 存在，但没有 self_test()；请单独跑它的测试脚本")
    except Exception as e:                                   # noqa: BLE001
        print(f"   调用失败：{e}")
else:
    print("   plastic.py 还没交回来")
