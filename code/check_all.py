# -*- coding: utf-8 -*-
"""全面体检：把还没验过的项目一次查完。

 1. 四个感觉区**各自**能不能区分信号（视觉/听觉/触觉/平衡）
 2. **前额叶**能不能区分来自不同感觉区的信号（只听/只触/只看/只平衡）
 3. 抑制两条（前额叶"别动"链 / 感觉区反射抑制链）到底有没有用
 4. 各层活动水平（是不是整体饱和）
 5. 阈值整体调控现在是开还是关
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Config  # noqa: E402
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


def neutral(v=0.5, s=0.5):
    ui.set_values(v)
    ui.set_signals(audio=np.full(sp.n_audio, s), touch=np.full(sp.n_touch, s),
                   balance=np.full(sp.n_balance, s))
    return ui.drive()


print("=== 1. 四个感觉区各自能不能区分信号 ===")
base = settle(neutral())
for mod, n, k in (("vision", sp.px * sp.px * sp.n_rgb, "image"),
                  ("audio", sp.n_audio, "audio"),
                  ("touch", sp.n_touch, "touch"),
                  ("balance", sp.n_balance, "balance")):
    best = 0.0
    which = -1
    for ci in range(0, n, max(1, n // 12)):        # 抽样测，省时间
        ui.set_values(0.5)
        ui.set_signals(audio=np.full(sp.n_audio, .5), touch=np.full(sp.n_touch, .5),
                       balance=np.full(sp.n_balance, .5))
        if k == "image":
            im = np.full((sp.px, sp.px, sp.n_rgb), 0.5, np.float32)
            im.reshape(-1, sp.n_rgb)[ci // sp.n_rgb, ci % sp.n_rgb] = 0.9
            ui.set_image(im)
        else:
            v = getattr(ui, "sig")[k].copy(); v[ci] = 0.9
            ui.set_signals(**{k: v})
        a = settle(ui.drive())
        d = np.abs(a[ids[f"{mod}_L2"]] - base[ids[f"{mod}_L2"]]).max()
        if d > best:
            best, which = d, ci
    a = settle(ui.drive())
    print(f"  {mod:8s} 只改一个通道 → 第2层最大差 {best:.3f}（第 {which} 号通道）")

print("\n=== 2. 前额叶能不能区分'是哪种感觉' ===")
pfc_sets = {}
for mod in ("vision", "audio", "touch", "balance"):
    ui.set_image(np.zeros((sp.px, sp.px, sp.n_rgb), np.float32))
    ui.set_signals(audio=np.zeros(sp.n_audio), touch=np.zeros(sp.n_touch),
                   balance=np.zeros(sp.n_balance))
    if mod == "vision":
        ui.set_image(np.full((sp.px, sp.px, sp.n_rgb), 0.8, np.float32))
    else:
        ui.set_signals(**{mod: np.full(len(ui.sig[mod]), 0.8)})
    a = settle(ui.drive())
    pfc_sets[mod] = set(np.nonzero(a[ids["pfc"]] > 0.35)[0].tolist())
    print(f"  只有{mod:8s}有信号 → 前额叶点亮 {len(pfc_sets[mod]):4d} 个")
ks = list(pfc_sets)
print("  两两差多少个：")
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print(f"    {ks[i]:8s} vs {ks[j]:8s}: 差 {len(pfc_sets[ks[i]] ^ pfc_sets[ks[j]]):4d}")

print("\n=== 3. 两条抑制通路到底有没有用 ===")
m = 0
for label, pfc_on, sense_on in (("两条都断", False, False), ("只通别动链", True, False),
                                ("只通反射抑制链", False, True), ("两条都通", True, True)):
    B.act[:] = 0.0
    ext = np.zeros(B.n, dtype=np.float32)
    ext[ids["motor_pfc"][m]] = 1.6                      # 该肌肉的运动相关组全亮
    ext[ids["dontmove"]] = 1.6 if pfc_on else 0.0        # "别动"指令
    ext[np.r_[ids["neg"]["touch"][-1], ids["neg"]["balance"][-1]]] = 1.6 if sense_on else 0.0
    for _ in range(140):
        B.step(external=ext)
    print(f"  {label:12s} → 抑制细胞 {float(B.act[ids['inh'][m]]):.2f}  "
          f"力度档 {int((B.act[ids['force'][m]] > 0.35).sum())}/{sp.n_force}")

print("\n=== 4. 各层活动水平（是不是整体饱和）===")
ui.set_values(0.5)
ui.set_signals(audio=np.full(sp.n_audio, .5), touch=np.full(sp.n_touch, .5),
               balance=np.full(sp.n_balance, .5))
for _ in range(140):
    B.step(external=ui.drive())
for key in ("vision_L1on", "vision_L2", "vision_L3", "vision_L4", "pfc", "force0"):
    a = B.act[ids[key]]
    print(f"  {key:12s} 均值 {a.mean():.3f}  最大 {a.max():.3f}  "
          f">0.9 的比例 {float((a > 0.9).mean()):.2f}")

print(f"\n=== 5. 阈值整体调控：{'开' if B.cfg.target_active else '关（target_active=0）'}"
      f"，gain={B.gain:.3f} ===")
