# -*- coding: utf-8 -*-
"""按深度给门槛比例：越深门槛比例越低（上一层的活跃密度逐级下降）。

  L2（输入来自第1层，约 42% 活跃）：门槛 0.25 × 权重和，带宽 0.45
  L3（输入来自第2层，较稀）：      门槛 0.15 × 权重和，带宽 0.35
  L4（输入来自第3层，更稀）：      门槛 0.10 × 权重和，带宽 0.30
路上各级与前额叶组按它们所属的 L 号用同一张表。
整体阈值调控保持开启（±3.0）。

测两条：色块位置（左上 vs 右下）、单通道变化，从第2层一路看能不能到前额叶。
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402

FRAC = {}   # 不再手工覆盖：用 build.py 内置的按深度比例
PAT = re.compile(r"_L(\d)")

sp = Spec()
B, ids, sp, geo = build(sp, verbose=False)
ui = Interface(B, ids, sp)
sw = np.bincount(B.dst, weights=B.w, minlength=B.n)
thr = np.asarray(B.thr, dtype=np.float64).copy()
soft = np.asarray(B.soft, dtype=np.float64).copy()
n_set = 0
for i in range(B.n):
    r = B.region[i]
    if not (r.endswith(tuple("_L1 _L2 _L3 _L4".split())) or r.startswith("hop_")
            or r.startswith("pfc_")):
        continue
    m = PAT.search(r)
    if not m:
        continue
    key = f"L{m.group(1)}"
    if key in FRAC:
        f, g = FRAC[key]
        thr[i] = f * sw[i]
        soft[i] = max(1e-4, g * sw[i])
        n_set += 1
B.thr = thr.tolist()
B.soft = soft.tolist()
print(f"按深度设了 {n_set} 个神经元的门槛")

P, C, q = sp.px, sp.n_rgb, max(2, sp.px // 3)


def run(im=None, sig=None):
    if im is not None:
        ui.set_image(im)
    ui.set_signals(audio=np.zeros(sp.n_audio), touch=np.zeros(sp.n_touch),
                   balance=np.zeros(sp.n_balance))
    if sig:
        ui.set_signals(**sig)
    B.act[:] = 0.0
    for _ in range(140):
        B.step(external=ui.drive())
    return B.act.copy()


z = np.zeros((P, P, C), np.float32)
i1 = z.copy(); i1[0:q, 0:q, 0] = 0.8
i2 = z.copy(); i2[P - q:, P - q:, 0] = 0.8
base = np.full((P, P, C), 0.5, np.float32)
ib = base.copy(); ib[3, 4, 1] = 0.6

for tag, (im1, im2) in (("色块位置（左上 vs 右下）", (i1, i2)),
                        ("单通道变化（0.5→0.6）", (base, ib))):
    a, b = run(im1), run(im2)
    print(f"\n{tag}")
    for key in ("vision_L2", "vision_L3", "vision_L4", "hop_vision_L4_1",
                "pfc_vision_L4", "pfc"):
        g = ids[key] if key in ids else B.ids_of(key)
        d = np.abs(a[g] - b[g])
        print(f"   {key:18s} n={len(g):5d} 活动 {a[g].mean():.3f}→{b[g].mean():.3f}  "
              f"最大差 {d.max():.3f}  差>0.02 的 {int((d > 0.02).sum()):5d}")
