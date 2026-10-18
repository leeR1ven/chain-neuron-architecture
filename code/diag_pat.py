# -*- coding: utf-8 -*-
"""用**有图案的变化**逐区测：各区能不能分辨信号（这才是有意义的尺子）。

测法：每个区各给一个"局部变化"，看该区第2层 / 第4层 / 它送往前额叶的那一组
      在两种情形下的活动与差别；同时打出活动水平（看是饱和、是死、还是分级）。

  视觉：一个像素的一个颜色通道 0.5→0.6；色块位置（左上 vs 右下）；颜色（全红 vs 全绿）
  听觉：一个频带 0.5→0.8
  触觉：一个触点 0.5→0.8
  平衡：一个量   0.5→0.8
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
P, C = sp.px, sp.n_rgb


def settle(steps=140):
    B.act[:] = 0.0
    for _ in range(steps):
        B.step(external=ui.drive())
    return B.act.copy()


def base_img(v=0.5):
    ui.set_image(np.full((P, P, C), v, np.float32))


def base_sig(v=0.5):
    ui.set_signals(audio=np.full(sp.n_audio, v), touch=np.full(sp.n_touch, v),
                   balance=np.full(sp.n_balance, v))


def show(tag, keys, a, b):
    print(f"\n{tag}")
    for k in keys:
        g = ids[k] if k in ids else B.ids_of(k)
        d = np.abs(a[g] - b[g])
        print(f"   {k:18s} n={len(g):5d} 活动 {a[g].mean():.3f}→{b[g].mean():.3f}  "
              f"最大差 {d.max():.3f}  差>0.02 的 {int((d > 0.02).sum()):5d}")


# 视觉：单通道
base_img(); base_sig()
a = settle()
im = np.full((P, P, C), 0.5, np.float32); im[3, 4, 1] = 0.6
ui.set_image(im)
b = settle()
show("视觉：一个像素的一个颜色通道 0.5→0.6", ["vision_L2", "vision_L4", "pfc_vision_L4"], a, b)

# 视觉：色块位置
base_img(0.0)
q = max(2, P // 3)
im1 = np.zeros((P, P, C), np.float32); im1[0:q, 0:q, 0] = 0.8
im2 = np.zeros((P, P, C), np.float32); im2[P - q:P, P - q:P, 0] = 0.8
ui.set_image(im1); a = settle()
ui.set_image(im2); b = settle()
show("视觉：左上块红 vs 右下块红", ["vision_L2", "vision_L4", "pfc_vision_L4"], a, b)

# 听觉 / 触觉 / 平衡：单通道
for mod, n, key2, key4 in (("听觉", sp.n_audio, "audio_L2", "audio_L4"),
                           ("触觉", sp.n_touch, "touch_L2", "touch_L4"),
                           ("平衡", sp.n_balance, "balance_L2", "balance_L4")):
    base_img(); base_sig()
    a = settle()
    v = getattr(ui, "sig")[{"听觉": "audio", "触觉": "touch", "平衡": "balance"}[mod]].copy()
    v[3] = 0.8
    ui.set_signals(**{{"听觉": "audio", "触觉": "touch", "平衡": "balance"}[mod]: v})
    b = settle()
    show(f"{mod}：第3号通道 0.5→0.8", [key2, key4], a, b)
