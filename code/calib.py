# -*- coding: utf-8 -*-
"""门槛与带宽都按"这个神经元自己平时信号有多大"来定（全网络一条规矩）。

   门槛 = 0.7 × 它在参考输入下的驱动（中位）
   带宽 = 2.0 × 它在几组参考输入之间驱动的起伏（标准差） + 0.15 × 中位

做法：取三组参考输入（整幅灰、有图案、随机档位），量出每个神经元的驱动；
按浅→深顺序跑两遍。
"""
from __future__ import annotations

import re

import numpy as np

from brain import Brain


def _refs(ui):
    sp = ui.sp
    out = []

    def snap():
        out.append(np.asarray(ui.drive(), dtype=np.float32).copy())

    ui.set_values(0.5)
    ui.set_signals(audio=np.full(sp.n_audio, 0.5), touch=np.full(sp.n_touch, 0.5),
                   balance=np.full(sp.n_balance, 0.5))
    snap()
    im = np.zeros((sp.px, sp.px, sp.n_rgb), np.float32)
    q = max(1, sp.px // 2)
    im[0:q, 0:q, 0] = 0.8
    ui.set_image(im)
    snap()
    rng = np.random.default_rng(3)
    ui.set_image(rng.random((sp.px, sp.px, sp.n_rgb)).astype(np.float32))
    ui.set_signals(audio=rng.random(sp.n_audio).astype(np.float32),
                   touch=rng.random(sp.n_touch).astype(np.float32),
                   balance=rng.random(sp.n_balance).astype(np.float32))
    snap()
    return out


def ref_calibrate(B: Brain, ids: dict, ui, passes: int = 2, **kw) -> None:
    refs = _refs(ui)
    fwd = np.array([not B.region[int(x)].startswith("ret_") for x in B.src])
    sw = np.bincount(B.dst[fwd], weights=B.w[fwd], minlength=B.n)

    groups: dict = {}
    for i in range(B.n):
        r = B.region[i]
        if r.startswith("ret_") or r.endswith(("_L1on", "_L1off")):
            continue
        if r.startswith(("hop_", "pfc_")) or re.search(r"_L\d+$", r):
            groups.setdefault(r, []).append(i)
    order = sorted(groups, key=lambda nm: float(B.xyz(np.array(groups[nm]))[:, 0].mean()))

    for _p in range(passes):
        for nm in order:
            g = np.array(groups[nm], dtype=int)
            drv = []
            for ext in refs:
                B.act[:] = 0.0
                for _ in range(40):
                    B.step(external=ext)
                drv.append(np.abs(np.bincount(B.dst, weights=B.sgn * B.w * B.act[B.src],
                                               minlength=B.n))[g])
            D = np.vstack(drv)
            mid = np.median(D, axis=0)
            std = D.std(axis=0)
            thr = np.asarray(B.thr, dtype=np.float64).copy()
            soft = np.asarray(B.soft, dtype=np.float64).copy()
            relay = nm.startswith(("hop_", "pfc_"))
            # 细胞少的组（最小的感觉区）量出来的"起伏"不可靠：改成按它自己的平均信号定带宽
            small = len(g) < 64 or any(w in nm for w in ("audio", "touch", "balance"))
            soft[g] = np.maximum(1e-4, (0.8 * mid) if (small or relay)
                                 else (2.0 * std + 0.15 * mid))
            base_thr = (0.30 if relay else 0.7) * mid
            if relay:
                # 防饱和：参考状态下最大的驱动也要落在分级区间里，
                # 否则整级钉在 1.0，任何输入变化都表现不出来
                hi = D.max(axis=0)
                base_thr = np.maximum(base_thr, hi - 0.9 * soft[g])
            thr[g] = np.maximum(1e-4, base_thr)
            B.thr = thr.tolist()
            B.soft = soft.tolist()
