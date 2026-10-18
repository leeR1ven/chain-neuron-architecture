# -*- coding: utf-8 -*-
"""验收（重写版）：逐层看"位置 / 颜色"有没有被区分出来，再看门控与实时性。

验收方式按你的要求：**看哪些神经元在变**，而不是看整层的平均值。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Config  # noqa: E402
from build import Geometry, Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run_imgs(B, ui, imgs, sig=None, steps=140):
    out = {}
    for k, im in imgs.items():
        B.act[:] = 0.0
        ui.set_image(im)
        if sig:
            ui.set_signals(**sig.get(k, {}))
        for _ in range(steps):
            B.step(external=ui.drive())
        out[k] = B.act.copy()
    return out


def layer_report(B, _ids, name, ids_sel, acts, ks, pair):
    """对一层：报告 pair 两张画面下，差别最大的几个神经元及其读数。"""
    d = np.abs(acts[pair[0]][ids_sel] - acts[pair[1]][ids_sel])
    n_diff = int((d > 0.02).sum())
    row = (f"  {name:18s} {len(ids_sel):5d} 个  差>0.02 的 {n_diff:4d} 个"
           f"  最大差 {d.max():.3f}")
    if d.max() > 0.02:
        ne = ids_sel[int(np.argmax(d))]
        row += "   " + " ".join(f"{k}={acts[k][ne]:.2f}" for k in ks)
    return row


def main() -> int:
    sp = Spec()
    B, ids, sp, geo = build(sp)
    ui = Interface(B, ids, sp)
    ref_calibrate(B, ids, ui)          # 逐层按各自的驱动定门槛/带宽
    P = sp.px
    q = max(2, P // 3)

    def blk(r0, c0):
        a = np.zeros((P, P, 3), np.float32); a[r0:r0 + q, c0:c0 + q, 0] = 0.8; return a
    imgs = {
        "全红": np.stack([np.full((P, P), .6), np.zeros((P, P)), np.zeros((P, P))], -1),
        "全绿": np.stack([np.zeros((P, P)), np.full((P, P), .6), np.zeros((P, P))], -1),
        "红+蓝": np.stack([np.full((P, P), .6), np.zeros((P, P)), np.full((P, P), .6)], -1),
        "左上块红": blk(0, 0), "右下块红": blk(P - q, P - q),
    }
    ks = list(imgs)
    acts = run_imgs(B, ui, imgs)

    print("\n=== 视觉区逐层：位置（左上块红 vs 右下块红）与颜色（全红 vs 全绿）===")
    print(layer_report(B, ids, "第1层ON", ids["vision_L1on"], acts, ks,
                       ("左上块红", "右下块红")))
    print(layer_report(B, ids, "第1层OFF", ids["vision_L1off"], acts, ks,
                       ("左上块红", "右下块红")))
    for li in (2, 3, 4):
        print(layer_report(B, ids, f"第{li}层", ids[f"vision_L{li}"], acts, ks,
                           ("左上块红", "右下块红")))
    print("  —— 颜色 ——")
    for li in (2, 3, 4):
        print(layer_report(B, ids, f"第{li}层", ids[f"vision_L{li}"], acts, ks,
                           ("全红", "全绿")))
    print(layer_report(B, ids, "前额叶(视觉末层)", ids["pfc_vision"], acts, ks,
                       ("左上块红", "右下块红")))
    print(layer_report(B, ids, "前额叶(视觉末层)", ids["pfc_vision"], acts, ks,
                       ("全红", "全绿")))

    # ---------------- 门控 ----------------
    print("\n=== 门控：运动相关神经元是不是主要被激活的 ===")
    for n_on in (1, 4, 6, 10):
        B.act[:] = 0.0
        ext = np.zeros(B.n, dtype=np.float32)
        ext[ids["other_pfc"]] = 5.0                    # 与运动无关的一大片亮着
        ext[ids["motor_pfc"][0][:n_on]] = 5.0          # 再点亮前 n_on 个运动相关
        for _ in range(120):
            B.step(external=ext)
        dr = float(B.act[ids["down"][0]].max())
        fo = int((B.act[ids["force"][0]] > 0.35).sum())
        gi = float(B.act[ids["gate"][0]]); gd = float(B.act[ids["dis"][0]])
        print(f"  运动相关亮 {n_on:2d} 个 → I(门控)={gi:.2f}  D(去抑制)={gd:.2f}  "
              f"下降链最大={dr:.2f}  力度档={fo}/{sp.n_force}")

    # ---------------- 实时性 ----------------
    print("\n=== 实时性（四种感觉全开）===")
    ui.set_values(0.5)
    ui.set_signals(audio=np.full(sp.n_audio, .4), touch=np.full(sp.n_touch, .6),
                   balance=np.full(sp.n_balance, .3))
    for _ in range(20):
        B.step(external=ui.drive())
    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        B.step(external=ui.drive())
    ms = (time.perf_counter() - t0) / N * 1000
    print(f"  神经元 {B.n}，连接 {len(B.src):,} → 每步 {ms:.2f} 毫秒"
          f"（20 毫秒一拍占 {ms/20*100:.1f}%）")

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"n_neurons": int(B.n), "n_edges": int(len(B.src)),
                   "ms_per_step": ms}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
