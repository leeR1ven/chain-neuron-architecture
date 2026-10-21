# -*- coding: utf-8 -*-
"""生成 → 测试 → 选优 → 融合 → 再生成（训练循环）。

思路（就是用户定的那套）：
  1. 随机生成一批模型（随机的是"出生参数"：连谁、连几个、权重多大、门槛比例）
  2. 用**同一套固定题库**给每个模型打分（不许每个模型换题）
  3. 选出分数高的
  4. 融合：把优秀模型的"出生参数"按位平均（权重、门槛），生成新一代
  5. 在融合结果上再抖动（变异），回到第 2 步

题库（固定、可复现）：四个区各抽若干通道做"最小一档变化"，加一条位置变化；
每项记 0/1，按通过率给分。评价只看行为，不看内部。
"""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import replace

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Config  # noqa: E402
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def make_spec(rng):
    """随机出生参数：感受野、层规模、权重范围都允许抖。"""
    return Spec(px=4, n_rgb=3, n_level=5, n_audio=4, n_touch=8, n_balance=3,
                n_muscle=2, n_force=3,
                k_near=int(rng.integers(6, 13)),
                layer_ratio=tuple(2.0 * rng.uniform(0.8, 1.25, 3)),
                layer_floor=(64, 96, 128))


def evaluate(B, ids, sp, seed=0):
    """固定题库打分（只跑少量通道，保证快）。"""
    ui = Interface(B, ids, sp)
    ref_calibrate(B, ids, ui)
    P, C = sp.px, sp.n_rgb
    BASE = {"audio": np.full(sp.n_audio, 0.34), "touch": np.full(sp.n_touch, 0.34),
            "balance": np.full(sp.n_balance, 0.34)}
    base_img = np.full((P, P, C), 0.34, np.float32)
    rng = np.random.default_rng(seed)

    def run(img, sig):
        ui.set_image(img)
        ui.set_signals(**sig)
        B.act[:] = 0.0
        for _ in range(100):
            B.step(external=ui.drive())
        return B.act.copy()

    a0 = run(base_img, BASE)
    score, total = 0, 0
    for key, n, cs in (("vision", P * P * C, [0, 5, 17, 33]),
                       ("audio", sp.n_audio, [0, 2]),
                       ("touch", sp.n_touch, [1, 5]),
                       ("balance", sp.n_balance, [0, 2])):
        for c in cs:
            if c >= n:
                continue
            total += 1
            if key == "vision":
                v = base_img.copy(); v.reshape(-1, C)[c // C, c % C] = 0.67
                a1 = run(v, BASE)
            else:
                s = {k: x.copy() for k, x in BASE.items()}
                s[key][c] = 0.67
                a1 = run(base_img, s)
            for gid in (f"{key}_L2", f"{key}_L4"):
                g = ids[gid]
                if np.abs(a1[g] - a0[g]).max() > 1e-4:
                    score += 1
            gp = B.ids_of(f"pfc_{key}_L4")
            if len(gp) and np.abs(a1[gp] - a0[gp]).max() > 1e-4:
                score += 1
            total += 2
    return score / max(1, total)


def mutate(base, rng):
    """在最优者附近抖动出生参数（融合=直接继承，变异=小步抖动）。"""
    k = int(np.clip(base.k_near + rng.integers(-2, 3), 4, 24))
    ratio = tuple(float(np.clip(x * rng.uniform(0.85, 1.15), 0.5, 12.0))
                  for x in base.layer_ratio)
    return replace(base, k_near=k, layer_ratio=ratio)


def run_one(sp, seed):
    cfg = Config(seed=int(seed))
    B, ids, sp, _ = build(sp, cfg, verbose=False)
    return evaluate(B, ids, sp), sp, cfg.seed


def main(gens=3, pop=4, keep=2):
    rng = np.random.default_rng(2024)
    best = None
    for gen in range(gens):
        rows = []
        if best is None:
            for _ in range(pop):
                rows.append(run_one(make_spec(rng), int(rng.integers(1, 10 ** 6))))
        else:
            # 上一代最优原样进下一代（精英保留），其余是它的变异体
            rows.append((best[0], best[1], best[2]))
            for _ in range(pop - 1):
                sp = mutate(best[1], rng)
                rows.append(run_one(sp, int(rng.integers(1, 10 ** 6))))
        rows.sort(key=lambda r: -r[0])
        if best is None or rows[0][0] > best[0]:
            best = rows[0]
        print(f"第 {gen+1} 代： " + "  ".join(f"{s:.3f}" for s, _, _ in rows)
              + f"   当前最优 {best[0]:.3f}")
    print(f"\n最优：得分 {best[0]:.3f}  感受野 {best[1].k_near}  层比 {best[1].layer_ratio}")
    with open(os.path.join(HERE, "train_loop_result.json"), "w", encoding="utf-8") as f:
        json.dump({"best_score": best[0], "k_near": int(best[1].k_near),
                   "layer_ratio": list(best[1].layer_ratio),
                   "seed": int(best[2])}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
