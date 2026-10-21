# -*- coding: utf-8 -*-
"""结构可塑性：链上动态增/减神经元 + "常亮不带特征"的前额叶筛选。

不改链的走法（温度计编码、就近 (y,z) 连线、逐神经元门槛/带宽、
每一层经中继送前额叶、回程调制、运动区分档、门控都保持原样），只改"链上有多少神经元、
哪些信号继续往前额叶送"：

  1. **加神经元**：只加在"中间使用频率"的父神经元旁边。
     （立体网络那边的实测：高频分裂会伤稀有信号，中间频段 + 少输入最稳。）
     新神经元只接 4 条输入 = 父神经元里 2 条最强的 + 2 条随机的，
     所以它比父神经元更专一；位置在同层 (y,z) 上轻微抖动；
     它仍按 (y,z) 就近接进**这一层原有的中继级**，走"每层往前额叶送"的原路。
  2. **减神经元**：只删两类——
       跨输入几乎不激活的（没用），
       跨输入始终同一读数的（常亮/常灭，不带特征）。
     删完把它带死的 hop / 前额叶神经元级联删掉。
  3. **常亮不送前额叶**：用一批随机输入看**跨输入的起伏**（不是时间上的变化）。
     std≈0 的神经元不把信号送进 hop / 前额叶（把它的入边去掉），
     留下的都是有特征、能区分输入的那批。
  4. 每次结构改动后重新标定门槛（按每个神经元自己的入边权重和 / 实测驱动），
     再复测前额叶四区相似度；变差就回滚。

用法：
    python structure.py            # 中等规模演示（几分钟）
    python structure.py --full     # 目标规模（更慢）
"""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Config  # noqa: E402
from build import (DEPTH_GAIN, GROUP_PFC_FWD, Geometry, Interface, Spec,  # noqa: E402
                   build)
from calib import ref_calibrate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REGIONS = ("vision", "audio", "touch", "balance")
PROBE_STEPS = 90
FEAT_EPS = 0.02        # 起伏 → [0,1) 的软刻度
FEAT_THR = 0.05        # 低于它 = 跨输入不带特征，不往前额叶送
SILENT = 1e-3          # 平均活动低于它 = 从不用
CONST_STD = 1e-3       # 跨输入起伏低于它 = 常亮/常灭


@dataclass
class Probe:
    image: np.ndarray
    signals: dict
    region: str | None
    label: str


# ------------------------------------------------------------------ 参考输入
def _base(sp, v=0.34):
    P, C = sp.px, sp.n_rgb
    img = np.full((P, P, C), v, np.float32)
    sig = {"audio": np.full(sp.n_audio, v, np.float32),
           "touch": np.full(sp.n_touch, v, np.float32),
           "balance": np.full(sp.n_balance, v, np.float32)}
    return img, sig


def _n_chan(sp, region: str) -> int:
    return {"vision": sp.px * sp.px * sp.n_rgb, "audio": sp.n_audio,
            "touch": sp.n_touch, "balance": sp.n_balance}[region]


def make_change_probes(sp, per_region: int = 5, seed: int = 0, alt: float = 0.67):
    """每个区抽若干通道，各改最小一档；第一条是基准输入。"""
    img0, sig0 = _base(sp)
    probes = [Probe(img0.copy(), {k: v.copy() for k, v in sig0.items()}, None, "base")]
    for region in REGIONS:
        n = _n_chan(sp, region)
        idx = np.unique(np.linspace(0, n - 1, min(per_region, n)).astype(int))
        for c in idx:
            img, sig = img0.copy(), {k: v.copy() for k, v in sig0.items()}
            if region == "vision":
                v = img.reshape(-1, sp.n_rgb)
                v[int(c) // sp.n_rgb, int(c) % sp.n_rgb] = alt
            else:
                sig[region][int(c)] = alt
            probes.append(Probe(img, sig, region, f"{region}:{int(c)}"))
    return probes


def make_random_probes(sp, n: int = 8, seed: int = 1):
    """随机输入（用来量"跨输入的起伏"，不是用来算相似度）。"""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        img = rng.random((sp.px, sp.px, sp.n_rgb)).astype(np.float32)
        sig = {"audio": rng.random(sp.n_audio).astype(np.float32),
               "touch": rng.random(sp.n_touch).astype(np.float32),
               "balance": rng.random(sp.n_balance).astype(np.float32)}
        out.append(Probe(img, sig, None, f"random:{i}"))
    return out


def observe(B, ui, probes, steps: int = PROBE_STEPS) -> np.ndarray:
    """逐条输入跑到稳态，返回 (输入数 × 神经元数) 的活动矩阵。"""
    out = np.zeros((len(probes), B.n), dtype=np.float32)
    for p, pr in enumerate(probes):
        B.act[:] = 0.0
        ui.set_image(pr.image)
        ui.set_signals(**pr.signals)
        for _ in range(steps):
            B.step(external=ui.drive())
        out[p] = B.act
    return out


# ------------------------------------------------------------------ 指标
def feature_score(acts: np.ndarray) -> np.ndarray:
    """跨输入的起伏 → [0,1)。常亮/常灭 ≈ 0，能区分输入 ≈ 1。"""
    std = acts.std(axis=0)
    return std / (std + FEAT_EPS)


def region_similarity(B, acts, ids: dict, probes, region: str) -> float:
    """前额叶对"该区不同通道变化"的响应图案两两 |余弦|；越低越分得开。"""
    g = ids["pfc_groups"][(region, 4)]
    g = g[B.alive[g]]
    sel = [k for k, p in enumerate(probes) if p.region == region]
    if len(g) == 0 or len(sel) < 2:
        return float("nan")
    D = acts[sel][:, g] - acts[0, g]
    nrm = np.linalg.norm(D, axis=1)
    keep = nrm > 1e-6
    if int(keep.sum()) < 2:
        return float("nan")
    M = D[keep] / nrm[keep, None]
    S = np.abs(M @ M.T)          # 用 |余弦|：反向对齐（负值）同样算"分得开"
    off = S[~np.eye(len(S), dtype=bool)]
    return float(off.mean())


def all_similarities(B, acts, ids: dict, probes) -> dict:
    return {r: region_similarity(B, acts, ids, probes, r) for r in REGIONS}


# ------------------------------------------------------------------ 删
def _cascade_empty(B, rounds: int = 4):
    """删掉"所有输入都没了"的中继/前额叶/回程神经元（它们永远收不到信号）。"""
    removed = []
    for _ in range(rounds):
        indeg = np.bincount(B.dst, minlength=B.n)
        cand = [i for i in range(B.n)
                if B.alive[i] and indeg[i] == 0
                and B.region[i].startswith(("hop_", "pfc_", "ret_", "wire_"))]
        if not cand:
            break
        B.remove(cand)
        removed += cand
    return removed


def prune_constant_pfc(B, acts, feat_thr: float = FEAT_THR,
                       group: int = GROUP_PFC_FWD) -> dict:
    """常亮/常灭的神经元不往前额叶送：删掉它进入 hop / 前额叶的边。"""
    feat = feature_score(acts)
    m = B.group == group
    drop = m & (feat[B.src] < feat_thr)
    n_drop = int(drop.sum())
    w_drop = float(B.w[drop].sum()) if n_drop else 0.0
    if n_drop:
        keep = ~drop
        B.src, B.dst, B.w = B.src[keep], B.dst[keep], B.w[keep]
        B.sgn, B.group = B.sgn[keep], B.group[keep]
    cascade = _cascade_empty(B)
    return {"edges_removed": n_drop, "weight_removed": w_drop,
            "cascade_removed": len(cascade),
            "sources_without_feature": int((feat < feat_thr).sum())}


def prune_useless(B, ids: dict, acts, mean_thr: float = SILENT,
                  std_thr: float = CONST_STD) -> dict:
    """只删各区第 2~4 层里：从不用（平均≈0）与常亮/常灭（跨输入 std≈0）的神经元。"""
    mean = acts.mean(axis=0)
    std = acts.std(axis=0)
    silent, constant = [], []
    for (region, depth), lay in ids["layers"].items():
        if depth < 2:
            continue
        lay = lay[B.alive[lay]]
        if len(lay) == 0:
            continue
        silent += list(lay[mean[lay] < mean_thr])
        constant += list(lay[(std[lay] < std_thr) & (mean[lay] >= mean_thr)])
    uniq = np.unique(np.r_[silent, constant]).astype(int)
    n = B.remove(uniq) if len(uniq) else 0
    cascade = _cascade_empty(B)
    return {"silent": len(silent), "constant": len(constant),
            "removed": n, "cascade_removed": len(cascade)}


# ------------------------------------------------------------------ 加
def grow_layer(B, ids: dict, acts, sp, geo: Geometry, region: str, depth: int,
               fraction: float = 0.10, k_strong: int = 2, k_rand: int = 2,
               seed: int = 0) -> np.ndarray:
    """在某区某一层加神经元：中间使用频段的父神经元 + 2 强 + 2 随机输入。"""
    rng = np.random.default_rng(seed)
    key = (region, depth)
    lay_all = ids["layers"][key]
    lay = lay_all[B.alive[lay_all]]
    if len(lay) == 0:
        return np.zeros(0, dtype=int)

    m = max(1, int(round(fraction * len(lay))))
    usage = acts[:, lay].mean(axis=0)
    lo, hi = np.percentile(usage, [40, 60])
    band = lay[(usage >= lo) & (usage <= hi)]
    if len(band) < m:
        band = lay
    # 中间频段里优先挑"起伏大"的（更有特征），而不是最亮的
    order = np.argsort(-acts[:, band].std(axis=0))
    parents = band[order[:m]]

    y0, y1 = geo.bands[region]
    dy = (y1 - y0) / max(len(lay) - 1, 1)
    dz = 0.8 / max(len(lay) - 1, 1)
    f, g = DEPTH_GAIN.get(depth, (0.25, 0.45))
    new_ids = []
    for pid in parents:
        e = np.nonzero(B.dst == pid)[0]
        if len(e) == 0:
            continue
        pos = B.xyz([pid])[0].copy()
        pos[1] = float(np.clip(pos[1] + rng.uniform(-0.5, 0.5) * dy, y0, y1))
        pos[2] = float(np.clip(pos[2] + rng.uniform(-0.5, 0.5) * dz, 0.10, 0.90))
        child = int(B.add(f"{region}_L{depth}", pos[None, :], thr=0.30, soft=0.40)[0])
        srcs, ws = B.src[e], B.w[e]
        order_e = np.argsort(-ws)
        strong = srcs[order_e[:k_strong]]
        rest = srcs[order_e[k_strong:]]
        extra = (rng.choice(rest, size=min(k_rand, len(rest)), replace=False)
                 if len(rest) else np.zeros(0, dtype=int))
        picks = np.unique(np.r_[strong, extra]).astype(int)
        if len(picks) == 0:
            B.remove([child])
            continue
        # 仍按出生规矩：新神经元只接自己 (y,z) 最近的这些输入
        B.connect([child], picks, k=len(picks), w_range=(0.06, 0.16),
                  axes=(1, 2), thr_frac=f, soft_frac=g)
        new_ids.append(child)

    new_ids = np.asarray(new_ids, dtype=int)
    if len(new_ids):
        ids["layers"][key] = np.r_[lay_all, new_ids]
        if sp.chain_mode == "one2one":
            # 专用线：新神经元各配一根自己的链，1:1 接到前额叶原来的汇聚端
            pp = B.xyz(new_ids)
            x0 = float(geo.x_in + 0.10)
            for hj, frac in enumerate((1 / 3, 2 / 3)):
                xh = x0 + (geo.x_pfc - x0) * frac
                hp = np.stack([np.full(len(pp), xh), pp[:, 1], pp[:, 2]],
                              axis=1).astype(np.float32)
                w = B.add(f"wire_{region}_L{depth}_g{hj}", hp, thr=0.0, soft=1.0)
                src = new_ids if hj == 0 else prev_w
                B.link(src, w, w_range=(1.0, 1.0), group=GROUP_PFC_FWD)
                prev_w = w
                ids["hops"][(region, depth)][hj] = np.r_[
                    ids["hops"][(region, depth)][hj], w]
            g = ids["pfc_groups"][(region, depth)]
            g = g[B.alive[g]]
            if len(g):
                B.connect(g, prev_w, k=3, w_range=(0.06, 0.16), axes=(1, 2),
                          group=GROUP_PFC_FWD,
                          exact=(sp.pick_mode == "exact"))
        else:
            # 仍走原路：接到这一层已有的最后一级中继，再由中继送前额叶
            hops = ids["hops"][(region, depth)][-1]
            hops = hops[B.alive[hops]]
            if len(hops):
                B.connect(hops, new_ids, k=3, w_range=(0.05, 0.12), axes=(1, 2),
                          group=GROUP_PFC_FWD)
    return new_ids


# ------------------------------------------------------------------ 演化一轮
def _mean_sim(sims: dict) -> float:
    vals = [v for v in sims.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def evolve(B, ids, ui, sp, geo: Geometry, rounds: int = 3, grow_frac: float = 0.10,
           per_region: int = 5, feature_n: int = 8, steps: int = PROBE_STEPS,
           seed: int = 0, verbose: bool = True):
    """先筛常亮，再按"最难分的那一区"逐轮加神经元；变差回滚。"""
    probes = make_change_probes(sp, per_region=per_region, seed=seed)
    rand = make_random_probes(sp, n=feature_n, seed=seed + 1)
    report: dict = {"start_neurons": int(B.n_alive), "start_edges": int(len(B.src)),
                    "rounds": []}

    if verbose:
        print(f"参考输入：变化 {len(probes)} 条 / 随机 {len(rand)} 条；"
              f"初始 {B.n_alive} 个神经元、{len(B.src):,} 条连接")
    acts_ch = observe(B, ui, probes, steps)
    acts_rand = observe(B, ui, rand, steps)
    acts_all = np.vstack([acts_ch, acts_rand])   # 常亮/常灭要跨所有参考输入看
    sims0 = all_similarities(B, acts_ch, ids, probes)
    report["before"] = {k: (None if not np.isfinite(v) else round(v, 4))
                        for k, v in sims0.items()}

    # ① 常亮/常灭：不往前额叶送
    st = prune_constant_pfc(B, acts_all)
    ref_calibrate(B, ids, ui, passes=2, steps=28)
    report["gate"] = st
    if verbose:
        print(f"① 常亮筛选：去掉 {st['edges_removed']:,} 条进中继/前额叶的边"
              f"（{st['sources_without_feature']} 个无特征来源），"
              f"级联删 {st['cascade_removed']} 个中继/前额叶神经元")

    # ② 从不用 / 常亮的神经元
    acts_rand = observe(B, ui, rand, steps)
    acts_all = np.vstack([observe(B, ui, probes, steps), acts_rand])
    st = prune_useless(B, ids, acts_all)
    if any(st.values()):
        ref_calibrate(B, ids, ui, passes=1, steps=28)
    report["prune"] = st
    if verbose:
        print(f"② 减神经元：静默 {st['silent']}、常亮/常灭 {st['constant']}，"
              f"实删 {st['removed']}（级联 {st['cascade_removed']}）")

    # ③ 逐轮给最难分的那一区加神经元
    sims = all_similarities(B, observe(B, ui, probes, steps), ids, probes)
    for r in range(rounds):
        finite = {k: v for k, v in sims.items() if np.isfinite(v)}
        if not finite:
            break
        region = max(finite, key=finite.get)
        depth = 4
        snapB, snapIds = copy.deepcopy(B), copy.deepcopy(ids)
        new = grow_layer(B, ids, acts_all, sp, geo,
                         region, depth, fraction=grow_frac, seed=seed + 100 + r)
        if len(new) == 0:
            B, ids, ui = snapB, snapIds, Interface(snapB, snapIds, sp)
            break
        ref_calibrate(B, ids, ui, passes=1, steps=28)
        sims_new = all_similarities(B, observe(B, ui, probes, steps), ids, probes)
        improved = _mean_sim(sims_new) < _mean_sim(sims) - 5e-3
        report["rounds"].append({
            "round": r + 1, "region": region, "depth": depth,
            "added": int(len(new)),
            "sim_before": round(finite[region], 4),
            "sim_after": (None if not np.isfinite(sims_new[region])
                          else round(sims_new[region], 4)),
            "kept": bool(improved),
        })
        if verbose:
            print(f"③ 第 {r + 1} 轮：{region} L{depth} 加 {len(new)} 个 → "
                  f"相似度 {finite[region]:.3f} → "
                  f"{sims_new.get(region, float('nan')):.3f}；"
                  f"{'保留' if improved else '回滚'}")
        if improved:
            sims = sims_new
        else:
            B, ids, ui = snapB, snapIds, Interface(snapB, snapIds, sp)
            break

    report["end_neurons"] = int(B.n_alive)
    report["end_edges"] = int(len(B.src))
    report["after"] = {k: (None if not np.isfinite(v) else round(v, 4))
                       for k, v in sims.items()}
    report["removed_total"] = int(report["start_neurons"] - report["end_neurons"]
                                  + sum(x.get("added", 0) for x in report["rounds"]
                                        if x.get("kept")))
    return B, ids, ui, report


def main(argv=None) -> int:
    full = "--full" in (argv if argv is not None else sys.argv)
    if full:
        sp = Spec()
    else:
        sp = Spec(px=6, n_level=6, n_audio=10, n_touch=16, n_balance=4,
                  n_muscle=2, n_force=5, k_near=8,
                  layer_ratio=(2.0, 4.0, 6.0), layer_floor=(96, 128, 160))
    B, ids, sp, geo = build(sp, Config(), verbose=True)
    ui = Interface(B, ids, sp)
    ref_calibrate(B, ids, ui)
    B, ids, ui, rep = evolve(B, ids, ui, sp, geo, rounds=3, grow_frac=0.10,
                             per_region=4 if not full else 5,
                             feature_n=6 if not full else 8)
    print("\n=== 前额叶对各区变化的相似度（越低越分得开）===")
    for r in REGIONS:
        b, a = rep["before"].get(r), rep["after"].get(r)
        print(f"  {r:8s} {b if b is not None else '--':>6} → "
              f"{a if a is not None else '--':>6}")
    print(f"\n神经元 {rep['start_neurons']} → {rep['end_neurons']}"
          f"；连接 {rep['start_edges']:,} → {rep['end_edges']:,}")
    with open(os.path.join(HERE, "structure_result.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
