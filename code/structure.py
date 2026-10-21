# -*- coding: utf-8 -*-
"""结构可塑性：链上动态增/减神经元 + "常亮不带特征"的前额叶筛选。

不改链的走法（温度计编码、就近 (y,z) 连线、逐神经元门槛/带宽、
每一层经中继送前额叶、回程调制、运动区分档、门控都保持原样），只改"链上有多少神经元、
哪些信号继续往前额叶送"：

  1. **加神经元**：照立体网络最新规则，加在**最后一层**。
     立体网络是 L5：取使用率 40–60% 的父单元，5% 父单元 → 每个拆 2 个子单元、
     各拿 2 条输入（2+2，总数 +10%）。链里的最后一层是**前额叶汇聚端**，不是 L4：
     全尺寸实测加 L4 变差（0.305→0.337），加前额叶稳定变好（0.330→0.314）。
     子单元直接接原来喂父单元的那些链（仍走专用线/中继）。
  2. **减神经元**：只删两类——
       跨输入几乎不激活的（没用），
       跨输入始终同一读数的（常亮/常灭，不带特征）。
     删完把它带死的 hop / 前额叶神经元级联删掉。
  3. **常亮不送前额叶**：用一批随机输入看**跨输入的起伏**（不是时间上的变化）。
     只筛"平均 > 0.5、std≈0"的常亮来源，把它进 hop / 前额叶的边去掉；
     常灭的不在这里砍（参考输入不足时会误伤，完整规模实测过）。
  4. 每次结构改动后重新标定门槛（步数必须和初始一致，28 步会把深层相似度带偏），
     再复测前额叶四区相似度；定点改动只看目标区变好、别区不显著变差，否则回滚。

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
                       group: int = GROUP_PFC_FWD, always_on: bool = True,
                       cascade: bool = False) -> dict:
    """常亮（默认只筛"总是亮"）的神经元不往前额叶送：删掉它进入 hop / 前额叶的边。

    只筛"总是亮"是因为：常灭的神经元确实没用，但**参考输入太少时**，
    一个只是"这批输入没碰到"的神经元也会量成常灭；把它当前额叶入边删掉，
    等于用样本不足的统计去砍通路（完整规模上实测：筛完前额叶反而更分不开）。
    cascade 默认关：删边后没有输入的中继/前额叶神经元只是静默，留着不会影响余弦，
    但删掉会改变前额叶响应向量的维度，反而把相似度指标带偏。
    """
    feat = feature_score(acts)
    mean = acts.mean(axis=0)
    weak = feat < feat_thr
    if always_on:
        weak = weak & (mean > 0.5)
    m = B.group == group
    drop = m & weak[B.src]
    n_drop = int(drop.sum())
    w_drop = float(B.w[drop].sum()) if n_drop else 0.0
    if n_drop:
        keep = ~drop
        B.src, B.dst, B.w = B.src[keep], B.dst[keep], B.w[keep]
        B.sgn, B.group = B.sgn[keep], B.group[keep]
    dead = _cascade_empty(B) if cascade else []
    return {"edges_removed": n_drop, "weight_removed": w_drop,
            "cascade_removed": len(dead),
            "sources_without_feature": int(weak.sum())}


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
    # acts 可能是在这一轮之前量的（新加的神经元还没有活动记录）——只拿有记录的老神经元当父本
    lay = lay_all[B.alive[lay_all] & (lay_all < acts.shape[1])]
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
def _rebuild_pfc_index(ids: dict) -> None:
    """加了前额叶神经元后，重建按区/按类的索引（gate 用的 other_pfc 也要跟着更新）。"""
    def layers(r):
        return [ids["pfc_groups"][(r, d)] for d in (1, 2, 3, 4)]

    ids["pfc_vision"] = np.concatenate(layers("vision"))
    ids["pfc_other"] = np.concatenate([g for r in ("audio", "touch", "balance")
                                       for g in layers(r)])
    ids["pfc"] = np.r_[ids["pfc_vision"], ids["pfc_other"]]
    motor = np.concatenate(ids["motor_pfc"])
    ids["other_pfc"] = ids["pfc"][~np.isin(ids["pfc"], motor)]


def grow_pfc(B, ids: dict, acts, sp, geo: Geometry, region: str, depth: int = 4,
             fraction: float = 0.05, children: int = 2, k_in: int = 2,
             seed: int = 0) -> np.ndarray:
    """按立体网络的增宽规则加"最后一层"神经元（链里就是前额叶汇聚端）。

    立体网络的规则：N≥5000 时，取使用率 40–60% 的中间频段父单元，
    每个父单元拆成 2 个子单元、各拿 2 条输入（2+2），总数 +10%。
    这里照搬：父单元是前额叶神经元，子单元仍前额叶；输入就是原来喂它的那些链。
    """
    rng = np.random.default_rng(seed)
    key = (region, depth)
    g_all = ids["pfc_groups"][key]
    g = g_all[B.alive[g_all] & (g_all < acts.shape[1])]
    if len(g) == 0:
        return np.zeros(0, dtype=int)
    m = max(1, int(round(fraction * len(g))))
    usage = acts[:, g].mean(axis=0)
    lo, hi = np.percentile(usage, [40, 60])
    band = g[(usage >= lo) & (usage <= hi)]
    if len(band) < m:
        band = g
    parents = band[np.argsort(-acts[:, band].std(axis=0))[:m]]

    f0, g0 = DEPTH_GAIN.get(depth, (0.25, 0.45))
    f, gg = 0.3 * f0, 0.5 * g0      # 和 build 里前额叶那一层的门槛/带宽比例一致
    new_ids = []
    for pid in parents:
        e = np.nonzero(B.dst == pid)[0]
        if len(e) == 0:
            continue
        order = np.argsort(-B.w[e])
        srcs, ws = B.src[e][order], B.w[e][order]
        for c in range(children):
            sl = slice(c * k_in, (c + 1) * k_in)
            S, W = srcs[sl], ws[sl]
            if len(S) == 0:
                continue
            pos = B.xyz([pid])[0].copy()
            pos[1] = float(np.clip(pos[1] + rng.uniform(-0.002, 0.002), 0.0, 1.0))
            pos[2] = float(np.clip(pos[2] + rng.uniform(-0.002, 0.002), 0.10, 0.90))
            child = int(B.add(f"pfc_{region}_L{depth}", pos[None, :],
                              thr=0.30, soft=0.35)[0])
            B.add_edges(S, [child] * len(S), W, group=GROUP_PFC_FWD)
            s = float(np.abs(W).sum())
            B.thr[child] = max(1e-4, f * s)
            B.soft[child] = max(1e-4, gg * s)
            new_ids.append(child)
    new_ids = np.asarray(new_ids, dtype=int)
    if len(new_ids):
        ids["pfc_groups"][key] = np.r_[g_all, new_ids]
        _rebuild_pfc_index(ids)
    return new_ids


def _mean_sim(sims: dict) -> float:
    vals = [v for v in sims.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def evolve(B, ids, ui, sp, geo: Geometry, rounds: int = 3, grow_frac: float = 0.10,
           grow_target: str = "pfc", per_region: int = 5, feature_n: int = 8,
           steps: int = PROBE_STEPS, post_passes: int = 2, post_steps: int = 40,
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
    sims = sims0

    # ①② 常亮不送额叶 + 减静默/常亮神经元：一起改，复测四区相似度；整体变差就整块回滚
    snapB, snapIds = copy.deepcopy(B), copy.deepcopy(ids)
    st_gate = prune_constant_pfc(B, acts_all)
    st_prune = prune_useless(B, ids, acts_all)
    if any(st_gate.values()) or any(st_prune.values()):
        ref_calibrate(B, ids, ui, passes=post_passes, steps=post_steps)
        acts_ch = observe(B, ui, probes, steps)
        acts_rand = observe(B, ui, rand, steps)
        sims1 = all_similarities(B, acts_ch, ids, probes)
    else:
        sims1 = sims0
    if _mean_sim(sims1) > _mean_sim(sims0) + 0.01:
        B, ids, ui = snapB, snapIds, Interface(snapB, snapIds, sp)
        report["structure_skipped"] = True
        if verbose:
            print("①② 常亮筛选 + 减神经元：四区整体变差，已回滚（不动结构）")
    else:
        sims = sims1
        report["gate"], report["prune"] = st_gate, st_prune
        if verbose:
            print(f"① 常亮筛选：去掉 {st_gate['edges_removed']:,} 条进中继/前额叶的边"
                  f"（{st_gate['sources_without_feature']} 个常亮来源），"
                  f"级联删 {st_gate['cascade_removed']} 个中继/前额叶神经元")
            print(f"② 减神经元：静默 {st_prune['silent']}、常亮/常灭 {st_prune['constant']}，"
                  f"实删 {st_prune['removed']}（级联 {st_prune['cascade_removed']}）")

    # 加神经元要用改动后的活动矩阵
    acts_all = np.vstack([observe(B, ui, probes, steps),
                          observe(B, ui, rand, steps)])

    # ③ 逐轮给最难分的那一区加神经元
    for r in range(rounds):
        finite = {k: v for k, v in sims.items() if np.isfinite(v)}
        if not finite:
            break
        region = max(finite, key=finite.get)
        depth = 4
        snapB, snapIds = copy.deepcopy(B), copy.deepcopy(ids)
        if grow_target == "pfc":
            # 立体网络是在**最后一层**（L5）加神经元：链里的最后一层就是前额叶汇聚端。
            # 规则照搬：取使用率 40–60% 的父单元，5% 父单元 → 每个拆 2 个子单元、各拿 2 条输入。
            new = grow_pfc(B, ids, acts_all, sp, geo, region, depth,
                           fraction=0.05, children=2, k_in=2, seed=seed + 100 + r)
        else:
            new = grow_layer(B, ids, acts_all, sp, geo,
                             region, depth, fraction=grow_frac, seed=seed + 100 + r)
        if len(new) == 0:
            B, ids, ui = snapB, snapIds, Interface(snapB, snapIds, sp)
            break
        ref_calibrate(B, ids, ui, passes=post_passes, steps=post_steps)
        sims_new = all_similarities(B, observe(B, ui, probes, steps), ids, probes)
        # 结构改动是**定点**的：看目标区有没有变好，而不是要求四区平均都变好
        # （加 16 个听觉神经元把听觉从 0.305 拉到 0.287，但重标定会让别的区抖 0.01，
        #  按"平均"判会把这种真提升误判成变差）
        delta = (sims.get(region, np.nan) - sims_new.get(region, np.nan))
        others = [sims_new[k] - sims[k] for k in sims
                  if k != region and np.isfinite(sims[k]) and np.isfinite(sims_new[k])]
        worst_other = max(others) if others else 0.0
        improved = bool(np.isfinite(delta) and delta > 0.01 and worst_other < 0.02)
        report["rounds"].append({
            "round": r + 1, "region": region, "depth": depth,
            "mode": grow_target,
            "added": int(len(new)),
            "sim_before": round(finite[region], 4),
            "sim_after": (None if not np.isfinite(sims_new[region])
                          else round(sims_new[region], 4)),
            "kept": bool(improved),
        })
        if verbose:
            print(f"③ 第 {r + 1} 轮：{grow_target} {region} L{depth} 加 {len(new)} 个 → "
                  f"相似度 {finite[region]:.3f} → "
                  f"{sims_new.get(region, float('nan')):.3f}；"
                  f"{'保留' if improved else '回滚'}")
        if improved:
            sims = sims_new
            if r + 1 < rounds:
                # 下轮要用新的活动矩阵（这一轮加了神经元）
                acts_all = np.vstack([observe(B, ui, probes, steps),
                                      observe(B, ui, rand, steps)])
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
