# -*- coding: utf-8 -*-
"""结构可塑性：链上动态增/减神经元 + "常亮不带特征"的前额叶筛选。

不改链的走法（温度计编码、就近 (y,z) 连线、逐神经元门槛/带宽、
每一层经中继送前额叶、回程调制、运动区分档、门控都保持原样），只改"链上有多少神经元、
哪些信号继续往前额叶送"：

  1. **加神经元**：链的区内链是 L1 传感器 + L2–L5；候选层就是 L2–L5，逐层试、按目标区相似度选层。
     每层规则：使用率 40–60% 的父单元，5% 父单元 → 每个拆 2 个子单元、各拿 2 条输入；
     子单元局部重连下一层，并各配专用线送前额叶；每加一个子单元，
     给这一层的前额叶组 +1 个接收神经元（前额叶是落点，不是候选层）。
     加完**不重标定整网**（实测重标会把提升洗掉：L3 0.192→0.178，重标后 0.203）。
     实测：中等规模 L2 0.566 / L3 0.543 / L4 0.532 / L5 0.511——越深越好。
  2. **减神经元/筛常亮**：只对"最新网络"的区做（默认 `gate_regions=("vision",)`）。
     听觉/触觉/平衡的专用网络还没做出来（现在是旧的），大量细胞在参考输入下静默，
     如果按"静默就删"会把旧网络连带打塌；所以这三个区先不动结构。
     筛常亮只筛"平均 > 0.5、std≈0"的来源（常灭的走删除，不在这里砍）。
  3. **加神经元后不重标定**：只给新神经元/新接收端设门槛。实测整体重标会把增益洗掉
     （L3 0.192→0.178，重标后 0.203）；加完直接复测四区相似度，按前额叶组保护（任一组掉 >0.02 就回滚）。

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


def region_similarity_at(B, acts, ids: dict, probes, region: str, depth: int) -> float:
    """只看某一层那一组前额叶的相似度（用来防止"总码变好、某一组塌掉"）。"""
    g = ids["pfc_groups"][(region, depth)]
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
    S = np.abs(M @ M.T)
    off = S[~np.eye(len(S), dtype=bool)]
    return float(off.mean())


def region_group_sims(B, acts, ids: dict, probes, region: str) -> dict:
    ds = sorted(d for (r, d) in ids["pfc_groups"] if r == region)
    return {d: region_similarity_at(B, acts, ids, probes, region, d) for d in ds}


def region_similarity(B, acts, ids: dict, probes, region: str,
                      all_groups: bool = False) -> float:
    """前额叶对"该区不同通道变化"的响应图案两两 |余弦|；越低越分得开。

    all_groups=True：把这个区 L1–L4 的前额叶组拼起来看
    （加在 L2/L3 的神经元只进各自那一组，只看 L4 组会看不到）。
    """
    if all_groups:
        ds = sorted(d for (r, d) in ids["pfc_groups"] if r == region)
        g = np.concatenate([ids["pfc_groups"][(region, d)] for d in ds])
    else:
        g = ids["pfc_groups"][(region, max(d for (r, d) in ids["pfc_groups"]
                                           if r == region))]
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


def all_similarities(B, acts, ids: dict, probes, all_groups: bool = False) -> dict:
    return {r: region_similarity(B, acts, ids, probes, r, all_groups=all_groups)
            for r in REGIONS}


def region_margin(B, ui, ids: dict, sp, region: str, n: int = 8, steps: int = PROBE_STEPS,
                  seed: int = 0) -> dict:
    """区分强度（立体网络用的 margin）：同一底图小改动 vs 不同底图的距离比。

    用这个区**所有层的前额叶组**拼成的码（L1–L4），这样加在任何一层都能被看到。
    返回 ratio（同底图/不同底图，越低越分得开）和 min_diff（不同底图的最小距离，越高越好）。
    """
    rng = np.random.default_rng(seed)
    groups = [ids["pfc_groups"][(region, d)] for d in (1, 2, 3, 4)]
    code_ids = np.concatenate([g[B.alive[g]] for g in groups])

    def rand_input():
        img = rng.random((sp.px, sp.px, sp.n_rgb)).astype(np.float32)
        sig = {"audio": rng.random(sp.n_audio).astype(np.float32),
               "touch": rng.random(sp.n_touch).astype(np.float32),
               "balance": rng.random(sp.n_balance).astype(np.float32)}
        return img, sig

    def run(img, sig):
        B.act[:] = 0.0
        ui.set_image(img)
        ui.set_signals(**sig)
        for _ in range(steps):
            B.step(external=ui.drive())
        return B.act[code_ids].astype(np.float64)

    base_codes, same = [], []
    for _ in range(n):
        img, sig = rand_input()
        c0 = run(img, sig)
        base_codes.append(c0)
        img2 = img.copy()
        sig2 = {k: v.copy() for k, v in sig.items()}
        if region == "vision":
            v = img2.reshape(-1, sp.n_rgb)
            px = int(rng.integers(0, v.shape[0]))
            ch = int(rng.integers(0, sp.n_rgb))
            v[px, ch] = float(np.clip(v[px, ch] + rng.choice([-0.33, 0.33]), 0, 1))
        else:
            c = int(rng.integers(0, len(sig2[region])))
            sig2[region][c] = float(np.clip(sig2[region][c]
                                            + rng.choice([-0.33, 0.33]), 0, 1))
        same.append(float(np.abs(run(img2, sig2) - c0).sum()))
    M = np.stack(base_codes)
    D = np.abs(M[:, None, :] - M[None, :, :]).sum(axis=2)
    iu = np.triu_indices(n, 1)
    min_diff = float(D[iu].min()) if len(iu[0]) else 0.0
    med_diff = float(np.median(D[iu])) if len(iu[0]) else 0.0
    same_m = float(np.mean(same))
    return {"same": same_m, "min_diff": min_diff, "median_diff": med_diff,
            "ratio": same_m / max(min_diff, 1e-9)}


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


def _region_of(region_name: str) -> str:
    """wire_audio_L5_0 / pfc_audio_L5 / hop_audio_L5_0 / ret_audio_L5_0 -> audio"""
    parts = region_name.split("_")
    return parts[1] if len(parts) > 1 else ""


def prune_constant_pfc(B, acts, feat_thr: float = FEAT_THR,
                       group: int = GROUP_PFC_FWD, always_on: bool = True,
                       cascade: bool = False, regions=None) -> dict:
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
    if regions is not None:
        dst_reg = np.array([_region_of(B.region[int(d)]) for d in B.dst])
        drop = drop & np.isin(dst_reg, list(regions))
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
                  std_thr: float = CONST_STD, cascade: bool = True,
                  regions=None) -> dict:
    """只删各区第 2~4 层里：从不用（平均≈0）与常亮/常灭（跨输入 std≈0）的神经元。"""
    mean = acts.mean(axis=0)
    std = acts.std(axis=0)
    silent, constant = [], []
    for (region, depth), lay in ids["layers"].items():
        if depth < 2:
            continue
        if regions is not None and region not in regions:
            continue
        lay = lay[B.alive[lay]]
        if len(lay) == 0:
            continue
        silent += list(lay[mean[lay] < mean_thr])
        constant += list(lay[(std[lay] < std_thr) & (mean[lay] >= mean_thr)])
    uniq = np.unique(np.r_[silent, constant]).astype(int)
    n = B.remove(uniq) if len(uniq) else 0
    dead = _cascade_empty(B) if cascade else []
    return {"silent": len(silent), "constant": len(constant),
            "removed": n, "cascade_removed": len(dead)}


# ------------------------------------------------------------------ 加
def grow_layer(B, ids: dict, acts, sp, geo: Geometry, region: str, depth: int,
               fraction: float = 0.05, children: int = 2, k_in: int = 2,
               k_down: int = 3, seed: int = 0) -> np.ndarray:
    """在某一层加神经元（照立体网络 stable-ID 的做法，L2–L4 都能加）。

    - 取使用率 40–60% 的父单元，5% 父单元；
    - 每个父单元拆 2 个子单元，各拿父单元输入里的 2 条（2+2，权重照抄）；
    - **局部重连下游**：子单元往下一层最近的几个神经元各加一条边（L4 的下一层就是前额叶；
      走专用线送上去，所以 L4 这里只加线、不重连层内）；
    - 每个子单元仍配一根自己的专用线到这一层的前额叶组（"每层都往前额叶送"）。
    """
    rng = np.random.default_rng(seed)
    key = (region, depth)
    lay_all = ids["layers"][key]
    lay = lay_all[B.alive[lay_all] & (lay_all < acts.shape[1])]
    if len(lay) == 0:
        return np.zeros(0, dtype=int)

    m = max(1, int(round(fraction * len(lay))))
    usage = acts[:, lay].mean(axis=0)
    lo, hi = np.percentile(usage, [40, 60])
    band = lay[(usage >= lo) & (usage <= hi)]
    if len(band) < m:
        band = lay
    parents = band[np.argsort(-acts[:, band].std(axis=0))[:m]]

    y0, y1 = geo.bands[region]
    dy = (y1 - y0) / max(len(lay) - 1, 1)
    dz = 0.8 / max(len(lay) - 1, 1)
    f0, g0 = DEPTH_GAIN.get(depth, (0.25, 0.45))
    hops = ids["hops"][(region, depth)]
    g_pfc = ids["pfc_groups"][(region, depth)]
    f2, g2 = 0.3 * f0, 0.5 * g0      # 前额叶接收端的门槛/带宽比例（同 build）
    x0 = float(geo.x_in + 0.10)
    new_ids = []
    for pid in parents:
        e = np.nonzero(B.dst == pid)[0]
        if len(e) == 0:
            continue
        order = np.argsort(-B.w[e])
        srcs, ws = B.src[e][order], B.w[e][order]
        pos0 = B.xyz([pid])[0]
        for c in range(children):
            S, W = srcs[c * k_in:(c + 1) * k_in], ws[c * k_in:(c + 1) * k_in]
            if len(S) == 0:
                continue
            pos = pos0.copy()
            pos[1] = float(np.clip(pos[1] + rng.uniform(-0.5, 0.5) * dy, y0, y1))
            pos[2] = float(np.clip(pos[2] + rng.uniform(-0.5, 0.5) * dz, 0.10, 0.90))
            child = int(B.add(f"{region}_L{depth}", pos[None, :], thr=0.30, soft=0.40)[0])
            B.add_edges(S, [child] * len(S), W)
            s = float(np.abs(W).sum())
            B.thr[child] = max(1e-4, f0 * s)
            B.soft[child] = max(1e-4, g0 * s)
            # 局部重连下游（不是最后一层时；最后一层的下一层就是前额叶，用下面的专用线）
            last = max(d for (r, d) in ids["layers"] if r == region)
            if depth < last:
                nxt = ids["layers"][(region, depth + 1)]
                nxt = nxt[B.alive[nxt]]
                if len(nxt):
                    dd = np.linalg.norm(B.xyz(nxt)[:, 1:3] - pos[None, 1:3], axis=1)
                    tg = nxt[np.argsort(dd)[:min(k_down, len(nxt))]]
                    B.add_edges([child] * len(tg), tg,
                                rng.uniform(0.06, 0.16, len(tg)))
            # 专用线：子单元 → 两个中继 → 这一层的前额叶组
            prev = np.asarray([child], dtype=int)
            wlist = []
            for hj, frac in enumerate((1 / 3, 2 / 3)):
                xh = x0 + (geo.x_pfc - x0) * frac
                hp = np.array([[xh, pos[1], pos[2]]], dtype=np.float32)
                wnode = int(B.add(f"wire_{region}_L{depth}_g{hj}", hp,
                                  thr=0.0, soft=1.0)[0])
                B.link(prev, np.asarray([wnode]), w_range=(1.0, 1.0),
                       group=GROUP_PFC_FWD)
                hops[hj] = np.r_[hops[hj], wnode]
                wlist.append(wnode)
                prev = np.asarray([wnode])
            g = g_pfc[B.alive[g_pfc]]
            # 注意：**不要**把新线接到所有已有的前额叶神经元上——那会给每个老神经元
            # 每个子单元加 3 条边（32 个子单元 = 96 条），一下把它们全推到饱和。
            # 新信号的落点是下面新加的那个接收神经元。
            # 前额叶接收端也按比例 +1：链的终点跟着扩容（相当于立体网络加完神经元要扩 top-k），
            # 它不是"另一层"，是这一层信号的落点；输入就接这条子单元自己的两根中继。
            gp = ids["pfc_groups"][(region, depth)]
            if len(wlist):
                pfc_pos = np.array([[geo.x_pfc, pos[1], pos[2]]], dtype=np.float32)
                pc = int(B.add(f"pfc_{region}_L{depth}", pfc_pos,
                               thr=0.0, soft=1.0)[0])
                B.connect([pc], np.asarray(wlist), k=len(wlist),
                          w_range=(0.6, 0.6), axes=(1, 2),
                          group=GROUP_PFC_FWD, exact=(sp.pick_mode == "exact"))
                # 新接收端按上游实际活动量级定门槛：thr=0、带宽=2×入边和（原样收，不压死）
                B.thr[pc] = 0.0
                B.soft[pc] = max(1e-4, 2.0 * 0.6 * len(wlist))
                ids["pfc_groups"][(region, depth)] = np.r_[gp, pc]
                _rebuild_pfc_index(ids)
            new_ids.append(child)

    new_ids = np.asarray(new_ids, dtype=int)
    if len(new_ids):
        ids["layers"][key] = np.r_[lay_all, new_ids]
    return new_ids


# ------------------------------------------------------------------ 演化一轮
def _rebuild_pfc_index(ids: dict) -> None:
    """加了前额叶神经元后，重建按区/按类的索引（gate 用的 other_pfc 也要跟着更新）。"""
    def layers(r):
        ds = sorted(d for (rr, d) in ids["pfc_groups"] if rr == r)
        return [ids["pfc_groups"][(r, d)] for d in ds]

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
           grow_target: str = "auto", per_region: int = 5, feature_n: int = 8,
           steps: int = PROBE_STEPS, post_passes: int = 2, post_steps: int = 40,
           gate_regions: tuple = ("vision",),
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

    # ①② 常亮不送额叶 + 减静默/常亮神经元：**不重标定**，改完直接看每一层前额叶组有没有掉。
    # 只对"最新网络"的区做筛/删（默认视觉）；听觉/触觉/平衡的专用网络还没做出来，先不动结构，
    # 否则旧网络里大量静默细胞会被当成"没用"删掉、连带把它们的额叶组打塌。
    s_g0 = {r: region_group_sims(B, acts_ch, ids, probes, r) for r in REGIONS}
    snapB, snapIds = copy.deepcopy(B), copy.deepcopy(ids)
    st_gate = prune_constant_pfc(B, acts_all, regions=gate_regions)
    st_prune = prune_useless(B, ids, acts_all, regions=gate_regions)
    if any(st_gate.values()) or any(st_prune.values()):
        acts_ch = observe(B, ui, probes, steps)
        acts_rand = observe(B, ui, rand, steps)
        sims1 = all_similarities(B, acts_ch, ids, probes)
        s_g1 = {r: region_group_sims(B, acts_ch, ids, probes, r) for r in REGIONS}
        drops = []
        for r in REGIONS:
            for d, v0 in s_g0[r].items():
                v1 = s_g1[r].get(d, np.nan)
                if not np.isfinite(v0):
                    continue
                drops.append((v1 - v0) if np.isfinite(v1) else -9.0)
        worst = min(drops) if drops else 0.0
    else:
        sims1 = sims0
        worst = 0.0
    if worst < -0.02 or _mean_sim(sims1) > _mean_sim(sims0) + 0.02:
        B, ids, ui = snapB, snapIds, Interface(snapB, snapIds, sp)
        report["structure_skipped"] = True
        if verbose:
            print(f"①② 常亮筛选 + 减神经元：有一组前额叶掉 {worst:.3f}，已回滚")
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
    # 选层要看这个区**所有层**的前额叶组（加在 L2/L3 只进各自那一组，只看 L4 会看不到）
    sims_all = all_similarities(B, acts_all[:len(probes)], ids, probes, all_groups=True)

    # ③ 逐轮给最难分的那一区加神经元（候选是区内的 L2/L3/L4；前额叶是落点，不是候选层）
    for r in range(rounds):
        finite = {k: v for k, v in sims.items() if np.isfinite(v)}
        if not finite:
            break
        region = max(finite, key=finite.get)
        depth = 4
        s_before_g = region_group_sims(B, acts_all[:len(probes)], ids, probes, region)
        snapB, snapIds = copy.deepcopy(B), copy.deepcopy(ids)
        added = np.zeros(0, dtype=int)
        mode_used, depth_used = grow_target, 4
        margin_before = margin_after = None
        if grow_target == "auto":
            margin_before = region_margin(B, ui, ids, sp, region, n=8, steps=steps,
                                          seed=seed + r)
            cands = [("layer", d) for d in range(2, 2 + len(sp.layer_ratio))]
            best = None
            for cmode, cdepth in cands:
                B, ids = copy.deepcopy(snapB), copy.deepcopy(snapIds)
                ui = Interface(B, ids, sp)
                new = grow_layer(B, ids, acts_all, sp, geo, region, cdepth,
                                 fraction=0.05, children=2, k_in=2,
                                 seed=seed + 100 + r)
                if len(new) == 0:
                    continue
                # 加完**不重标定**：只给新神经元/新接收端设了门槛，老神经元不动
                # （实测重标会把加层带来的提升洗掉：L3 0.192→0.178 变好，重标后又回到 0.203）
                sim_try = region_similarity(B, observe(B, ui, probes, steps),
                                            ids, probes, region, all_groups=True)
                if verbose:
                    print(f"    试 L{cdepth}: 区内 +{len(new)}（含中继/接收端） "
                          f"该区相似度 {sims_all.get(region, float('nan')):.4f} → {sim_try:.4f}")
                if best is None or sim_try < best[5]:
                    best = (copy.deepcopy(B), copy.deepcopy(ids), cmode, cdepth,
                            new, sim_try)
            if best is None:
                B, ids, ui = snapB, snapIds, Interface(snapB, snapIds, sp)
                break
            B, ids, mode_used, depth_used, added, _ = best
            ui = Interface(B, ids, sp)
            margin_after = region_margin(B, ui, ids, sp, region, n=8, steps=steps,
                                         seed=seed + r)
            sims_new = all_similarities(B, observe(B, ui, probes, steps), ids, probes)
            sims_new_all = all_similarities(B, observe(B, ui, probes, steps), ids,
                                            probes, all_groups=True)
            # 防"总码变好、某一层的前额叶组塌掉"：逐组比较，最差的那组不许掉超过 0.02
            acts_b = observe(B, ui, probes, steps)
            s_after_g = region_group_sims(B, acts_b, ids, probes, region)
            drops = []
            for d, v0 in s_before_g.items():
                v1 = s_after_g.get(d, np.nan)
                if not np.isfinite(v0):
                    continue
                drops.append((v1 - v0) if np.isfinite(v1) else -9.0)
            worst_group = min(drops) if drops else 0.0
            delta = (sims_all.get(region, np.nan) - sims_new_all.get(region, np.nan))
            others = [sims_new_all[k] - sims_all[k] for k in sims_all
                      if k != region and np.isfinite(sims_all[k])
                      and np.isfinite(sims_new_all[k])]
            worst_other = max(others) if others else 0.0
            improved = bool(np.isfinite(delta) and delta > 0.01
                            and worst_group > -0.02 and worst_other < 0.02)
            if improved:
                sims_all = sims_new_all
        else:
            last = max(d for (r, d) in ids["layers"] if r == region)
            if grow_target == "pfc":
                added = grow_pfc(B, ids, acts_all, sp, geo, region, last,
                                 fraction=0.05, children=2, k_in=2,
                                 seed=seed + 100 + r)
            else:
                added = grow_layer(B, ids, acts_all, sp, geo, region, last,
                                   fraction=0.05, children=2, k_in=2,
                                   seed=seed + 100 + r)
            if len(added) == 0:
                B, ids, ui = snapB, snapIds, Interface(snapB, snapIds, sp)
                break
            # 同样不重标定：只让新神经元用自己的门槛
            sims_new = all_similarities(B, observe(B, ui, probes, steps), ids, probes)
            delta = (sims.get(region, np.nan) - sims_new.get(region, np.nan))
            others = [sims_new[k] - sims[k] for k in sims
                      if k != region and np.isfinite(sims[k]) and np.isfinite(sims_new[k])]
            worst_other = max(others) if others else 0.0
            improved = bool(np.isfinite(delta) and delta > 0.01 and worst_other < 0.02)

        report["rounds"].append({
            "round": r + 1, "region": region, "depth": depth_used,
            "mode": mode_used,
            "added": int(len(added)),
            "margin_before": (None if margin_before is None
                              else {k: round(v, 4) for k, v in margin_before.items()}),
            "margin_after": (None if margin_after is None
                             else {k: round(v, 4) for k, v in margin_after.items()}),
            "sim_before": round(finite[region], 4),
            "sim_after": (None if not np.isfinite(sims_new[region])
                          else round(sims_new[region], 4)),
            "kept": bool(improved),
        })
        if verbose:
            extra = ""
            if margin_before is not None and margin_after is not None:
                extra = (f" margin {margin_before['ratio']:.4f} → "
                         f"{margin_after['ratio']:.4f}")
            print(f"③ 第 {r + 1} 轮：{mode_used} {region} L{depth_used} 加 "
                  f"{len(added)} 个 → 相似度 {finite[region]:.3f} → "
                  f"{sims_new.get(region, float('nan')):.3f}{extra}；"
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
                  layer_ratio=(2.0, 4.0, 6.0, 8.0), layer_floor=(96, 128, 160, 192))
    B, ids, sp, geo = build(sp, Config(), verbose=True)
    ui = Interface(B, ids, sp)
    ref_calibrate(B, ids, ui)
    B, ids, ui, rep = evolve(B, ids, ui, sp, geo, rounds=2, grow_frac=0.10,
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
