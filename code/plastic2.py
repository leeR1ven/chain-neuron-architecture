# -*- coding: utf-8 -*-
"""可塑连接规则（按用户补充修正）：**只有"刚刚变化"的神经元才参与连接**。

用户指出：靠近输入层的神经元总是被激活（甚至一直亮）。若让"当前亮着"的神经元都能连，
它们会和几乎所有神经元连上 → 网络卡死（总是同样的神经元在连）。

所以参与连接的神经元必须是"刚变化的"：
    rising  = 上一帧灭、这一帧亮   （刚亮的）
    falling = 上一帧亮、这一帧灭   （刚灭的）
    always  = 两帧都亮 / 两帧都灭  → **不参与任何连接**（常亮的被彻底排除）

连接在 (falling → rising) 与 (rising ↔ rising) 之间产生；
增量 Δw = η·a_i·a_j / (1 + w/w_ref)^pow，**无上限、不归一化**；
另外每 T 拍所有**可塑边**整体降权 w ← w·γ（先天连接不动）。
"""
from __future__ import annotations

import numpy as np


class PlasticCfg:
    thr: float = 0.35          # 判定"亮"的门槛
    eta: float = 0.02          # 学习步长
    w_ref: float = 1.0
    pow: float = 1.0
    T: int = 200               # 每多少拍整体降权一次
    gamma: float = 0.99
    max_new_per_tick: int = 100
    prune: float = 1e-3


class Plastic2:
    def __init__(self, cfg: PlasticCfg | None = None):
        self.cfg = cfg or PlasticCfg()
        self.enabled = False
        self.edges: dict[tuple[int, int], float] = {}
        self.t = 0
        self._cursor = 0        # inject 的轮转游标

    # ---------------------------------------------------------------- 一拍
    def tick(self, prev_act: np.ndarray, act: np.ndarray):
        if not self.enabled:
            return
        c = self.cfg
        self.t += 1
        hi_p, hi = prev_act >= c.thr, act >= c.thr
        rising = np.nonzero(hi & ~hi_p)[0]
        falling = np.nonzero(hi_p & ~hi)[0]
        # 常亮的（hi_p & hi）与常灭的（~hi_p & ~hi）都不在 rising/falling 里 → 自动被排除
        amp = act.astype(np.float64)

        def bump(i, j):
            if i == j:
                return
            w = self.edges.get((i, j), 0.0)
            w += c.eta * amp[i] * amp[j] / (1.0 + w / c.w_ref) ** c.pow
            self.edges[(i, j)] = w

        # 灭的 → 刚亮的（先后）：**"先亮的"那个神经元要用它上一帧的活动**——
        # 它这一帧已经灭了（活动 0），用当前活动算增量会恒等于 0，边长不出来
        amp_prev = prev_act.astype(np.float64)
        for i in falling[:64]:
            for j in rising[:64]:
                if int(i) == int(j):
                    continue
                w = self.edges.get((int(i), int(j)), 0.0)
                w += c.eta * amp_prev[int(i)] * amp[int(j)] / (1.0 + w / c.w_ref) ** c.pow
                self.edges[(int(i), int(j))] = w
        # 刚亮的 ↔ 刚亮的（同时，但只限"刚变化"的）
        for a_ in range(min(len(rising), 48)):
            for b_ in range(a_ + 1, min(len(rising), 48)):
                bump(int(rising[a_]), int(rising[b_]))
                bump(int(rising[b_]), int(rising[a_]))
        # 周期整体降权（只降可塑边）
        if self.t % c.T == 0:
            for k in list(self.edges):
                self.edges[k] *= c.gamma
                if self.edges[k] < c.prune:
                    del self.edges[k]

    def inject(self, B):
        """把可塑边并进引擎（不存在的加为新边，已有的同步权重）。

        注：用游标轮转，每拍只处理 max_new_per_tick 条但**不会永远只处理前 100 条**
        （原来 list(edges)[:100] 会让第 100 条之后的边永远轮不到）。
        """
        if not self.enabled or not self.edges:
            return 0
        items = list(self.edges.items())
        n = len(items)
        start = self._cursor % n
        take = min(self.cfg.max_new_per_tick, n)
        batch = items[start:start + take]
        if len(batch) < take:                      # 到尾部后从头上补
            batch += items[:take - len(batch)]
        self._cursor = (start + take) % n
        added = 0
        for (i, j), w in batch:
            hit = np.nonzero((B.src == i) & (B.dst == j))[0]
            if len(hit):
                B.w[hit[0]] = w
            else:
                B.src = np.r_[B.src, np.int32(i)]
                B.dst = np.r_[B.dst, np.int32(j)]
                B.w = np.r_[B.w, np.float32(w)]
                B.sgn = np.r_[B.sgn, np.float32(1.0)]
                B.group = np.r_[B.group, np.int16(-1)]      # -1 = 可塑边
                B.n_edges_added = getattr(B, "n_edges_added", 0) + 1
                added += 1
        return added


# -------------------------------------------------------------------- 自测
def self_test():
    rng = np.random.default_rng(0)
    N = 40
    # 0 号神经元：**一直亮**（模拟输入层常亮）；1~20 号：随机变化；21~39 号：常灭
    seq = []
    for t in range(300):
        a = np.zeros(N)
        a[0] = 1.0
        a[1:21] = (rng.random(20) > 0.5) * rng.uniform(0.6, 1.0, 20)
        seq.append(a)
    pl = Plastic2()
    pl.enabled = True
    for t in range(1, len(seq)):
        pl.tick(seq[t - 1], seq[t])
    keys = np.array(list(pl.edges.keys())) if pl.edges else np.zeros((0, 2), int)
    n_with_always = int(((keys == 0).any(axis=1)).sum()) if len(keys) else 0
    print(f"可塑边总数 {len(pl.edges)}（其中与'常亮的 0 号'相连的 {n_with_always} 条 —— 应为 0）")
    print(f"最大值 {max(pl.edges.values()):.3f}（无上限）  "
          f"常灭神经元参与 {int((keys >= 21).any(axis=1).sum()) if len(keys) else 0} 条 —— 应为 0")


if __name__ == "__main__":
    self_test()
