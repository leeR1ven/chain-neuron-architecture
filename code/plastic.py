# -*- coding: utf-8 -*-
"""后天可塑连接（严格按用户定的三条规则）。

规则一：时序赫布增长
    · 上一步还亮着、这一步刚亮的 → 从"先亮的"指向"刚亮的"加强
    · 同一步一起亮的         → 互相加强
    · 增量  Δw = η·a_i·a_j / (1 + w/w_ref)^pow
      — 没有上限（不加 1.0 封顶）
      — 不做任何归一化
      — 双向是"顺序 + 同时"两种情况自然给出的，不额外加反向学习机制

规则二：周期整体降权
    · 每隔 T 拍，把所有**可改变的**连接  w ← w·γ
    · **出生连接（先天固定连接）不动** —— 骨架不能被磨掉
    · 与规则一叠加：常用的净增长、少用的趋近 0；不需要任何"哪条该降"的判断

规则三：新增连接
    · 允许在共同活跃的神经元之间**长出新的边**（这是"自己产生连接"的前提）
    · 新边可以跨位置、跨层、跨区，不受出生连线"只连更靠近自己的"这条限制

默认**关闭**（当前阶段要求先天全固定、可塑 0 条）；打开后才生效。

用法（在引擎里开启，一行）：
    from plastic import Plastic
    pl = Plastic(); pl.enabled = True
    ...每次 B.step(external=...) 之后：
    pl.tick(B.act)      # 记录这一拍的活动：可塑边增长 / 长新边 / 周期降权
    pl.inject(B)        # 把新长出的边并进引擎，并同步已有可塑边的权重
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlasticCfg:
    enabled: bool = False        # 默认关闭
    eta: float = 0.02            # 学习步长
    w_ref: float = 1.0           # 软饱和尺度
    pow: float = 1.0             # 越涨越难涨的指数
    T: int = 200                 # 每隔多少拍整体降权一次
    gamma: float = 0.99          # 每次降权的倍率
    act_thr: float = 0.35        # 判定"活跃"的门槛
    new_per_tick: int = 100      # 每拍最多尝试新增多少对
    max_plastic: int = 200_000   # 可塑边总数上限（纯内存考虑）
    w_min: float = 1e-3          # 权重低于它直接删（簿记）
    seed: int = 1


class Plastic:
    """可塑连接的规则实现。自带边表，不改引擎；用 inject() 并进引擎。"""

    def __init__(self, cfg: PlasticCfg | None = None):
        self.cfg = cfg or PlasticCfg()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.src = np.zeros(0, dtype=np.int64)
        self.dst = np.zeros(0, dtype=np.int64)
        self.w = np.zeros(0, dtype=np.float64)
        self.n_new_edges = 0
        self.n_decays = 0
        self.tick_count = 0
        self._prev = None
        self._idx: dict[tuple[int, int], int] = {}
        self._in_engine = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------ 开关
    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    @enabled.setter
    def enabled(self, v: bool) -> None:
        self.cfg.enabled = bool(v)

    def n_edges(self) -> int:
        return int(len(self.w))

    # ------------------------------------------------------------------ 每拍
    def tick(self, act) -> None:
        now = np.asarray(act, dtype=np.float64)
        cfg = self.cfg
        if not cfg.enabled:
            self._prev = now.copy()
            self.tick_count += 1
            return
        prev = self._prev if (self._prev is not None and len(self._prev) == len(now)) else now

        # 规则一：已存在的可塑边按时序赫布增长
        if len(self.w):
            s, d = self.src, self.dst
            h = (prev[s] * now[d] + now[s] * prev[d] + 0.5 * now[s] * now[d])
            self.w += cfg.eta * h / (1.0 + self.w / cfg.w_ref) ** cfg.pow

        # 规则三：在共同活跃的神经元之间长新边（含"先后"与"同时"两种）
        if cfg.new_per_tick > 0 and len(self.w) < cfg.max_plastic:
            self._grow_new(now, prev)

        # 规则二：周期整体降权（只动可改变的边）
        self.tick_count += 1
        if cfg.T > 0 and self.tick_count % cfg.T == 0 and len(self.w):
            self.w *= cfg.gamma
            self.n_decays += 1
            keep = self.w >= cfg.w_min
            if not keep.all():
                self.src, self.dst, self.w = self.src[keep], self.dst[keep], self.w[keep]
                self._idx = {k: i for i, k in enumerate(zip(self.src.tolist(),
                                                            self.dst.tolist()))}
                self._in_engine = np.full(len(self.w), -1, dtype=np.int64)
        self._prev = now.copy()

    # ------------------------------------------------------------------ 长新边
    def _grow_new(self, now: np.ndarray, prev: np.ndarray) -> None:
        cfg = self.cfg
        prev_on = np.nonzero(prev > cfg.act_thr)[0]
        now_on = np.nonzero(now > cfg.act_thr)[0]
        if len(now_on) == 0:
            return
        n = max(1, cfg.new_per_tick // 2)
        # (a) 先亮的 → 刚亮的
        if len(prev_on):
            for _ in range(n):
                i = int(self.rng.choice(prev_on))
                j = int(self.rng.choice(now_on))
                if i != j:
                    self._strengthen(i, j, cfg.eta * prev[i] * now[j])
        # (b) 同一步一起亮 → 互相加强（两个方向都长）
        if len(now_on) >= 2:
            for _ in range(n):
                i, j = (int(x) for x in self.rng.choice(now_on, size=2, replace=False))
                self._strengthen(i, j, cfg.eta * now[i] * now[j])
                self._strengthen(j, i, cfg.eta * now[j] * now[i])

    def _strengthen(self, i: int, j: int, dw: float) -> None:
        """已有这条边就加强它；没有就长一条新的（这是"自己产生连接"）。"""
        if dw <= 1e-9:
            return
        k = self._idx.get((i, j))
        if k is not None:
            w = self.w[k]
            self.w[k] = w + dw / (1.0 + w / self.cfg.w_ref) ** self.cfg.pow
            return
        self._add(i, j, dw)

    def _add(self, i: int, j: int, w0: float) -> None:
        self._idx[(i, j)] = len(self.w)
        self.src = np.append(self.src, i)
        self.dst = np.append(self.dst, j)
        self.w = np.append(self.w, max(w0, 1e-6))
        self._in_engine = np.append(self._in_engine, -1)
        self.n_new_edges += 1

    # ------------------------------------------------------------------ 并进引擎
    def inject(self, B) -> int:
        """把还没并进引擎的可塑边追加进去（兴奋性），并同步已有可塑边的权重。"""
        if not self.cfg.enabled:
            return 0
        if len(self._in_engine) != len(self.w):
            self._in_engine = np.full(len(self.w), -1, dtype=np.int64)
        todo = np.nonzero(self._in_engine < 0)[0]
        added = 0
        if len(todo):
            base = len(B.src)
            B.src = np.r_[B.src, self.src[todo].astype(np.int32)]
            B.dst = np.r_[B.dst, self.dst[todo].astype(np.int32)]
            B.w = np.r_[B.w, self.w[todo].astype(np.float32)]
            B.sgn = np.r_[B.sgn, np.ones(len(todo), dtype=np.float32)]
            B.group = np.r_[B.group, np.full(len(todo), -1, dtype=np.int16)]
            self._in_engine[todo] = np.arange(base, base + len(todo))
            added = len(todo)
        m = self._in_engine >= 0
        if m.any():
            B.w[self._in_engine[m]] = self.w[m].astype(np.float32)
        return added
