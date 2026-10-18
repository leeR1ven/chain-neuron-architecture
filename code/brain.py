# -*- coding: utf-8 -*-
"""链式架构引擎（重写版）。

从这一路的教训直接写进来的几条硬规则：

  1. **权重随机、每条边各不相同、不做归一化**。
     归一化会让"此消彼长的两个数加起来变成常数"，画面差别就被加没了。
  2. **就近只看图像平面（y、z），绝不让层与层之间的固定 x 差参与距离**。
     之前用三维距离，x 的差（0.05~0.27）远大于平面间距（~0.004），
     结果每个神经元看到的都是上一层最靠边的那一排，图像的"行"直接丢了。
  3. **门槛/带宽从连线本身算出来**（该神经元入边权重之和的一个比例），
     不用任何"整层统计量"去标定 —— 那种做法不是饱和（整层全亮）
     就是沉底（整片 0），两者都试过。
  4. **抑制是独立的抑制细胞**，不在连接上写负权重；
     抑制细胞的输出对目标取负。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    tau: float = 0.03            # 活动时间常数
    dt: float = 0.01
    gain_lr: float = 0.002       # 整体阈值调控的速度
    gain_range: tuple = (-0.05, 0.05)   # 只做微调，不许主导（实测 ±0.5 时它会顶到上限、把深层压死）
    target_active: int = 0       # 0 = 自动（按网络规模定）
    seed: int = 1


class Brain:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.pos: list[np.ndarray] = []
        self.region: list[str] = []
        self.thr: list[float] = []
        self.soft: list[float] = []
        self.is_inh: list[bool] = []
        self.counted: list[bool] = []
        self.act = np.zeros(0, dtype=np.float32)
        self.gain = 0.0
        self.n = 0
        # 连接
        self.src = np.zeros(0, dtype=np.int32)
        self.dst = np.zeros(0, dtype=np.int32)
        self.w = np.zeros(0, dtype=np.float32)
        self.sgn = np.zeros(0, dtype=np.float32)      # 抑制来源 = −1
        self.group = np.zeros(0, dtype=np.int16)
        self._cache_n = -1

    # ---------------------------------------------------------------- 建
    def add(self, region: str, positions, thr: float = 0.30, soft: float = 0.30,
            inh: bool = False, counted: bool = True) -> np.ndarray:
        p = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        start = self.n
        self.pos.append(p)
        self.region += [region] * len(p)
        self.thr += [float(thr)] * len(p)
        self.soft += [float(soft)] * len(p)
        self.is_inh += [bool(inh)] * len(p)
        self.counted += [bool(counted)] * len(p)
        self.n += len(p)
        self.act = np.zeros(self.n, dtype=np.float32)
        return np.arange(start, start + len(p))

    def xyz(self, ids) -> np.ndarray:
        if self._cache_n != self.n:
            self._allp = np.vstack(self.pos)
            self._cache_n = self.n
        return self._allp[np.asarray(ids, dtype=int)]

    # ---------------------------------------------------------------- 连
    def connect(self, targets, sources, k: int, w_range=(0.05, 0.20),
                axes=(1, 2), bias: float = 3.0, group: int = 0,
                thr_frac: float | None = None, soft_frac: float | None = None):
        """出生连线：**随机连接更靠近自己的神经元**。

        targets 的每个神经元，从 sources 里按 axes 指定的平面距离取最近的
        k×bias 个作为候选，再随机挑 k 个；权重各自随机。
        axes=(1,2) 表示只看图像平面（不看层与层之间的 x 差）。
        thr_frac/soft_frac: 门槛与带宽 = 该神经元入边权重之和 × 这个比例
                            （从连线算出来，不用统计标定）。
        """
        targets = np.asarray(targets, dtype=int)
        sources = np.asarray(sources, dtype=int)
        if len(targets) == 0 or len(sources) == 0:
            return
        sp = self.xyz(sources)[:, list(axes)]
        tp = self.xyz(targets)[:, list(axes)]
        pool = min(len(sources), max(k, int(k * bias)))
        S, D, W = [], [], []
        for t, q in zip(targets, tp):
            d = np.linalg.norm(sp - q, axis=1)
            cand = np.argpartition(d, min(pool, len(d) - 1))[:pool]
            sel = self.rng.choice(cand, size=min(k, len(cand)), replace=False)
            for si in sel:
                S.append(int(sources[si]))
                D.append(int(t))
                W.append(float(self.rng.uniform(*w_range)))
        S = np.asarray(S, dtype=np.int32)
        D = np.asarray(D, dtype=np.int32)
        W = np.asarray(W, dtype=np.float32)
        sgn = np.where([self.is_inh[i] for i in S], -1.0, 1.0).astype(np.float32)
        self.src = np.r_[self.src, S]
        self.dst = np.r_[self.dst, D]
        self.w = np.r_[self.w, W]
        self.sgn = np.r_[self.sgn, sgn]
        self.group = np.r_[self.group, np.full(len(S), group, dtype=np.int16)]
        if thr_frac is not None or soft_frac is not None:
            # 用**全部入边**的权重和（含之前几次连线加的边），
            # 否则被多次连线的神经元门槛会被后一次覆盖，和真实权重和对不上
            tot = np.bincount(self.dst, weights=np.abs(self.w), minlength=self.n)
            for t in np.unique(D):
                s = float(tot[int(t)])
                if thr_frac is not None:
                    self.thr[int(t)] = thr_frac * s
                if soft_frac is not None:
                    self.soft[int(t)] = max(1e-4, soft_frac * s)

    def chain(self, relays, w_range=(0.35, 0.45), group: int = 0,
              thr_frac: float = 0.55, soft_frac: float = 0.45):
        """一条链：顺次相接（每隔一小段一个神经元，前一个 → 后一个）。"""
        relays = np.asarray(relays, dtype=int)
        for i in range(len(relays) - 1):
            self.connect(relays[i + 1:i + 2], relays[i:i + 1], k=1,
                         w_range=w_range, axes=(1, 2), group=group,
                         thr_frac=thr_frac, soft_frac=soft_frac)

    # ---------------------------------------------------------------- 走
    def step(self, external=None):
        c = self.cfg
        drive = np.bincount(self.dst, weights=self.sgn * self.w * self.act[self.src],
                            minlength=self.n).astype(np.float32)
        if external is not None:
            drive = drive + external
        # 门槛下限只留数值保底：增益是相对量，门槛不会被压成负数，
        # 0.02 这种固定下限会让下降链每过一级固定损失一点（力度低端被压缩）
        eff = np.maximum(np.asarray(self.thr, dtype=np.float32) * (1.0 + self.gain), 1e-4)
        want = np.clip((drive - eff) / np.asarray(self.soft, dtype=np.float32), 0.0, 1.0)
        self.act = self.act + (c.dt / c.tau) * (want - self.act)
        np.clip(self.act, 0.0, 1.0, out=self.act)
        if c.target_active:
            counted = np.asarray(self.counted)
            n_on = int(((self.act > 0.35) & counted).sum())
            n_sat = int(((self.act > 0.98) & counted).sum())
            self.gain += c.gain_lr * (n_on / c.target_active - 1.0)
            # 抗饱和：饱和细胞超过目标的一成就抬高门槛，否则饱和的级传不出变化
            self.gain += c.gain_lr * 2.0 * (n_sat / max(1, c.target_active) - 0.1)
            self.gain = float(np.clip(self.gain, *c.gain_range))

    # ---------------------------------------------------------------- 看
    def ids_of(self, prefix: str) -> np.ndarray:
        return np.array([i for i in range(self.n) if self.region[i].startswith(prefix)],
                        dtype=int)

    def n_active(self, prefix: str, t: float = 0.35) -> int:
        ids = self.ids_of(prefix)
        return int((self.act[ids] > t).sum()) if len(ids) else 0
