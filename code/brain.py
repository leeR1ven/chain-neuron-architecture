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
        self.alive = np.zeros(0, dtype=bool)   # 结构可塑性：被删掉的神经元保留编号，但不再参与
        self.gain = 0.0
        self.n = 0
        self._allp = np.zeros((0, 3), dtype=np.float32)
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
        # 追加而不是清零：结构可塑性会在运行中加神经元，已有活动不能被抹掉
        act = np.zeros(self.n, dtype=np.float32)
        act[:start] = self.act
        self.act = act
        self.alive = np.r_[self.alive, np.ones(len(p), dtype=bool)]
        self._cache_n = -1
        return np.arange(start, start + len(p))

    def remove(self, ids) -> int:
        """标记删除（不改编号，保证 ids / 位置 / 统计仍然对得上）。

        删除的神经元：活动清零、门槛与带宽置 0（永远不会再发），
        与它相连的边整条去掉（不是权重置 0）——这样 step 的边数才是真实边数。
        """
        ids = np.asarray(ids, dtype=int).ravel()
        ids = ids[self.alive[ids]]
        if len(ids) == 0:
            return 0
        self.alive[ids] = False
        self.act[ids] = 0.0
        for i in ids:
            self.thr[int(i)] = 0.0
            self.soft[int(i)] = 1.0
        drop = np.isin(self.src, ids) | np.isin(self.dst, ids)
        if drop.any():
            keep = ~drop
            self.src = self.src[keep]
            self.dst = self.dst[keep]
            self.w = self.w[keep]
            self.sgn = self.sgn[keep]
            self.group = self.group[keep]
        return int(len(ids))

    def xyz(self, ids) -> np.ndarray:
        if self._cache_n != self.n:
            self._allp = np.vstack(self.pos)
            self._cache_n = self.n
        return self._allp[np.asarray(ids, dtype=int)]

    # ---------------------------------------------------------------- 连
    def connect(self, targets, sources, k: int, w_range=(0.05, 0.20),
                axes=(1, 2), bias: float = 3.0, group: int = 0,
                thr_frac: float | None = None, soft_frac: float | None = None,
                nearest: bool = True, exact: bool = False):
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
            if nearest:
                d = np.linalg.norm(sp - q, axis=1)
                if exact:
                    # 确定性：只取最近的 k 个，不在 3k 候选里随机挑
                    # （随机挑会掺进相对远、关系不强的输入，等于把无关信号混进来）
                    sel = np.argsort(d)[:k]
                else:
                    cand = np.argpartition(d, min(pool, len(d) - 1))[:pool]
                    sel = self.rng.choice(cand, size=min(k, len(cand)), replace=False)
            else:
                # 对照实验用：不管位置，从整层里随机挑（模拟"无关信号被同一个链接受"）
                cand = self.rng.choice(len(sources), size=pool, replace=False)
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

    def link(self, sources, targets, w_range=(1.0, 1.0), group: int = 0) -> np.ndarray:
        """一对一专用线：第 i 个 source 只连第 i 个 target（长度必须相同）。

        "一个链式神经元只接一个上游神经元"就用它；配合 thr=0、soft=w，
        中继变成一根原样传信号的线（不混合、不丢细节）。
        """
        S = np.asarray(sources, dtype=np.int32).ravel()
        D = np.asarray(targets, dtype=np.int32).ravel()
        if len(S) != len(D):
            raise ValueError(f"link 要求一一对应：{len(S)} 个来源 vs {len(D)} 个目标")
        if len(S) == 0:
            return np.zeros(0, dtype=np.float32)
        W = self.rng.uniform(*w_range, size=len(S)).astype(np.float32)
        sgn = np.where([self.is_inh[int(i)] for i in S], -1.0, 1.0).astype(np.float32)
        self.src = np.r_[self.src, S]
        self.dst = np.r_[self.dst, D]
        self.w = np.r_[self.w, W]
        self.sgn = np.r_[self.sgn, sgn]
        self.group = np.r_[self.group, np.full(len(S), group, dtype=np.int16)]
        return W

    def add_edges(self, sources, targets, weights, group: int = 0) -> None:
        """显式加一批边（来源/目标/权重一一对应）；结构可塑性复制父神经元输入时用它。"""
        S = np.asarray(sources, dtype=np.int32).ravel()
        D = np.asarray(targets, dtype=np.int32).ravel()
        W = np.asarray(weights, dtype=np.float32).ravel()
        if not (len(S) == len(D) == len(W)):
            raise ValueError(f"add_edges 长度不一致：{len(S)}/{len(D)}/{len(W)}")
        if len(S) == 0:
            return
        sgn = np.where([self.is_inh[int(i)] for i in S], -1.0, 1.0).astype(np.float32)
        self.src = np.r_[self.src, S]
        self.dst = np.r_[self.dst, D]
        self.w = np.r_[self.w, W]
        self.sgn = np.r_[self.sgn, sgn]
        self.group = np.r_[self.group, np.full(len(S), group, dtype=np.int16)]

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
        """按区名前缀取编号；已删除的神经元默认不返回（避免统计把死细胞算进去）。"""
        return np.array([i for i in range(self.n)
                         if self.alive[i] and self.region[i].startswith(prefix)],
                        dtype=int)

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def n_active(self, prefix: str, t: float = 0.35) -> int:
        ids = self.ids_of(prefix)
        return int((self.act[ids] > t).sum()) if len(ids) else 0
