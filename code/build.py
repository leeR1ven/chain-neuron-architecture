# -*- coding: utf-8 -*-
"""按规格把链式架构搭出来（重写版）。

【规格 —— 你在对话里说过的】
 1. 输入：每个像素有具体位置；每像素分 R/G/B 三个基础通道；
    每通道温度计编码 N 档 × 正反一对 → **每对恰好一个被激活**（输入层永远恰好一半活跃）
 2. 区与区远 → 每隔一小段空间放一个神经元，收附近很多神经元的信号，朝前额叶方向传
 3. **越靠近输入层数量越少，越靠近前额叶越多**（输入层特征少；皮层信号分得清；太少就区分不了）
 4. 前额叶 = 各条链的汇聚端，**内部先不互连**
 5. 运动区 = 通往肌肉的链 + 力度编码神经元（每肌肉 N 个，亮的越多力越大）；不要正反一对
 6. 运动区有抑制神经元压住行动；抑制两条：前额叶并行的"别动"链 + 感觉区直接的反射抑制链
 7. 抑制与去抑制都走**抑制神经元 / 去抑制神经元**（不直接互抑）
 8. 不是运动相关神经元一亮就发运动信号：要看它是不是**主要被激活的**那批；
    只有**大量运动相关神经元同时被激活**才发得出去
 9. 出生连线**只能随机连接更靠近自己的**；**现在全部是不可改变的连接**
10. 听觉/触觉/平衡同样编码、同样汇入前额叶；规模可缩（像素 10×10，其他按比例）
11. 不用海马体

【实现要点（这一路踩坑换来的）】
 · 就近只看图像平面 (y,z)，不看层间的 x 差
 · 权重随机、不归一；门槛/带宽 = 该神经元入边权重和 × 比例（从连线算，不做统计标定）
 · 第 1 层分 ON / OFF 两条：ON 读"正"（亮就发），OFF 读"反"（暗就发）→ 避免此消彼长被相加抵掉
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from brain import Brain, Config


@dataclass
class Spec:
    px: int = 10                 # 像素网格
    n_rgb: int = 3               # 每个像素三个最基础的 RGB 通道
    n_level: int = 10            # 每通道温度计档数（10 对正反）
    n_audio: int = 20            # 听觉频带数
    n_touch: int = 40            # 触觉触点数
    n_balance: int = 6           # 平衡量个数
    n_muscle: int = 8            # 肌肉数
    n_force: int = 10            # 每块肌肉的力度档数
    # 中间各层相对"本感觉区通道数"的倍数（越靠近前额叶越多）
    layer_ratio: tuple = (2.0, 4.0, 6.0)   # 各层 = 通道数 × 这个倍数（单调递增）
    k_near: int = 16             # 每个神经元接上一层几个（实测 16 最好；越大局部图案越被淹没）
    layer_floor: tuple = (128, 192, 256)   # 各层神经元数下限（小模型可调小）
    pfc_ratio: float = 1.0       # 每一层各送多少比例的神经元到前额叶（加大以便承载细微差别）
    relay_gap: int = 0            # 链上每隔几个位置放一个（0=相邻）
    chain_mode: str = "one2one"  # "one2one"=一条专用线只接一个神经元（默认）；"knn"=就近多输入；"random"=对照
    layer_mode: str = "knn"      # 区内 L2–L4 的接线："knn" / "random"（后者只用于对照）
    pick_mode: str = "random"    # 就近候选怎么挑："random"=旧的 3k 里随机；"exact"=只取最近 k 个


# 按深度定门槛比例：每深一层活跃密度约减半，门槛比例必须跟着减半，否则下一级凑不满门槛
DEPTH_FRAC = {1: (0.25, 0.45), 2: (0.25, 0.45), 3: (0.08, 0.30), 4: (0.04, 0.25)}

# 出生连线最后统一按深度覆盖一次（与 build 末尾的 _DEPTH 一致；结构可塑性新增神经元也用它）
DEPTH_GAIN = {1: (0.30, 0.90), 2: (0.26, 0.45), 3: (0.22, 0.32), 4: (0.22, 0.28)}

# 边分组：所有"通往中继、进入前额叶"的边都打这个组号。
# structure.py 靠它找出"往前额叶送"的通路，做"常亮不带特征"的筛选。
GROUP_PFC_FWD = 30


@dataclass
class Geometry:
    """各感觉区在图像平面 (y,z) 上占的条带，以及沿 x 的层深。"""
    bands: dict = field(default_factory=lambda: {
        "vision": (0.00, 0.40), "audio": (0.42, 0.58),
        "touch": (0.60, 0.78), "balance": (0.80, 0.95)})
    x_in: float = 0.00
    x_pfc: float = 0.60
    x_motor: float = 0.85


def build(sp: Spec | None = None, cfg: Config | None = None,
          geo: Geometry | None = None, verbose: bool = True):
    sp, cfg, geo = sp or Spec(), cfg or Config(), geo or Geometry()
    B = Brain(cfg)
    L = sp.n_level
    ids: dict = {}
    n_motor = sp.n_muscle * sp.n_force

    # ================================================================ 输入层
    # 每个"通道"= 一个像素的一个颜色（或一个频带 / 一个触点 / 一个平衡量）
    # 一个通道占 L 对 = 2L 个细胞；每对恰好亮一个。
    modalities = {
        "vision": sp.px * sp.px * sp.n_rgb,
        "audio": sp.n_audio, "touch": sp.n_touch, "balance": sp.n_balance,
    }
    chan_pos: dict = {}          # 每个通道在平面上的位置
    pos_ids: dict = {}
    neg_ids: dict = {}
    for name, n_chan in modalities.items():
        y0, y1 = geo.bands[name]
        ys = np.linspace(y0, y1, n_chan)
        zz = np.linspace(0.02, 0.98, L)
        P = np.array([[geo.x_in, y, z] for y in ys for z in zz], dtype=np.float32)
        pos = B.add(f"{name}_pos", P, thr=0.10, soft=0.25, counted=False)
        neg = B.add(f"{name}_neg", P + np.array([0.0, 0.0015, 0.0], np.float32),
                    thr=0.10, soft=0.25, counted=False)
        pos_ids[name] = pos.reshape(n_chan, L)
        neg_ids[name] = neg.reshape(n_chan, L)
        chan_pos[name] = np.stack([np.full(n_chan, geo.x_in), ys,
                                   np.full(n_chan, 0.5)], axis=1).astype(np.float32)
    ids["pos"], ids["neg"] = pos_ids, neg_ids

    # ================================================================ 各感觉区逐层
    # 第 1 层：ON / OFF 两个神经元（每个通道一对），位置就在该通道处
    # 之后每层数量按 layer_ratio 增长，感受野是上一层里 (y,z) 最近的一小圈
    pfc_parts, mod_info = [], {}
    for name, n_chan in modalities.items():
        y0, y1 = geo.bands[name]
        ys = chan_pos[name][:, 1]
        layer1_pos = np.stack([np.full(n_chan, geo.x_in + 0.10), ys,
                               np.full(n_chan, 0.5)], axis=1).astype(np.float32)
        on = B.add(f"{name}_L1on", layer1_pos, thr=0.25, soft=0.75)
        off = B.add(f"{name}_L1off", layer1_pos + np.array([0.0, 0.0015, 0.0], np.float32),
                    thr=0.25, soft=0.75)
        for ci in range(n_chan):
            # ON：读该通道的"正"（有几档过线就发多强）—— 线性读出，不设高门槛
            B.connect(on[ci:ci + 1], pos_ids[name][ci], k=L, w_range=(0.30, 0.45),
                      axes=(1, 2), thr_frac=0.0, soft_frac=1.0)
            # OFF：读同一通道的"反"（暗就发）
            B.connect(off[ci:ci + 1], neg_ids[name][ci], k=L, w_range=(0.30, 0.45),
                      axes=(1, 2), thr_frac=0.0, soft_frac=1.0)
        prev = np.r_[on, off]
        layers = [prev]
        ids[f"{name}_L1on"], ids[f"{name}_L1off"] = on, off
        ids.setdefault("layers", {})[(name, 1)] = np.r_[on, off]
        xs = np.linspace(geo.x_in + 0.10, geo.x_pfc - 0.04, len(sp.layer_ratio) + 1)
        for li, ratio in enumerate(sp.layer_ratio):
            # 小感觉区（平衡只有 6 个通道）层内也要有下限，否则整区太小、送上去的图案分不开
            _floor = tuple(sp.layer_floor)[min(li, len(sp.layer_floor) - 1)]
            cnt = max(_floor, int(round(n_chan * ratio)))
            yy = np.linspace(y0, y1, cnt)
            zz = np.linspace(0.10, 0.90, cnt)
            p = np.stack([np.full(cnt, xs[li + 1]), yy, zz], axis=1).astype(np.float32)
            cur = B.add(f"{name}_L{li + 2}", p, thr=0.30, soft=0.40)
            ids.setdefault("layers", {})[(name, li + 2)] = cur
            if li == 0:
                # 只有这一跳分两条路：prev 的前半是"正"、后半是"反"
                half = len(prev) // 2
                # 另外直接读输入层的温度计细胞（每个通道 10 档那些），
                # 让"某个通道动一档"在第2层就占足够大的比重
                raw = np.r_[pos_ids[name].ravel(), neg_ids[name].ravel()]
                B.connect(cur, raw, k=16, w_range=(0.15, 0.25), axes=(1, 2),
                          thr_frac=0.20, soft_frac=0.60)
                kk = max(8, min(sp.k_near, half // 2))
                f, gg = DEPTH_FRAC.get(2, (0.25, 0.45))
                B.connect(cur[0::2], prev[:half], k=kk, w_range=(0.06, 0.16),
                          axes=(1, 2), thr_frac=f, soft_frac=gg,
                          nearest=(sp.layer_mode != "random"),
                          exact=(sp.pick_mode == "exact"))
                B.connect(cur[1::2], prev[half:], k=kk, w_range=(0.06, 0.16),
                          axes=(1, 2), thr_frac=f, soft_frac=gg,
                          nearest=(sp.layer_mode != "random"),
                          exact=(sp.pick_mode == "exact"))
            else:
                # 更深的层**不再分路**：按索引对半切等于把这一层的空间范围砍一半，
                # 上层每个神经元只能看到下一层一半的空间 → 活跃密度逐级减半 → 饿死
                kk = max(3, min(sp.k_near, len(prev) // 8))
                f, gg = DEPTH_FRAC.get(li + 2, (0.25, 0.45))
                B.connect(cur, prev, k=kk, w_range=(0.06, 0.16), axes=(1, 2),
                          thr_frac=f, soft_frac=gg,
                          nearest=(sp.layer_mode != "random"),
                          exact=(sp.pick_mode == "exact"))
            prev = cur
            layers.append(cur)
            ids[f"{name}_L{li + 2}"] = cur
        mod_info[name] = layers
        pfc_parts.append(prev)

    # ================================================================ 前额叶
    # **每一层都往上送**：对每个区的第 1…4 层各铺一组前额叶神经元，
    # 每组只接"那一层"里 (y,z) 就近的神经元（保持局部）；越深的层送得越多。
    pfc_by_area: dict = {}
    for name in mod_info:
        y0, y1 = geo.bands[name]
        groups = []
        for li, lay in enumerate(mod_info[name], start=1):
            # 小感觉区也要有足够多的神经元往前额叶送，否则它送的信号在前额叶区分不出来
            cnt = max(128, int(round(len(lay) * sp.pfc_ratio * (1.0 + 0.25 * (li - 1)))))
            yy = np.linspace(y0, y1, cnt)
            zz = np.linspace(0.10, 0.90, cnt)
            p = np.stack([np.full(cnt, geo.x_pfc), yy, zz], axis=1).astype(np.float32)
            g = B.add(f"pfc_{name}_L{li}", p, thr=0.30, soft=0.35)
            ids.setdefault("pfc_groups", {})[(name, li)] = g
            # 这一层到前额叶之间，**每隔一小段放一级神经元**，一节一节往前送；
            # 越靠近前额叶这一级数量越多
            x0 = float(geo.x_in + 0.10)
            hops = []
            one = sp.chain_mode == "one2one"
            for hj, frac in enumerate((1 / 3, 2 / 3)):
                xh = x0 + (geo.x_pfc - x0) * frac
                if one:
                    # 一条专用线：每个上游神经元配一个自己的中继，只接这一个，原样传下去
                    prevs = lay if hj == 0 else hops[-1]
                    pp = B.xyz(prevs)
                    hc = len(prevs)
                    hp = np.stack([np.full(hc, xh), pp[:, 1], pp[:, 2]],
                                  axis=1).astype(np.float32)
                    # 门槛 0、带宽 = 权重 → 中继就是一根线，活动等于上游活动
                    h = B.add(f"wire_{name}_L{li}_{hj}", hp, thr=0.0, soft=1.0)
                    B.link(prevs, h, w_range=(1.0, 1.0), group=GROUP_PFC_FWD)
                else:
                    hc = max(2, int(round(cnt * (0.4 + 0.3 * hj))))
                    yh = np.linspace(y0, y1, hc)
                    zh = np.linspace(0.10, 0.90, hc)
                    hp = np.stack([np.full(hc, xh), yh, zh], axis=1).astype(np.float32)
                    h = B.add(f"hop_{name}_L{li}_{hj}", hp, thr=0.30, soft=0.35)
                    srch = lay if hj == 0 else hops[-1]
                    hf = len(srch) // 2
                    kh = max(3, min(32, hf // 8))
                    f, gg = DEPTH_FRAC.get(li, (0.25, 0.45))
                    B.connect(h[0::2], srch[:hf], k=kh, w_range=(0.06, 0.16),
                              axes=(1, 2), thr_frac=f, soft_frac=gg,
                              group=GROUP_PFC_FWD,
                              nearest=(sp.chain_mode != "random"),
                              exact=(sp.pick_mode == "exact"))
                    B.connect(h[1::2], srch[hf:], k=kh, w_range=(0.06, 0.16),
                              axes=(1, 2), thr_frac=f, soft_frac=gg,
                              group=GROUP_PFC_FWD,
                              nearest=(sp.chain_mode != "random"),
                              exact=(sp.pick_mode == "exact"))
                hops.append(h)
            ids.setdefault("hops", {})[(name, li)] = hops
            srcg = hops[-1]
            gf = len(srcg) // 2
            kg = max(3, min(sp.k_near, gf // 8))
            f, gg = DEPTH_FRAC.get(li, (0.25, 0.45))
            B.connect(g[0::2], srcg[:gf], k=kg, w_range=(0.06, 0.16),
                      axes=(1, 2), thr_frac=f, soft_frac=gg, group=GROUP_PFC_FWD,
                      exact=(sp.pick_mode == "exact"))
            B.connect(g[1::2], srcg[gf:], k=kg, w_range=(0.06, 0.16),
                      axes=(1, 2), thr_frac=f, soft_frac=gg, group=GROUP_PFC_FWD,
                      exact=(sp.pick_mode == "exact"))
            # ---- 回程：去程的原路反向（前额叶组 → 返回中继 → 回到这一层按位置最近的神经元）
            # 第 1 层是外界直接驱动的传感器，**不建回程**（落回去会锁死）。
            if li > 1:
                gsz = len(g)
                rg = B.add(f"rt_inh_{name}_L{li}", np.array(
                    [[geo.x_pfc - 0.03, (y0 + y1) / 2, 0.5]], np.float32),
                    thr=0.30, soft=0.20, inh=True)
                # 抑制细胞只收一小片前额叶（24 条随机铺开），压住第一级回程中继；
                # 权重小，只够当"门槛的一部分"，不能把回程直接压死
                B.connect(rg, g, k=min(gsz, 12), w_range=(0.010, 0.020), axes=(1, 2),
                          thr_frac=0.02, soft_frac=0.05, group=16, nearest=False)
                prevr = g
                for hj, frac in enumerate((2 / 3, 1 / 3)):
                    # 回程中继要少：它们是调制，不是第二条前向通路（边数按"≤前向一半"卡）
                    rc = max(2, min(int(round(gsz * (0.20 - 0.08 * hj))), 16))
                    yh = np.linspace(y0, y1, rc)
                    zh = np.linspace(0.10, 0.90, rc)
                    xh = geo.x_in + 0.10 + (geo.x_pfc - geo.x_in - 0.10) * frac
                    rp = np.stack([np.full(rc, xh), yh, zh], axis=1).astype(np.float32)
                    r = B.add(f"ret_{name}_L{li}_{hj}", rp, thr=0.30, soft=0.35)
                    # 第一级只看 6 条**铺开**的前额叶、第二级只看 4 条上一级：
                    # 必须"这一片里很多条同时亮"才开（随机铺开才叫"一大片"；只取最近会变成看局部小簇）
                    B.connect(r, prevr, k=(6 if hj == 0 else 4),
                              w_range=(0.05, 0.12), axes=(1, 2),
                              thr_frac=0.45, soft_frac=0.40, group=17, nearest=False)
                    if hj == 0:
                        B.connect(r, rg, k=1, w_range=(0.15, 0.25), axes=(1, 2), group=18)
                    else:
                        # 最后一级送回**原来那一层**，位置对位置（每个层神经元只收 2 条回程边）；
                        # 权重要够到"接近门槛"的神经元，才算调制（太小等于没有回程）
                        B.connect(lay, r, k=2, w_range=(0.30, 0.45), axes=(1, 2),
                                  thr_frac=0.25, soft_frac=0.45, group=19)
                    prevr = r
            groups.append(g)
            ids[f"pfc_{name}_L{li}"] = g
        pfc_by_area[name] = np.concatenate(groups)
    ids["pfc_vision"] = pfc_by_area["vision"]
    ids["pfc_other"] = np.concatenate([pfc_by_area[k] for k in ("audio", "touch", "balance")])
    pfc = np.r_[ids["pfc_vision"], ids["pfc_other"]]
    ids["pfc"] = pfc

    # 前额叶里指定"运动相关"（每块肌肉 n_force 个）与"非运动"
    # 运动相关：前额叶里**成片的一块**（取视觉最后一组的前 n_motor 个，位置是连着的）
    base = ids["pfc_vision_L4"]
    ids["motor_pfc"] = [base[m * sp.n_force:(m + 1) * sp.n_force]
                        for m in range(sp.n_muscle)]
    motor_pick = np.concatenate(ids["motor_pfc"])
    mask = np.ones(len(pfc), bool)
    mask[np.isin(pfc, motor_pick)] = False
    ids["other_pfc"] = pfc[mask]
    x_gate = (geo.x_pfc + geo.x_motor) / 2
    # ================================================================ 门控：I 与 D
    gate_pos = np.array([[x_gate, 0.06 + 0.88 * m / max(sp.n_muscle - 1, 1), 0.50]
                         for m in range(sp.n_muscle)], dtype=np.float32)
    ids["gate"] = B.add("gate", gate_pos, thr=0.30, soft=0.30, inh=True)         # I
    ids["dis"] = B.add("dis", gate_pos + np.array([0.0, 0.0, 0.06], np.float32),
                       thr=0.60, soft=0.20, inh=True)                            # D
    for m in range(sp.n_muscle):
        # I 收"与运动无关"的前额叶（一片）→ 代表"前额叶在想别的事"
        B.connect(ids["gate"][m:m + 1], ids["other_pfc"], k=min(400, len(ids["other_pfc"])),
                  w_range=(0.004, 0.008), axes=(1, 2), thr_frac=0.02, soft_frac=0.06,
                  group=2)
        # D 收该肌肉的运动相关组 → 数量够多时把 I 压住（去抑制）
        mp = ids["motor_pfc"][m]
        B.connect(ids["dis"][m:m + 1], mp, k=len(mp), w_range=(0.12, 0.16),
                  axes=(1, 2), thr_frac=0.15, soft_frac=0.25, group=3)
        B.connect(ids["gate"][m:m + 1], ids["dis"][m:m + 1], k=1, w_range=(6.0, 7.0),
                  axes=(1, 2), group=4)

    # ================================================================ 下降链 → 运动区
    ids["down"], ids["force"] = [], []
    ys_m = 0.06 + 0.88 * np.arange(sp.n_muscle) / max(sp.n_muscle - 1, 1)
    for m in range(sp.n_muscle):
        y = float(ys_m[m])
        pts = np.array([[x, y, 0.5] for x in np.linspace(geo.x_pfc + 0.04,
                                                         geo.x_motor - 0.02, 3)],
                       dtype=np.float32)
        down = B.add(f"down{m}", pts, thr=0.30, soft=0.30)
        mp = ids["motor_pfc"][m]
        # 下降链起点要**分级**（一顶开就饱和的话，力度只能是"0 或满"）
        # 线性读出：输出 ≈ 亮的运动相关神经元个数 / 总数
        B.connect(down[:1], mp, k=len(mp), w_range=(0.10, 0.16), axes=(1, 2),
                  thr_frac=0.0, soft_frac=1.0, group=5)
        # 门控（I）压在这条"往回发"的通路上
        B.connect(down[:1], ids["gate"][m:m + 1], k=1, w_range=(0.60, 0.70),
                  axes=(1, 2), group=6)
        # 带宽只按"运动相关神经元那几条边"算（别把门那条抑制边算进来，否则输出被压小）
        B.thr[down[0]] = 0.0
        B.soft[down[0]] = 0.12 * len(mp)
        # 链上顺次（每条边都够强，做成可靠的线）
        for i in range(len(down) - 1):
            # 传送带：门槛低、带宽宽，让 0.2 / 0.5 / 1.0 的输入分别对应低/中/高的输出
            # 传送带：门槛 0、带宽 = 这条边的权重 → 原样把上一级的活动传下去
            B.connect(down[i + 1:i + 2], down[i:i + 1], k=1, w_range=(0.50, 0.62),
                      axes=(1, 2), thr_frac=0.0, soft_frac=1.0, group=7)
        # 链上每一级：带宽 = 它那一条入边的权重（原样传送，不衰减）
        for _i in range(1, len(down)):
            _m = B.dst == int(down[_i])
            if _m.any():
                B.thr[int(down[_i])] = 0.0
                B.soft[int(down[_i])] = float(np.abs(B.w[_m]).sum())
        ids["down"].append(down)
        # 运动区：每块肌肉 n_force 个力度编码神经元（门槛逐档抬高）
        fp = np.array([[geo.x_motor + 0.04, y, 0.08 + 0.84 * k / max(sp.n_force - 1, 1)]
                       for k in range(sp.n_force)], dtype=np.float32)
        force = B.add(f"force{m}", fp, thr=0.10, soft=0.20)
        for k in range(sp.n_force):
            # 力度那一列：门槛铺满下降链实际给出的驱动区间（每档 = 1/档数 的驱动）
            # 按下降链实际能达到的驱动铺满（实测末端约 0.77，不是 1.0）
            B.thr[force[k]] = 0.49 * (k + 0.5) / sp.n_force
        B.connect(force, down[-1:], k=1, w_range=(0.60, 0.70), axes=(1, 2), group=8)
        ids["force"].append(force)

    # ================================================================ 运动抑制神经元 + 两条来源
    ids["inh"] = []
    for m in range(sp.n_muscle):
        y = float(ys_m[m])
        ids["inh"].append(B.add(f"inh{m}",
                                np.array([[geo.x_motor - 0.02, y, 0.50]], np.float32),
                                thr=0.30, soft=0.20, inh=True))
    # 说明：抑制神经元的**输入不在这里写死** —— 那是后期可改变的连接，
    # 谁连它现在没人知道。这里只建抑制神经元本身，以及它去压力度神经元的那条边。

    # 抑制神经元 → 该肌肉的力度神经元（抑制性：由 brain 按细胞类型取负）
    for m in range(sp.n_muscle):
        B.connect(ids["force"][m], ids["inh"][m], k=1, w_range=(0.25, 0.35),
                  axes=(1, 2), group=15)
    ids["inh_out"] = (B.group == 15)
    # 参与整体阈值调控统计的只保留**各区各层**（这条先按原样保留：
    # 把中继/前额叶也算进 counted 会让 target 变大、gain 维持负值，
    # 是现在深层/PFC 还能活着的隐性条件之一；要改必须连同 gain 目标一起重新标定）。
    BIG = re.compile(r"^(ret_|rt_inh_|down|inh|force|gate|dis)")   # 路上/前额叶也纳入统计
    for i in range(B.n):
        if BIG.match(B.region[i]):
            B.counted[i] = False
    # ---- 统一按深度定门槛（最后一手，保证生效；比例按实测驱动量级取）----
    _DEPTH = DEPTH_GAIN
    # 只统计**前向、且非抑制**的入边权重和：
    # 回程边不该把门槛顶上去；抑制边是减法的，按正数算会把门槛虚高
    _fwd = np.array([not B.region[int(x)].startswith("ret_") and not B.is_inh[int(x)]
                     for x in B.src])
    _sw = np.bincount(B.dst[_fwd], weights=B.w[_fwd], minlength=B.n)
    _deg = np.bincount(B.dst[_fwd], minlength=B.n)      # 每个神经元实际有几条前向入边
    _thr = np.asarray(B.thr, dtype=np.float64).copy()
    _soft = np.asarray(B.soft, dtype=np.float64).copy()
    for _i in range(B.n):
        _r = B.region[_i]
        _depth = None
        if _r.startswith(("hop_", "pfc_", "ret_")) or re.search(r"_L\d+$", _r):
            _m = re.search(r"_L(\d)", _r)
            if _m:
                _depth = int(_m.group(1))
        if _r.startswith("ret_"):
            # 原来是 2.0×入边权重和：act≤1 时驱动最多等于权重和，回程**永远打不开**。
            # 改成 0.5×：要接近一半的输入同时活跃才过门槛，仍是"一大片才开"的用意。
            _f, _g = 0.5, 0.6
        elif _depth in _DEPTH:
            _f, _g = _DEPTH[_depth]
            if _r.startswith("hop_"):
                _f, _g = ((0.6 * _f, 0.8 * _g) if _r.endswith("_0")
                          else (0.4 * _f, 0.6 * _g))
            elif _r.startswith("pfc_"):
                _f, _g = 0.3 * _f, 0.5 * _g
        else:
            continue
        if _sw[_i] > 1e-9:
            # 入边越少，比例越低（层小/感受野窄时凑不齐活跃输入）
            _thr[_i] = _f * _sw[_i]
            _soft[_i] = max(1e-4, _g * _sw[_i])
    B.thr = _thr.tolist()
    B.soft = _soft.tolist()
    # 整体阈值调控：目标活跃 = counted 的 10%（暂时保留原值；改这个必须重新标定 gain，
    # 否则目标一变，gain 顶到 ±0.05，第 3/4 层会整层死掉）
    cfg.target_active = max(5, int(0.10 * int(np.sum(B.counted))))

    if verbose:
        print(f"输入编码 {sum(2 * v * L for v in modalities.values())} 个"
              f"（视觉 {sp.px}×{sp.px}×{sp.n_rgb} 通道 + 听觉 {sp.n_audio}"
              f" + 触觉 {sp.n_touch} + 平衡 {sp.n_balance}，每通道 {L} 对正反）")
        for k, v in mod_info.items():
            print(f"  {k:8s} 各层: " + " → ".join(str(len(x)) for x in v))
        print(f"前额叶汇聚端 {len(pfc)}（运动相关 {n_motor}、非运动 {len(ids['other_pfc'])}）")
        print(f"总神经元 {B.n}，连接 {len(B.src):,}，可塑 0（全部不可改变）")
    return B, ids, sp, geo


class Interface:
    """中性两头：喂四种感觉的数值；读出每块肌肉的出力等级。"""

    def __init__(self, B: Brain, ids: dict, sp: Spec):
        self.B, self.ids, self.sp = B, ids, sp
        self.image = np.zeros((sp.px, sp.px, sp.n_rgb), np.float32)
        self.sig = {k: np.zeros(n, np.float32) for k, n in
                    (("audio", sp.n_audio), ("touch", sp.n_touch),
                     ("balance", sp.n_balance))}

    def set_image(self, img):
        self.image = np.clip(np.asarray(img, np.float32).reshape(
            self.sp.px, self.sp.px, self.sp.n_rgb), 0, 1)

    def set_values(self, v):
        self.image = np.full((self.sp.px, self.sp.px, self.sp.n_rgb), float(v), np.float32)

    def set_signals(self, **kw):
        for k, v in kw.items():
            if k in self.sig:
                self.sig[k] = np.clip(np.asarray(v, np.float32).ravel()[:len(self.sig[k])],
                                      0, 1)

    def _enc(self, ext, pos, neg, vals):
        lv = np.clip(np.rint(np.asarray(vals, np.float32) * self.sp.n_level), 0,
                     self.sp.n_level).astype(int)
        for j in range(self.sp.n_level):
            on = lv > j
            ext[pos[:, j][on]] = 1.0
            ext[neg[:, j][~on]] = 1.0        # 每一对恰好亮一个

    def drive(self):
        ext = np.zeros(self.B.n, dtype=np.float32)
        self._enc(ext, self.ids["pos"]["vision"], self.ids["neg"]["vision"],
                  self.image.reshape(-1))
        for k in ("audio", "touch", "balance"):
            self._enc(ext, self.ids["pos"][k], self.ids["neg"][k], self.sig[k])
        return ext

    def forces(self):
        return np.array([float((self.B.act[self.ids["force"][m]] > 0.35).sum())
                         / self.sp.n_force for m in range(self.sp.n_muscle)])
