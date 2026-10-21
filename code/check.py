# -*- coding: utf-8 -*-
"""架构自检：只查"架构上一定不成立"的事，不看效果。

用法：
    python check.py            # 默认 5 层目标规模
    python check.py --small    # 小规模，快

整合前建议跑一次；任何 issue 都应当先修掉，再接其他部分。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Config  # noqa: E402
from build import GROUP_PFC_FWD, REGIONS, Geometry, Interface, Spec, build  # noqa: E402


def _is_ids(a) -> bool:
    return isinstance(a, np.ndarray) and a.dtype.kind in "iu"


def _arr_ids(ids: dict):
    """展开 ids 里所有神经元编号数组（含 layers/hops/pfc_groups 子字典）。"""
    for k, v in ids.items():
        if _is_ids(v):
            yield k, v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if _is_ids(vv):
                    yield f"{k}{kk}", vv
                elif isinstance(vv, (list, tuple)):
                    for i, x in enumerate(vv):
                        if _is_ids(x):
                            yield f"{k}{kk}[{i}]", x


def check_architecture(B, ids: dict, sp: Spec, geo: Geometry) -> dict:
    issues = []
    stats: dict = {}
    n = B.n
    x = B.xyz(np.arange(n))[:, 0] if n else np.zeros(0)

    # 1) 编号/边的基本一致性
    if len(B.region) != n or len(B.thr) != n or len(B.soft) != n:
        issues.append("神经元属性数组长度不一致")
    if len(B.src) != len(B.dst) or len(B.src) != len(B.w) or len(B.src) != len(B.sgn):
        issues.append("连接数组长度不一致")
    if len(B.src) and (B.src.min() < 0 or B.src.max() >= n
                       or B.dst.min() < 0 or B.dst.max() >= n):
        issues.append("存在越界连接")
    if len(B.w) and np.any(B.w <= 0):
        issues.append("存在非正权重")
    if len(B.sgn) and not np.all(np.isin(B.sgn, (-1.0, 1.0))):
        issues.append("sgn 只能是 ±1")
    if len(B.alive) != n:
        issues.append("alive 长度不对")

    for name, arr in _arr_ids(ids):
        arr = np.asarray(arr).ravel()
        if arr.size == 0:
            continue
        if arr.min() < 0 or arr.max() >= n:
            issues.append(f"ids[{name}] 越界")
        # 允许含已删除编号：结构可塑性用稳定 ID（和立体网络一样只改 alive，不移位）；
        # 各消费方（相似度/结构）都会先按 alive 过滤。

    if issues:
        return {"ok": False, "issues": issues, "stats": stats}

    indeg = np.bincount(B.dst, minlength=n)
    alive = B.alive

    # 2) 输入细胞不应有入边
    inp = np.r_[ids["pos"]["vision"].ravel(), ids["neg"]["vision"].ravel(),
                ids["pos"]["audio"].ravel(), ids["neg"]["audio"].ravel(),
                ids["pos"]["touch"].ravel(), ids["neg"]["touch"].ravel(),
                ids["pos"]["balance"].ravel(), ids["neg"]["balance"].ravel()]
    bad = int((indeg[inp] > 0).sum())
    if bad:
        issues.append(f"{bad} 个输入细胞有入边（应由外界直接驱动）")

    # 3) 每区每层：活着、有输入、有中继、有前额叶组、组里确实收到这一层的 group30 边
    last = max(d for (r, d) in ids["layers"] if r == REGIONS[0])
    x_pfc = float(B.xyz(ids["pfc_groups"][(REGIONS[0], last)])[:, 0].mean())
    for r in REGIONS:
        for d in range(1, last + 1):
            key = (r, d)
            if key not in ids["layers"]:
                issues.append(f"缺层 {key}")
                continue
            lay = ids["layers"][key][alive[ids["layers"][key]]]
            if len(lay) == 0:
                issues.append(f"{key} 没有存活神经元")
                continue
            dead_in = int((indeg[lay] == 0).sum())
            if dead_in:
                issues.append(f"{key} 有 {dead_in} 个存活神经元没有入边")
            if key not in ids["hops"] or len(ids["hops"][key]) < 2:
                issues.append(f"{key} 缺两级中继")
            if key not in ids["pfc_groups"]:
                issues.append(f"{key} 缺前额叶组")
                continue
            g = ids["pfc_groups"][key][alive[ids["pfc_groups"][key]]]
            if len(g) == 0:
                issues.append(f"{key} 前额叶组没有存活神经元")
                continue
            m = np.isin(B.dst, g) & (B.group == GROUP_PFC_FWD)
            if int(m.sum()) == 0:
                issues.append(f"{key} 的前额叶组没有收到 group30 的输入")

    # 4) 前向边方向：group30 的源 x 不能大于目标 x（沿前额叶方向走）
    m = B.group == GROUP_PFC_FWD
    if int(m.sum()):
        back = int((x[B.src[m]] > x[B.dst[m]] + 1e-6).sum())
        if back:
            issues.append(f"{back} 条前向边的方向反了（源比目标更靠前额叶）")
        stats["forward_edges"] = int(m.sum())

    # 5) 回程：边数 ≤ 前向一半；不落回第 1 层；只落在原来那一层的层内
    RET_GROUPS = (16, 17, 18, 19)
    ret_mask = np.isin(B.group, RET_GROUPS)
    stats["return_edges"] = int(ret_mask.sum())
    if int(m.sum()) and ret_mask.sum() > 0.5 * int(m.sum()):
        issues.append(f"回程边 {int(ret_mask.sum())} > 前向一半 {int(m.sum()) // 2}")
    for r in REGIONS:
        l1 = ids["layers"][(r, 1)]
        rets = np.r_[B.ids_of(f"ret_{r}_L1_0"), B.ids_of(f"ret_{r}_L1_1")]
        if len(rets) and int((np.isin(B.dst, l1) & np.isin(B.src, rets)).sum()):
            issues.append(f"{r} 的回程落回了第 1 层")

    # 6) 运动区/门控：运动组在视觉最后一层前额叶组里；力度链有输入；门控符号是抑制
    vg = ids["pfc_groups"][("vision", last)]
    motor = np.concatenate(ids["motor_pfc"])
    if not np.isin(motor, vg).all():
        issues.append("运动相关组不在视觉最后一层的前额叶组里")
    for mm in range(sp.n_muscle):
        f_ids = ids["force"][mm]
        src_ok = np.isin(B.src, ids["down"][mm][-1:]) & np.isin(B.dst, f_ids)
        if not src_ok.any():
            issues.append(f"肌肉 {mm} 的力度神经元没有收到下降链末端")
    # gate(抑制) → down；dis(抑制) → gate；符号都应为 -1
    g4 = B.group == 6
    if int(g4.sum()) and not np.all(B.sgn[g4] < 0):
        issues.append("门控 I → 下降链 的边不是抑制")
    g5 = B.group == 4
    if int(g5.sum()) and not np.all(B.sgn[g5] < 0):
        issues.append("去抑制 D → 门控 I 的边不是抑制")

    # 7) 温度计编码：每通道每对恰好一个
    ui = Interface(B, ids, sp)
    for v in (0.0, 0.37, 1.0):
        ui.set_values(v)
        ext = np.asarray(ui.drive(), dtype=np.float32)
        for r in REGIONS:
            pos = ids["pos"][r]
            neg = ids["neg"][r]
            on = ext[pos].reshape(len(pos), sp.n_level) + ext[neg].reshape(len(neg), sp.n_level)
            if not np.allclose(on, 1.0):
                issues.append(f"{r} 在 v={v} 时温度计不是每对恰好一个")
                break

    stats.update({
        "neurons": int(n), "alive": int(alive.sum()), "edges": int(len(B.src)),
        "layers": last, "pfc_x": round(x_pfc, 3),
        "regions": list(REGIONS),
    })
    return {"ok": not issues, "issues": issues, "stats": stats}


def main(argv=None) -> int:
    small = "--small" in (argv if argv is not None else sys.argv)
    if small:
        sp = Spec(px=3, n_level=4, n_audio=4, n_touch=6, n_balance=3,
                  n_muscle=2, n_force=4, k_near=6,
                  layer_ratio=(2.0, 3.0), layer_floor=(16, 24))
    else:
        sp = Spec()
    B, ids, sp, geo = build(sp, Config(), verbose=False)
    rep = check_architecture(B, ids, sp, geo)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
