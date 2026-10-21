# -*- coding: utf-8 -*-
"""结构可塑性自测（快，几秒）：专用线、常亮筛选、增/删神经元、回程门开。

用法：python test_structure.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Config  # noqa: E402
from build import GROUP_PFC_FWD, Spec, Interface, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402
import structure as S  # noqa: E402


def force_levels(B, ids, sp, ks=(2, 4, 6, 8, 10)):
    """点亮前额叶运动相关组 k 个 → 力度档；必须单调不降。"""
    out = []
    for k in ks:
        B.act[:] = 0.0
        ext = np.zeros(B.n, dtype=np.float32)
        ext[ids["motor_pfc"][0][:k]] = 5.0
        for _ in range(120):
            B.step(external=ext)
        out.append(int((B.act[ids["force"][0]] > 0.35).sum()))
    return out


def main() -> int:
    sp = Spec(px=3, n_level=4, n_audio=6, n_touch=6, n_balance=3,
              n_muscle=2, n_force=4, k_near=6, layer_floor=(24, 32, 48))
    B, ids, sp, geo = build(sp, Config(), verbose=False)
    ui = Interface(B, ids, sp)
    ref_calibrate(B, ids, ui, passes=1, steps=20)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra else ""))
        ok = ok and bool(cond)

    # 0) 力度分档（结构改动前，用同一台网络）
    lv = force_levels(B, ids, sp)
    check("运动力度单调不降（改动前）", all(b >= a for a, b in zip(lv, lv[1:])), lv)

    # 1) 专用线：每条线只有 1 条入边，门槛 0 / 带宽 1（原样传）
    wires = np.array([i for i in range(B.n) if B.region[i].startswith("wire_")])
    indeg = np.bincount(B.dst, minlength=B.n)
    check("专用线存在", len(wires) > 0, f"{len(wires)} 条")
    check("专用线一对一（每线 1 入）", bool(np.all(indeg[wires] == 1)))
    check("专用线原样传送",
          bool(np.all(np.abs(np.asarray(B.thr)[wires]) < 1e-9))
          and bool(np.all(np.abs(np.asarray(B.soft)[wires] - 1.0) < 1e-6)))

    # 2) 常亮/常灭不往前额叶送
    rand = S.make_random_probes(sp, n=4)
    ch = S.make_change_probes(sp, per_region=2)
    acts_all = np.vstack([S.observe(B, ui, rand, 40), S.observe(B, ui, ch, 40)])
    e0, n0 = len(B.src), B.n_alive
    st = S.prune_constant_pfc(B, acts_all)
    check("常亮筛选：有边被去掉", st["edges_removed"] > 0, st)
    check("常亮筛选：边数变少", len(B.src) < e0)
    st2 = S.prune_useless(B, ids, acts_all)
    check("减神经元：只删静默/常亮",
          st2["removed"] == st2["silent"] + st2["constant"], st2)
    check("减神经元：只减不增", B.n_alive <= n0)

    # 3) 加神经元：5% 父单元 × 2 子单元 × 2 条输入；L4 子单元配专用线
    key = ("audio", 4)
    lay0 = ids["layers"][key]
    before = int(B.alive[lay0].sum())
    new = S.grow_layer(B, ids, acts_all, sp, geo, "audio", 4, fraction=0.05,
                       children=2, k_in=2, seed=0)
    after = int(B.alive[ids["layers"][key]].sum())
    check("加神经元生效", len(new) > 0 and after >= before + len(new),
          f"{before} -> {after}（+{len(new)}）")
    din = np.bincount(B.dst, minlength=B.n)
    dout = np.bincount(B.src, minlength=B.n)
    check("新神经元各 2 条输入", bool(np.all(din[new] == 2)), din[new].tolist())
    check("新神经元接到专用线", bool(np.all(dout[new] >= 1)))
    new_wires = np.array([i for i in range(B.n) if B.region[i].startswith("wire_audio_L4_g")])
    check("新线仍是一对一", len(new_wires) == 2 * len(new) and bool(np.all(
        np.bincount(B.dst, minlength=B.n)[new_wires] == 1)))

    # 3b) 按立体网络规则加在**前额叶**（链里的"最后一层"）：5% 父单元 × 2 子单元 × 2 输入
    gkey = ("audio", 4)
    n_g0 = len(ids["pfc_groups"][gkey])
    newp = S.grow_pfc(B, ids, acts_all, sp, geo, "audio", 4, fraction=0.05)
    din2 = np.bincount(B.dst, minlength=B.n)
    check("加前额叶神经元生效",
          len(newp) > 0 and len(ids["pfc_groups"][gkey]) == n_g0 + len(newp),
          f"{n_g0} -> {len(ids['pfc_groups'][gkey])}")
    check("新前额叶神经元各 2 条输入", bool(np.all(din2[newp] == 2)),
          din2[newp].tolist())
    check("前额叶索引已重建", bool(np.isin(newp, ids["other_pfc"]).all()))

    # 4) 回程：边数 ≤ 前向一半；少量前额叶打不开、一大片能打开；不落回第 1 层
    fwd = int((B.group == GROUP_PFC_FWD).sum())
    ret_e = int(np.isin(B.group, [16, 17, 18, 19]).sum())
    check("回程边 ≤ 前向一半", ret_e <= 0.5 * fwd, f"{ret_e}/{fwd}")
    l1 = ids["layers"][("audio", 1)]
    r_l1 = np.r_[B.ids_of("ret_audio_L1_0"), B.ids_of("ret_audio_L1_1")]
    check("回程不落回第 1 层",
          int((np.isin(B.dst, l1) & np.isin(B.src, r_l1)).sum()) == 0)

    g = ids["pfc_groups"][("audio", 4)]
    g = g[B.alive[g]]
    ret = np.array([i for i in range(B.n) if B.region[i].startswith("ret_audio_L4_0")])
    ret = ret[B.alive[ret]] if len(ret) else ret

    def ret_act(on):
        B.act[:] = 0.0
        ext = np.zeros(B.n, dtype=np.float32)
        ext[on] = 5.0
        for _ in range(80):
            B.step(external=ext)
        return float(B.act[ret].max()) if len(ret) else 0.0

    if len(ret) and len(g) > 10:
        few, many = ret_act(g[:2]), ret_act(g[: max(3, int(0.9 * len(g)))])
        check("回程：只亮少量 → 打不开", few < 0.2, round(few, 3))
        check("回程：亮一大片 → 能打开", many > 0.2, round(many, 3))
    else:
        check("回程神经元存在", False, f"ret={len(ret)}, pfc={len(g)}")

    print("\n" + ("全部通过" if ok else "有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
