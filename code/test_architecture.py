# -*- coding: utf-8 -*-
"""架构级自测：规格校验 + 不变量 + 结构改动后仍然自洽。

用法：python test_architecture.py
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Config  # noqa: E402
from build import (DEPTH_GAIN, Geometry, Interface, Spec, build,  # noqa: E402
                   validate_spec)
from calib import ref_calibrate  # noqa: E402
from check import check_architecture  # noqa: E402
import structure as S  # noqa: E402


def main() -> int:
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra else ""))
        ok = ok and bool(cond)

    # ---------- 1) 规格校验：坏规格必须被拒 ----------
    base = Spec(px=3, n_level=4, n_audio=4, n_touch=6, n_balance=3,
                n_muscle=2, n_force=4, k_near=6,
                layer_ratio=(2.0, 3.0), layer_floor=(16, 24))
    geo = Geometry()
    check("好规格通过", validate_spec(base, geo) == [])
    bad = [
        ("layer_floor 长度不对", Spec(px=3, layer_ratio=(2.0, 3.0), layer_floor=(16,))),
        ("layer_ratio 不递增", Spec(px=3, layer_ratio=(3.0, 2.0), layer_floor=(16, 24))),
        ("层数超过 DEPTH 表", Spec(px=3, layer_ratio=(2.0, 3.0, 4.0, 5.0, 6.0),
                                   layer_floor=(16, 24, 32, 40, 48))),
        ("chain_mode 未知", Spec(px=3, layer_ratio=(2.0, 3.0),
                                 layer_floor=(16, 24), chain_mode="weird")),
    ]
    for name, sp in bad:
        try:
            errs = validate_spec(sp, geo)
            check(f"坏规格被拒：{name}", bool(errs), "; ".join(errs[:2]))
        except Exception as e:                       # noqa: BLE001
            check(f"坏规格被拒：{name}", True, str(e))
    g2 = Geometry(bands={"vision": (0.0, 0.5), "audio": (0.4, 0.7),
                         "touch": (0.7, 0.9), "balance": (0.9, 1.0)})
    check("条带重叠被拒", bool(validate_spec(base, g2)))

    # ---------- 2) 小规模整网自检 ----------
    B, ids, sp, geo = build(base, Config(), verbose=False)
    rep = check_architecture(B, ids, sp, geo)
    check("整网自检通过", rep["ok"], rep["issues"][:2])
    check("层数 = 1 + len(layer_ratio)",
          rep["stats"]["layers"] == 1 + len(sp.layer_ratio), rep["stats"]["layers"])

    # ---------- 3) 温度计：每对恰好一个；输入恰好一半活跃 ----------
    ui = Interface(B, ids, sp)
    n_chan = 3 * 3 * 3 + 4 + 6 + 3
    for v in (0.0, 0.5, 1.0):
        ui.set_values(v)
        ext = np.asarray(ui.drive(), dtype=np.float32)
        tot = 0
        for r in ("vision", "audio", "touch", "balance"):
            pos, neg = ids["pos"][r], ids["neg"][r]
            on = (ext[pos].reshape(len(pos), sp.n_level)
                  + ext[neg].reshape(len(neg), sp.n_level))
            tot += int(ext[pos].sum() + ext[neg].sum())
            if not np.allclose(on, 1.0):
                break
        check(f"温度计 v={v} 每对恰好一个", bool(np.allclose(on, 1.0)))
        check(f"温度计 v={v} 输入恰好一半活跃", tot == n_chan * sp.n_level,
              f"{tot} vs {n_chan * sp.n_level}")

    # ---------- 4) 结构改动后仍然自洽 ----------
    ref_calibrate(B, ids, ui, passes=1, steps=20)
    acts = S.observe(B, ui, S.make_change_probes(sp, per_region=2), 40)
    new = S.grow_layer(B, ids, acts, sp, geo, "audio", 2, fraction=0.2,
                       children=2, k_in=2, seed=0)
    check("加层神经元有产出", len(new) > 0)
    rep2 = check_architecture(B, ids, sp, geo)
    check("加层后自检通过", rep2["ok"], rep2["issues"][:2])
    check("新神经元有中继和额叶接收端",
          int((np.isin(B.src, new)).sum()) > 0)

    # ---------- 5) 减神经元/筛常亮后仍然自洽 ----------
    B2, ids2 = copy.deepcopy(B), copy.deepcopy(ids)
    ui2 = Interface(B2, ids2, sp)
    acts2 = S.observe(B2, ui2, S.make_change_probes(sp, per_region=2), 40)
    S.prune_useless(B2, ids2, acts2, regions=("vision",))
    rep3 = check_architecture(B2, ids2, sp, geo)
    check("删神经元后自检通过", rep3["ok"], rep3["issues"][:2])

    print("\n" + ("全部通过" if ok else "有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
