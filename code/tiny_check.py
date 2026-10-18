# -*- coding: utf-8 -*-
"""逐神经元一致性核对：驱动、门槛、带宽、gain、按公式算出的活动、实际活动，并排打出来。

只要有一行对不上，问题就当场现形（前几轮那些矛盾都是这么找出来的）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import Interface, Spec, build  # noqa: E402
from calib import ref_calibrate  # noqa: E402

sp = Spec(px=2, n_rgb=3, n_level=3, n_audio=2, n_touch=3, n_balance=2,
          n_muscle=2, n_force=3, k_near=8, layer_ratio=(2.0, 4.0, 6.0),
          layer_floor=(16, 24, 32))
B, ids, sp, geo = build(sp, verbose=False)
ui = Interface(B, ids, sp)
ref_calibrate(B, ids, ui)
fwd = np.array([not B.region[int(x)].startswith("ret_") for x in B.src])
sw = np.bincount(B.dst[fwd], weights=B.w[fwd], minlength=B.n)

P, C = sp.px, sp.n_rgb
ui.set_image(np.full((P, P, C), 0.33, np.float32))
ui.set_signals(audio=np.zeros(sp.n_audio), touch=np.zeros(sp.n_touch),
               balance=np.zeros(sp.n_balance))
B.act[:] = 0.0
for _ in range(120):
    B.step(external=ui.drive())
d = np.bincount(B.dst, weights=B.sgn * B.w * B.act[B.src], minlength=B.n)
gain = B.gain
print(f"gain={gain:.4f}")
for nm in ("vision_L2", "vision_L3", "vision_L4"):
    g = ids[nm] if nm in ids else B.ids_of(nm)
    print(f"\n{nm}")
    for t in list(g)[:3]:
        t = int(t)
        m = B.dst == t
        srcs = sorted(set(B.region[int(x)] for x in B.src[m]))
        thr = B.thr[t]; soft = B.soft[t]
        eff = max(thr * (1 + gain), 0.02)
        calc = min(max((d[t] - eff) / soft, 0.0), 1.0)
        print(f"  #{t}: 入边{int(m.sum())}条 来源{srcs}")
        print(f"      入边权重和(全部) {B.w[m].sum():.3f}  前向权重和 {sw[t]:.3f}")
        print(f"      门槛 {thr:.4f} 有效门槛 {eff:.4f} 带宽 {soft:.4f}")
        print(f"      驱动 {d[t]:.4f} → 公式算出活动 {calc:.4f} ；实际活动 {B.act[t]:.4f}")
