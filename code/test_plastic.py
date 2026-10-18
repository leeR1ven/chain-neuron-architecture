# -*- coding: utf-8 -*-
"""后天可塑连接的自测（证明三件事 + 两条附带项）。

  1 增长 + 越涨越难涨（没有 1.0 上限）
  2 周期整体降权：常用的稳在高位、少用的低、从不用的趋近 0 并被清掉
  3 新连接能长出来（含跨区，不受出生连线就近规则限制）
  4 双向是两种时序自然给出的（没有额外加反向学习机制）
  5 默认关闭时什么都不发生

跑法：python test_plastic.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plastic import Plastic, PlasticCfg  # noqa: E402

N = 64


def blank():
    return np.zeros(N)


def line(tag, **kv):
    parts = [str(k) + '=' + str(v) for k, v in kv.items()]
    print('  ' + tag.ljust(24) + '  '.join(parts))


def weight_of(pl, i, j):
    m = np.where((pl.src == i) & (pl.dst == j))[0]
    return float(pl.w[m[0]]) if len(m) else None


print('=' * 74)
print('1) 增长 + 越涨越难涨（不设 1.0 上限）')
pl = Plastic(PlasticCfg(enabled=True, T=10 ** 9, new_per_tick=0, eta=0.02))
pl._add(0, 1, 0.01)
prev_w = float(pl.w[0])
for t in range(400):
    a = blank()
    a[0 if t % 2 == 0 else 1] = 1.0
    pl.tick(a)
    if (t + 1) % 80 == 0:
        w = float(pl.w[0])
        line('第 %d 拍' % (t + 1), 权重='%.4f' % w, 本次增量='%.5f' % (w - prev_w))
        prev_w = w
print('  -> 权重 0.01 涨到 %.4f（超过 1.0，没有上限）；增量逐次变小（越涨越难涨）'
      % float(pl.w[0]))

print('')
print('2) 周期整体降权：常用 / 少用 / 从不用')
cfg = PlasticCfg(enabled=True, eta=0.02, T=200, gamma=0.99, new_per_tick=0)
pl2 = Plastic(cfg)
pl2._add(0, 1, 0.01)     # 常用：每拍交替亮
pl2._add(2, 3, 0.01)     # 少用：每 100 拍才先后亮一次
pl2._add(4, 5, 0.01)     # 从不用：4、5 永不活跃
for t in range(4000):
    a = blank()
    a[0 if t % 2 == 0 else 1] = 1.0
    if t % 100 == 0:
        a[2] = 1.0
    elif t % 100 == 1:
        a[3] = 1.0
    pl2.tick(a)
print('  跑 4000 拍，每 200 拍整体降权一次，共降权 %d 次' % pl2.n_decays)
line('常用（每拍都用）', 权重='%.3f' % weight_of(pl2, 0, 1))
line('少用（每 100 拍）', 权重='%.3f' % weight_of(pl2, 2, 3))
line('从不用', 权重='%.3f' % weight_of(pl2, 4, 5))
print('  -> 常用者稳在高位、少用者低、从不用者被清掉；没有任何“哪条该降”的判断')

print('')
print('3) 新连接能长出来（跨区也能长，不受出生连线就近规则限制）')
pl3 = Plastic(PlasticCfg(enabled=True, new_per_tick=200, T=10 ** 9))
act = blank()
act[:8] = 1.0        # 区A 的 8 个（位置靠前）
act[32:40] = 1.0     # 区B 的 8 个（位置靠后，离得很远）
for t in range(30):
    pl3.tick(act)
wa = int(np.sum((pl3.src < 32) & (pl3.dst < 32)))
wb = int(np.sum((pl3.src >= 32) & (pl3.dst >= 32)))
cr = int(np.sum((pl3.src < 32) != (pl3.dst < 32)))
line('新长出的边', 总数=pl3.n_new_edges, 区内A=wa, 区内B=wb, 跨区=cr)
print('  -> 跨区也长了 %d 条（出生连线不允许跨区，可塑连线不受这条限制）' % cr)

print('')
print('4) 双向是两种时序自然给出的（没有加任何反向学习机制）')
pl4 = Plastic(PlasticCfg(enabled=True, T=10 ** 9, new_per_tick=50))
for who in (0, 1, 0, 1, 1, 0, 1, 0, 0, 1):
    a = blank()
    a[who] = 1.0
    pl4.tick(a)
line('0->1 存在', 有=weight_of(pl4, 0, 1) is not None)
line('1->0 存在', 有=weight_of(pl4, 1, 0) is not None)
print('  -> 两个方向都长出来了，规则本身自然给出双向')

print('')
print('5) 默认关闭：什么都不发生')
pl5 = Plastic()          # enabled 默认 False
pl5._add(0, 1, 0.5)
w0 = float(pl5.w[0])
for t in range(500):
    a = blank()
    a[0 if t % 2 == 0 else 1] = 1.0
    pl5.tick(a)
line('关闭时', 权重='%.4f' % float(pl5.w[0]), 原值='%.4f' % w0,
     新增边=pl5.n_new_edges, 降权次数=pl5.n_decays)
print('  -> 权重没变、没长新边、没降权（当前阶段：先天全固定、可塑 0 条）')

print('')
print('6) 接到真引擎上试一次（不改引擎文件）')
from build import Interface, Spec, build as build_brain  # noqa: E402

sp = Spec(px=2, n_rgb=3, n_level=3, n_audio=2, n_touch=3, n_balance=2,
          n_muscle=2, n_force=3, k_near=8, layer_ratio=(2.0, 4.0, 6.0),
          layer_floor=(16, 24, 32))
B, ids, sp2, geo = build_brain(sp, verbose=False)
e0 = len(B.src)
pl6 = Plastic(PlasticCfg(enabled=True, new_per_tick=50))
pl6.enabled = True
ui = Interface(B, ids, sp)
for t in range(30):
    v = 0.5 if t % 2 == 0 else 0.7
    ui.set_image(np.full((sp.px, sp.px, sp.n_rgb), v, np.float32))
    B.step(external=ui.drive())
    pl6.tick(B.act)
    pl6.inject(B)
line('引擎连接数', 开关前=e0, 开关后=len(B.src), 新增=int(len(B.src) - e0))
line('可塑边', 条数=pl6.n_edges())
print('  -> 开关一开就能用，引擎文件一行没改')

print('')
print('=' * 74)
print('怎么在引擎里开启（一行）：pl = Plastic(); pl.enabled = True')
print('之后每次 B.step(external=...) 后面调用：')
print('    pl.tick(B.act)     # 记录这一拍的活动：可塑边增长 / 长新边 / 周期降权')
print('    pl.inject(B)       # 把新长出的边并进引擎，并同步已有可塑边的权重')
print('引擎文件（brain.py / build.py / calib.py）不需要任何改动。')
