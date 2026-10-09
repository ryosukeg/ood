"""Aggregate the evaluation jsons into summary.md / summary.json (per object and over all objects).

    python aggregate.py [--set test|train]
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OBJS = ['box', 'capsulemachine', 'espressomachine', 'laptop', 'microwave', 'notebook', 'waffleiron', 'ketchup',
        'mixer', 'phone']
JOINT_LIMIT = {'espressomachine': 1.5}  # others: pi
COND_NOTE = {
    'normal/grasp': '論文プロトコル: 持ち上げ0.3+-0.15 m, 水平+-0.15 m, 回転+-0.3 rad',
    'out/grasp_xy3': '水平移動 3倍 (+-0.45 m)、他は通常',
    'out/grasp_lift0.8': '持ち上げ高さ 0.8+-0.15 m (通常は0.3+-0.15)、他は通常',
    'out/grasp_rot3': '回転 3倍 (+-0.9 rad)、他は通常',
    'out/grasp_all': '水平3倍 + 持ち上げ0.8 + 回転3倍',
    'normal/arti': '目標角 0.5, 0.75, 1.0, 1.25, 1.5 rad (学習時 U(0.5,1.5))',
    'out/arti_angle': '目標角 0.1, 0.25, 1.8, 2.4, 3.0 rad (学習範囲外。espressoは上限1.5radのため0.1/0.25のみ)',
    'normal/compose': '水平-0.1..-0.05 m, 回転+-0.4 (ketchup +-0.3) rad, 開き 0.5..0.6 (ketchup 0.5..1.0) rad',
    'out/compose_pos': '水平 -0.3..-0.15 m (3倍)',
    'out/compose_ori': '回転 3倍 (+-1.2 rad, ketchup +-0.9 rad)',
    'out/compose_angle': '開き 1.7..1.8 rad (espressoは1.4..1.5で学習範囲内=実質OODなし)',
    'out/compose_all': '水平3倍 + 回転3倍 + 開き1.7..1.8 rad',
}


def load(root, sub):
    res = {}
    for obj in OBJS:
        p = os.path.join(root, sub, f'{obj}.json')
        if os.path.exists(p):
            res[obj] = json.load(open(p))
    return res


def m(x):
    return float(np.mean(x)) if len(x) else float('nan')


def grasp_stats(eps):
    s = np.array([e['success'] for e in eps])
    pe = np.array([e['final_pos_error'] for e in eps])
    ae = np.array([e['final_ori_error'] for e in eps])
    ok = s > 0.5
    return {'n': len(eps), 'suc': m(s), 'pe': m(pe[ok]), 'ae': m(ae[ok]), 'pe_all': m(pe), 'ae_all': m(ae)}


def arti_eps(obj, eps):
    lim = JOINT_LIMIT.get(obj, np.pi)
    return [e for e in eps if e['target_angle'] <= lim + 1e-6]


def arti_stats(eps):
    s = np.array([e['success'] for e in eps])
    aae = np.array([e['final_angle_error'] for e in eps])
    sd = np.array([e['sim_distance'] for e in eps])
    ok = s > 0.5
    return {'n': len(eps), 'suc': m(s), 'aae': m(aae[ok]), 'aae_all': m(aae), 'reach': m(aae < 0.5),
            'sd': m(sd[ok]), 'sd_all': m(sd)}


def compose_stats(tr):
    pe = np.array([t['pe'] for t in tr])
    ae = np.array([t['ae'] for t in tr])
    aae = np.array([t['aae'] for t in tr])
    return {'n': len(tr), 'suc_t': m([t['success'] for t in tr]), 'pe': m(pe), 'ae': m(ae), 'aae': m(aae),
            'pe_ok': m([t['pe_ok'] for t in tr]), 'ae_ok': m([t['ae_ok'] for t in tr]),
            'aae_ok': m([t['aae_ok'] for t in tr]), 'legacy': m([t['legacy_success'] for t in tr])}


def episodes_of(task, obj, d):
    if task == 'compose':
        return d['trials']
    eps = d['episodes']
    return arti_eps(obj, eps) if task == 'arti' else eps


STAT = {'grasp': grasp_stats, 'arti': arti_stats, 'compose': compose_stats}
COLS = {
    'grasp': [('n', 'n', '{:d}'), ('suc', 'Suc.R [%]', '{:.1%}'), ('pe', 'PE [m] (成功のみ)', '{:.3f}'),
              ('ae', 'AE [rad] (成功のみ)', '{:.3f}'), ('pe_all', 'PE 全体 [m]', '{:.3f}'),
              ('ae_all', 'AE 全体 [rad]', '{:.3f}')],
    'arti': [('n', 'n', '{:d}'), ('suc', 'Suc.R [%]', '{:.1%}'), ('aae', 'AAE [rad] (成功のみ)', '{:.3f}'),
             ('aae_all', 'AAE 全体 [rad]', '{:.3f}'), ('reach', 'AAE<0.5 達成率', '{:.1%}'),
             ('sd_all', '平均シミュ距離 全体', '{:.3f}')],
    'compose': [('n', 'n', '{:d}'), ('suc_t', 'Suc.T [%]', '{:.1%}'), ('pe', 'PE [m]', '{:.3f}'),
                ('ae', 'AE [rad]', '{:.3f}'), ('aae', 'AAE [rad]', '{:.3f}'), ('pe_ok', 'PE<0.05', '{:.1%}'),
                ('ae_ok', 'AE<0.2', '{:.1%}'), ('aae_ok', 'AAE<0.5', '{:.1%}'), ('legacy', '旧判定Suc [%]', '{:.1%}')],
}


def per_object(task, sub, root):
    data = load(root, sub)
    rows, pooled = {}, []
    for obj, d in data.items():
        eps = episodes_of(task, obj, d)
        if not eps:
            continue
        rows[obj] = STAT[task](eps)
        pooled += eps
    if not rows:
        return None
    keys = [k for k in rows[next(iter(rows))] if k != 'n']
    macro = {k: m([r[k] for r in rows.values() if not np.isnan(r[k])]) for k in keys}
    macro['n'] = sum(r['n'] for r in rows.values())
    pool = STAT[task](pooled)
    return {'objects': rows, 'macro': macro, 'pooled': pool}


def table(task, res):
    cols = COLS[task]
    out = ['| 物体 | ' + ' | '.join(c[1] for c in cols) + ' |', '|' + '---|' * (len(cols) + 1)]

    def row(name, r):
        cells = []
        for k, _, f in cols:
            v = r[k]
            cells.append('-' if (isinstance(v, float) and np.isnan(v)) else f.format(int(v) if f == '{:d}' else v))
        out.append(f'| {name} | ' + ' | '.join(cells) + ' |')

    for obj, r in res['objects'].items():
        row(obj, r)
    row('**全物体 (物体平均)**', res['macro'])
    row('**全物体 (全エピソード)**', res['pooled'])
    return '\n'.join(out)


def arti_by_target(sub, root):
    data = load(root, sub)
    by = {}
    for obj, d in data.items():
        for e in arti_eps(obj, d['episodes']):
            by.setdefault(round(e['target_angle'], 2), []).append(e)
    lines = ['| 目標角 [rad] | n | Suc.R [%] | AAE 全体 [rad] | AAE<0.5 達成率 | 到達角の平均 [rad] |', '|---|---|---|---|---|---|']
    for t in sorted(by):
        es = by[t]
        lines.append(f"| {t:.2f} | {len(es)} | {m([e['success'] for e in es]):.1%} | "
                     f"{m([e['final_angle_error'] for e in es]):.3f} | {m([e['final_angle_error'] < 0.5 for e in es]):.1%} | "
                     f"{m([e['final_angle'] for e in es]):.2f} |")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='test')
    args = ap.parse_args()
    root = os.path.join(ROOT, args.set)
    plan = {'grasp': ['normal/grasp', 'out/grasp_xy3', 'out/grasp_lift0.8', 'out/grasp_rot3', 'out/grasp_all'],
            'arti': ['normal/arti', 'out/arti_angle'],
            'compose': ['normal/compose', 'out/compose_pos', 'out/compose_ori', 'out/compose_angle',
                        'out/compose_all']}
    title = {'grasp': '把持タスク (floating base, 把持して目標位置・姿勢へ運ぶ)', 'arti': '開閉タスク (floating base)',
             'compose': '複合タスク (把持 -> 運搬 -> 開閉)'}
    summary, md = {}, [f'# 目標状態の汎化評価 ({args.set} set)\n']
    md.append('Suc.R/Suc.T、PE、AE、AAE は ArtiGrasp 論文・評価コードと同じ指標。Normal は論文プロトコル、Out は目標状態を学習範囲外へ変更。')
    md.append('各条件はラベル(把持参照)ごとに複数目標を1回ずつ試行した結果 (単一シード)。\n')
    for task, subs in plan.items():
        md.append(f'\n## {title[task]}\n')
        compare = []
        for sub in subs:
            res = per_object(task, sub, root)
            if res is None:
                continue
            summary[sub] = res
            name = sub.split('/')[-1] if sub.startswith('out') else 'normal'
            md.append(f'\n### {name}: {COND_NOTE[sub]}\n')
            md.append(table(task, res))
            if sub == 'out/arti_angle':
                md.append('\n目標角別 (全物体):\n')
                md.append(arti_by_target(sub, root))
            if sub == 'normal/arti':
                md.append('\n目標角別 (全物体):\n')
                md.append(arti_by_target(sub, root))
            compare.append((name, res))
        if len(compare) > 1:
            key = {'grasp': ('suc', 'pe_all', 'ae_all'), 'arti': ('suc', 'aae_all', 'reach'),
                   'compose': ('suc_t', 'pe', 'ae', 'aae')}[task]
            label = {'suc': 'Suc.R', 'pe_all': 'PE全体[m]', 'ae_all': 'AE全体[rad]', 'aae_all': 'AAE全体[rad]',
                     'reach': 'AAE<0.5', 'suc_t': 'Suc.T', 'pe': 'PE[m]', 'ae': 'AE[rad]', 'aae': 'AAE[rad]'}
            md.append(f'\n### {title[task]}: Normal と Out の比較 (全物体は物体平均)\n')
            head = '| 条件 | ' + ' | '.join(label[k] for k in key) + ' |'
            md += [head, '|' + '---|' * (len(key) + 1)]
            for name, res in compare:
                md.append(f'| {name} | ' + ' | '.join(
                    f"{res['macro'][k]:.1%}" if k in ('suc', 'suc_t', 'reach') else f"{res['macro'][k]:.3f}" for k in key) + ' |')
    out_dir = root
    open(os.path.join(out_dir, 'summary.md'), 'w').write('\n'.join(md) + '\n')
    json.dump(summary, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
