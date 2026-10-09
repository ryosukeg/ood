"""Run the ArtiGrasp goal-state evaluation (in-distribution 'normal' vs out-of-distribution 'out') for all objects.

Run from anywhere with the artigrasp conda env active:
    python run_eval.py [--set test|train] [--only normal|out]
Finished jobs (existing json) are skipped, so the script can be restarted.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GYM = os.path.expanduser('~/goto/artigrasp/raisimGymTorch')
ENVS = 'raisimGymTorch/env/envs'

GRASP_OBJS = ['box', 'capsulemachine', 'espressomachine', 'laptop', 'microwave', 'notebook', 'waffleiron',
              'ketchup', 'mixer', 'phone']
ARTI_OBJS = ['box', 'capsulemachine', 'espressomachine', 'laptop', 'microwave', 'notebook', 'waffleiron']
COMPOSE_OBJS = ARTI_OBJS + ['ketchup']
MAX_LABELS = 15
MAX_TRIALS = 60

# joint upper limits below pi cannot be driven past them (espresso 1.5)
COMPOSE_OUT_ANGLE = {'espressomachine': '1.4,1.5'}

GRASP_COND = {
    'normal': [],
    'out/grasp_xy3': ['-xy_scale', '3'],
    'out/grasp_lift0.8': ['-lift', '0.8', '-xy_scale', '1'],
    'out/grasp_rot3': ['-rot_scale', '3'],
    'out/grasp_all': ['-xy_scale', '3', '-lift', '0.8', '-rot_scale', '3'],
}


def compose_cond(obj):
    ketchup = obj == 'ketchup'
    ori = '0.9' if ketchup else '1.2'
    angle = COMPOSE_OUT_ANGLE.get(obj, '1.7,1.8')
    return {
        'normal': [],
        'out/compose_pos': ['--pos_range=-0.3,-0.15'],
        'out/compose_ori': ['-ori_range', ori],
        'out/compose_angle': ['-angle_range', angle],
        'out/compose_all': ['--pos_range=-0.3,-0.15', '-ori_range', ori, '-angle_range', angle],
    }


def jobs(only):
    out = []
    for obj in GRASP_OBJS:
        for cond, extra in GRASP_COND.items():
            out.append(('grasp', cond, obj, extra))
    for obj in ARTI_OBJS:
        out.append(('arti', 'normal', obj, []))
        out.append(('arti', 'out/arti_angle', obj, ['-angles', '0.1,0.25,1.8,2.4,3.0']))
    for obj in COMPOSE_OBJS:
        for cond, extra in compose_cond(obj).items():
            out.append(('compose', cond, obj, extra))
    if only:
        out = [j for j in out if (j[1] == 'normal') == (only == 'normal')]
    # normal first so partial results are already comparable
    return sorted(out, key=lambda j: j[1] != 'normal')


def command(task, obj, extra, path, test):
    base = ['-obj', obj, '-slow', '0', '-headless']
    if test:
        base.append('-test')
    if task in ('grasp', 'arti'):
        script = f'{ENVS}/floating_evaluation/runner_eval.py'
        cmd = base + ['-max_labels', str(MAX_LABELS), '-episodes_out', path]
        if task == 'grasp':
            cmd.append('-grasp')
    elif obj == 'ketchup':
        script = f'{ENVS}/compose_eval/ketchup_eval_strict.py'
        cmd = base + ['-out', path]
    else:
        script = f'{ENVS}/compose_eval/compose_eval_strict.py'
        cmd = base + ['-max_trials', str(MAX_TRIALS), '-out', path]
    return [sys.executable, script] + cmd + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='test', choices=['test', 'train'])
    ap.add_argument('--only', default=None, choices=['normal', 'out'])
    args = ap.parse_args()
    root = os.path.join(ROOT, args.set)
    todo = jobs(args.only)
    for n, (task, cond, obj, extra) in enumerate(todo, 1):
        sub = cond if cond != 'normal' else f'normal/{task}'
        path = os.path.join(root, sub, f'{obj}.json')
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log = os.path.join(root, 'logs', sub.replace('/', '_') + f'_{obj}.log')
        os.makedirs(os.path.dirname(log), exist_ok=True)
        cmd = command(task, obj, extra, path, args.set == 'test')
        print(f'[{n}/{len(todo)}] {sub} {obj}', flush=True)
        with open(log, 'w') as f:
            r = subprocess.run(cmd, cwd=GYM, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not os.path.exists(path):
            print(f'  FAILED (rc={r.returncode}), see {log}', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
