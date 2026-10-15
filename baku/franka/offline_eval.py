#!/usr/bin/env python3
"""Runs a Franka BAKU snapshot over recorded demos, step by step exactly as on the
robot (temporal aggregation included), and compares its actions with the labels.

It is open loop (the recorded images, not the ones the policy's own actions
would produce), so it checks the pipeline and the fit, not task success.

    python franka/offline_eval.py <run>/snapshot/60000.pt --data /abs/expert_demos/franka_teleop \
        --demos 0:2 --plot /tmp/eval.png
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from franka.policy import FrankaPolicy  # noqa: E402

NAMES = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('snapshot')
    parser.add_argument('--data', required=True, help='dataset folder with <task>.pkl')
    parser.add_argument('--task', default=None, help='task pkl name (default: the snapshot\'s first task)')
    parser.add_argument('--demos', default='0:', help='demo index range a:b within the pkl (default all)')
    parser.add_argument('--plot', default=None, help='write a PNG of predicted vs label actions')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    policy = FrankaPolicy(args.snapshot, device=args.device, max_steps=5000)
    task = args.task or policy.info['task_names'][0]
    policy.set_task(task)
    with open(Path(args.data) / ('%s.pkl' % task), 'rb') as f:
        data = pickle.load(f)
    key = policy.info['dataset']['action_key']
    start, _, end = args.demos.partition(':')
    indices = range(int(start or 0), int(end) if end else len(data['observations']))
    scale = (policy.stats['actions']['max'] - policy.stats['actions']['min']) / 2.0

    results = []
    for i in indices:
        observation, label = data['observations'][i], np.asarray(data[key][i])
        policy.reset()
        predicted = []
        entropy = []
        for step in range(len(label)):
            frames = {k: observation[k][step].transpose(2, 0, 1) for k in policy.pixel_keys}
            predicted.append(policy.act(frames, preprocessed=True))
            entropy.append(policy.last.get('entropy_avg', np.nan))
        predicted = np.array(predicted)
        error = np.abs(predicted - label)
        gripper_ok = np.mean((predicted[:, 6] > 0.5) == (label[:, 6] > 0.5))
        results.append((data['episode_ids'][i], label, predicted, np.array(entropy)))
        print('%s (%d steps): MAE %s | as fraction of the action scale %s | gripper agreement %.0f%% | '
              'mean entropy %.2f' % (data['episode_ids'][i], len(label),
                                     ' '.join('%s %.4f' % (n, e) for n, e in zip(NAMES[:6], error[:, :6].mean(0))),
                                     np.round(error[:, :6].mean(0) / scale[:6], 2), 100 * gripper_ok,
                                     np.nanmean(entropy)))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(8, len(results), figsize=(5 * len(results), 14), sharex='col', squeeze=False)
        for column, (episode, label, predicted, entropy) in enumerate(results):
            for row in range(7):
                axes[row, column].plot(label[:, row], label='label', color='k')
                axes[row, column].plot(predicted[:, row], label='policy', color='tab:red', alpha=0.8)
                axes[row, column].set_ylabel(NAMES[row])
            axes[7, column].plot(entropy, color='tab:green')
            axes[7, column].set_ylabel('VQ-BeT entropy')
            axes[7, column].set_xlabel('step (%.0f Hz)' % policy.rate_hz)
            axes[0, column].set_title(episode)
        axes[0, 0].legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=80)
        print('plot:', args.plot)


if __name__ == '__main__':
    main()
