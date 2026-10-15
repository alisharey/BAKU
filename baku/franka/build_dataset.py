#!/usr/bin/env python3
"""Exported Franka episodes (tools/baku/export.sh) -> BAKU expert-demo PKLs.

One PKL per task (all episodes with the same task text), read by
read_data/franka_teleop.py:

    {
      'observations': [ per demo: {
          'pixels':            (T, S, S, 3) uint8 RGB   third-person camera
          'pixels_egocentric': (T, S, S, 3) uint8 RGB   wrist camera
          'cartesian_states':  (T, 7) ee position + quaternion (x y z w), base frame
          'gripper_states':    (T,) 1 = open, 0 = closed
          'gripper_width':     (T,) measured opening [m]
          'joint_states':      (T, 7) [rad] } ],
      'actions':             [ (T, 7) TeleMoMa action as executed (see export_episodes.py) ],
      'delta_pose_actions':  [ (T, 7) next-step delta pose (dp, rotation vector) + gripper ],
      'task', 'task_emb' (all-MiniLM-L6-v2, as LIBERO), 'episode_ids', 'rate_hz',
      'image_specs', 'img_size', 'format'
    }

The camera keys match LIBERO's (pixels = agent view, pixels_egocentric = wrist),
so a LIBERO-trained BAKU sees the same token order.

    python franka/build_dataset.py /path/to/export/* --out /path/to/expert_demos/franka \
        --task "put the cube in the box" --subsample 3
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from franka import images  # noqa: E402

FORMAT = 'baku_franka_demos/1'


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def decode(path, wanted, spec, size):
    """Frames `wanted` (sorted indices, repeats allowed) of an MP4 -> (N, size, size, 3) RGB."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError('cannot open %s' % path)
    needed = set(int(i) for i in wanted)
    last = max(needed)
    kept = {}
    index = 0
    while index <= last:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in needed:
            kept[index] = images.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), spec, size)
        index += 1
    capture.release()
    missing = needed - set(kept)
    if missing:
        raise IOError('%s: %d frames missing (video has %d), e.g. %d' % (path, len(missing), index, min(missing)))
    return np.stack([kept[int(i)] for i in wanted])


def delta_pose(positions, quaternions, gripper):
    rotations = Rotation.from_quat(quaternions)
    out = np.zeros((len(positions), 7), dtype=np.float32)
    out[:-1, :3] = positions[1:] - positions[:-1]
    out[:-1, 3:6] = (rotations[1:] * rotations[:-1].inv()).as_rotvec()
    out[:, 6] = gripper
    return out


def load_episode(folder, args, camera_keys):
    folder = Path(folder)
    info = json.loads((folder / 'episode.json').read_text())
    data = np.load(folder / 'lowdim.npz')
    n = len(data['t'])
    keep = np.arange(n)
    if not args.no_trim:
        driven = np.nonzero(data['movement_enabled'])[0]
        if len(driven) == 0:
            raise ValueError('the arm was never driven')
        margin = int(round(args.trim_margin_s * info['fps']))
        keep = np.arange(max(driven[0] - margin, 0), min(driven[-1] + margin, n - 1) + 1)
    keep = keep[::args.subsample]
    if len(keep) < 2:
        raise ValueError('too short after trimming')

    observation = {}
    for key, (camera, spec) in camera_keys.items():
        observation[key] = decode(folder / info['cameras'][camera]['file'], data['frame_' + camera][keep], spec,
                                  args.size)
    positions = data['ee_pos'][keep]
    quaternions = data['ee_quat'][keep]
    gripper = data['gripper_open'][keep]
    observation['cartesian_states'] = np.concatenate([positions, quaternions], axis=1).astype(np.float32)
    observation['gripper_states'] = gripper.astype(np.float32)
    observation['gripper_width'] = data['gripper_width'][keep].astype(np.float32)
    observation['joint_states'] = data['q'][keep].astype(np.float32)
    action = data['action'][keep].astype(np.float32)
    return info, observation, action, delta_pose(positions, quaternions, gripper), info['fps'] / args.subsample


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episodes', nargs='+', help='exported episode folders (with lowdim.npz)')
    parser.add_argument('--out', required=True, help='folder for <task>.pkl (expert_dataset.path)')
    parser.add_argument('--task', default=None, help='task text for every episode (default: each episode.json task)')
    parser.add_argument('--agent-camera', default='cam_2', help='camera for "pixels" (third-person)')
    parser.add_argument('--wrist-camera', default='cam_1', help='camera for "pixels_egocentric"')
    parser.add_argument('--agent-image', default='resize', help='image spec for the agent camera (franka/images.py)')
    parser.add_argument('--wrist-image', default='resize', help='image spec for the wrist camera')
    parser.add_argument('--size', type=int, default=128, help='square image size (LIBERO BAKU: 128)')
    parser.add_argument('--subsample', type=int, default=3, help='keep every n-th frame (30 Hz / 3 = 10 Hz)')
    parser.add_argument('--no-trim', action='store_true', help='keep the idle start and end')
    parser.add_argument('--trim-margin-s', type=float, default=0.5, help='idle time kept around the driven part')
    parser.add_argument('--include-failed', action='store_true', help='also use episodes marked success = false')
    args = parser.parse_args()
    for spec in (args.agent_image, args.wrist_image):
        images.parse(spec)
    camera_keys = {'pixels': (args.agent_camera, args.agent_image),
                   'pixels_egocentric': (args.wrist_camera, args.wrist_image)}

    by_task = {}
    for folder in args.episodes:
        try:
            info, observation, action, delta, rate = load_episode(folder, args, camera_keys)
        except Exception as error:
            print('%s: SKIPPED (%s)' % (folder, error))
            continue
        task = args.task if args.task is not None else info['task']
        if not task:
            print('%s: SKIPPED (no task text: set it in the recorder or pass --task)' % folder)
            continue
        if info.get('success') is False and not args.include_failed:
            print('%s: SKIPPED (marked failed)' % folder)
            continue
        by_task.setdefault(task, []).append((info['episode_id'], observation, action, delta, rate))
        print('%s: %d steps at %.1f Hz' % (info['episode_id'], len(action), rate))
    if not by_task:
        sys.exit('no usable episodes')

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')  # tiny; leave the GPU alone
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'format': FORMAT, 'size': args.size, 'subsample': args.subsample,
                'image_specs': {k: list(v) for k, v in camera_keys.items()}, 'trim': not args.no_trim,
                'trim_margin_s': args.trim_margin_s, 'tasks': {}}
    for task, demos in sorted(by_task.items()):
        rates = [d[4] for d in demos]
        payload = {
            'format': FORMAT,
            'task': task,
            'task_emb': encoder.encode(task),
            'observations': [d[1] for d in demos],
            'actions': [d[2] for d in demos],
            'delta_pose_actions': [d[3] for d in demos],
            'episode_ids': [d[0] for d in demos],
            'rate_hz': float(np.mean(rates)),
            'image_specs': {k: list(v) for k, v in camera_keys.items()},
            'img_size': args.size,
        }
        path = out / ('%s.pkl' % slug(task))
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        manifest['tasks'][slug(task)] = {'task': task, 'episodes': payload['episode_ids'],
                                         'steps': int(sum(len(d[2]) for d in demos)), 'rate_hz': payload['rate_hz']}
        print('%s: %d demos, %d steps -> %s' % (task, len(demos), manifest['tasks'][slug(task)]['steps'], path))
    (out / 'dataset.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
