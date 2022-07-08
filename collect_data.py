import argparse
import pathlib

import csv
import os
import numpy as np
import tqdm
import sys
try:
    sys.path.append('egg_file/carla-0.9.9-py3.7-win-amd64.egg')
    sys.path.append(r'F:\software\carla\WindowsNoEditor\PythonAPI\carla')
    import carla
    from carla import ColorConverter as cc
except:
    raise ImportError('Please check your carla file')
import cv2

from PIL import Image

from envs.carla_env import CarlaEnv
from utils.common import visualize_birdview, colorize_segmentation

import math

def save(save_dir, observations, step, debug):
    rgb = observations['rgb']
    birdview = observations['birdview']
    segmentation = observations['segmentation']

    pos = observations['position']
    ori = observations['orientation']
    measurements = np.float32([pos[0], pos[1], pos[2], np.arctan2(ori[1], ori[0])])
    # print(math.degrees(np.arctan2(ori[1], ori[0])))

    if debug:
        cv2.imshow('rgb', cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_BGR2RGB))
        cv2.imshow('birdview', cv2.cvtColor(visualize_birdview(birdview), cv2.COLOR_BGR2RGB))
        cv2.imshow('segmentation', cv2.cvtColor(colorize_segmentation(segmentation), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
    else:
        np.save(save_dir / ('measurements_%04d' % step), measurements)
        np.save(save_dir / ('birdview_%04d' % step), birdview)

        Image.fromarray(rgb).save(save_dir / ('image_%04d.png' % step))
        Image.fromarray(segmentation).save(save_dir / ('segmentation_%04d.png' % step))



def collect_episode(env, save_dir, episode_length, frame_skip, debug):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for step in tqdm.tqdm(range(episode_length)):
        spectator = env._world.get_spectator()
        spectator.set_transform(
                carla.Transform(
                    env._player.get_location() + carla.Location(z=75),
                    carla.Rotation(pitch=-90)))

        observations = env.step()

        if step % frame_skip == 0:
            save(save_dir, observations, step // frame_skip, debug)
    cv2.destroyAllWindows()

    if not debug:
        # 写episode.csv
        csv_file = os.path.join(save_dir, 'episode.csv')
        with open(csv_file, "w", newline='') as f:
            writer = csv.writer(f)  # 创建写的对象
            writer.writerow(["step", "x", "y", 'z', 'angle'])  # 写入列的名称

            for i in range(episode_length // frame_skip):
                npy_file = os.path.join(save_dir, 'measurements_%04d.npy' % i)
                x, y, z, angle = np.load(npy_file)
                angle_degree = math.degrees(angle)
                angle = angle_degree if angle_degree >= 0 else angle_degree + 360
                writer.writerow([i, x, y, z, angle])

def main(config):
    np.random.seed(1)

    with CarlaEnv(town='Town04', port=config.port) as env:
        for episode in range(config.episodes):
            env.reset(
                    n_vehicles=0,
                    # n_vehicles=np.random.choice([10, 30, 40]),  # the number of vehicles
                    n_pedestrians=0, seed=episode)
            env._player.set_autopilot(True)

            collect_episode(
                    env,
                    config.save_dir / ('%03d' % episode),
                    config.episode_length, config.frame_skip, config.debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--save_dir', type=pathlib.Path, default='F:\\carla_datasets')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--frame_skip', type=int, default=5)
    parser.add_argument('--debug', action='store_true', default=False)

    main(parser.parse_args())