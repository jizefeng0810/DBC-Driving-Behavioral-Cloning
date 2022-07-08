import argparse

import numpy as np
import wandb
import cv2
import tqdm
import pandas as pd
import csv
from pathlib import Path

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from agents.image_agent import ImageAgent
from envs.carla_env import CarlaEnv

import sys
try:
    sys.path.append('egg_file/carla-0.9.9-py3.7-win-amd64.egg')
except IndexError:
    pass
import carla


VALID_AGENTS = [cls.__name__ for cls in [ImageAgent]]
EPISODE_LENGTH = 2000
# EPISODE_LENGTH = 200
IGNORE = 'video'


def rollout(env, agent, env_seed, config):
    env.reset(seed=env_seed, n_vehicles=0, n_pedestrians=0)    # 设置carla环境，车和行人的数量

    agent.reset()   # 更新环境

    action = None

    distance_traveled = 0.0
    speed = 0.0
    speed_counter = 0
    collided = False
    position = None
    images = list()

    csv_file = os.path.join(Path(wandb.run.dir), 'episode.csv')
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)  # 创建写的对象
        writer.writerow(["x", "y", 'angle', 'steer', 'throttle', 'brake'])  # 写入列的名称
        for step in tqdm.tqdm(range(EPISODE_LENGTH), leave=False, disable=not config['debug']):
            if collided:
                break

            spectator = env._world.get_spectator()
            spectator.set_transform(
                    carla.Transform(
                        env._player.get_location() + carla.Location(z=75),
                        carla.Rotation(pitch=-90)))
            env._world.debug.draw_string(
                    env._player.get_location() + carla.Location(z=15),
                    '%.2f %.2f %d' % (distance_traveled, speed, collided),
                    draw_shadow=False, color=carla.Color(255, 255, 255),
                    life_time=0.001)

            observations = env.step(action)
            action = agent.run_step(observations)

            # 写入csv
            pos_x, pos_y, _ = observations['position']
            steer = observations['steer']
            throttle = observations['throttle']
            brake = observations['brake']
            ori = observations['orientation']
            writer.writerow([pos_x, pos_y, np.arctan2(ori[1], ori[0]), steer, throttle, brake])

            if agent.debug_image is not None:
                if config['debug']:
                    cv2.imshow('debug', cv2.cvtColor(agent.debug_image, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(1)

                images.append(agent.debug_image.transpose(2, 0, 1))

            if not collided:
                collided = observations['collided']

                if step % 20 == 0:
                    if position is not None:
                        distance_traveled += np.linalg.norm(position - observations['position'])

                    position = observations['position']
                    speed_counter += 1
                    speed += (np.linalg.norm(observations['velocity']) - speed) / speed_counter

    return {
            'distance': distance_traveled,
            'speed': speed,
            'steps': step,
            'video': wandb.Video(np.array(images), fps=40, format='mp4')
            }


def main(agent_class, agent_args, config):
    np.random.seed(config['seed'])

    env_seeds = np.random.choice(range(10000), config['num_episodes'])  # 设置环境
    agent = agent_class(config['pid'], *agent_args)

    results = list()

    wandb.init(project='task-distillation-eval', config=config)
    # print(Path(wandb.run.dir))
    print(config)
    with CarlaEnv(town='Town04', port=config['port']) as env:
        for i, env_seed in enumerate(tqdm.tqdm(env_seeds, desc='Episode', disable=not config['debug'])):
            metrics = rollout(env, agent, env_seed, config)

            results.append(dict(metrics))
            results_csv = pd.DataFrame(results)

            for key in results_csv:
                if key in IGNORE:
                    continue

                values = np.array(results_csv[key].values)

                wandb.run.summary['%s_mean' % key] = np.mean(values)
                wandb.run.summary['%s_std' % key] = np.std(values)
                wandb.run.summary[key] = wandb.Histogram(values)

            wandb.log(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=3)

    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--agent_class', type=str, choices=VALID_AGENTS)
    parser.add_argument('--agent_args', nargs='*')

    args = parser.parse_args()

    agent_class = eval(args.agent_class)
    agent_args = args.agent_args

    config = {
            'port': args.port,
            'debug': args.debug,
            'pid': args.pid,
            'agent_class': args.agent_class,
            'agent_args': args.agent_args,
            'num_episodes': args.num_episodes,
            'seed': args.seed,
            }

    main(agent_class, agent_args, config)
