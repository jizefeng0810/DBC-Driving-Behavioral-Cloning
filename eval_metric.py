import argparse
import time

from pathlib import Path
from PIL import Image, ImageDraw

import tqdm
import numpy as np
import torch
import torchvision
import wandb
import cv2

from datasets import get_dataset, SOURCES
from models.network import Network
from utils.converter import ConverterTorch
from utils.common import visualize_birdview, load_yaml


def _log_visuals(rgb, birdview, loss, waypoints_map, waypoints, _waypoints):
    images = list()

    for i in range(birdview.shape[0]):
        canvas_rgb = np.uint8(rgb[i].detach().cpu().numpy().transpose(1, 2, 0) * 255).copy()
        canvas_rgb = Image.fromarray(canvas_rgb)
        draw_rgb = ImageDraw.Draw(canvas_rgb)

        canvas_map = np.uint8(birdview[i].detach().cpu().numpy().transpose(1, 2, 0) * 255).copy()
        canvas_map = visualize_birdview(canvas_map)
        canvas_map = Image.fromarray(canvas_map)
        draw_map = ImageDraw.Draw(canvas_map)

        def _dot(canvas, draw, i, j, color, rescale):
            if rescale:
                x = int((i + 1) / 2 * canvas.width)
                y = int((j + 1) / 2 * canvas.height)
            else:
                x = int(i)
                y = int(j)

            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

        for x, y in waypoints_map[i]:
            _dot(canvas_map, draw_map, x, y, (0, 0, 255), False)

        for x, y in waypoints[i]:
            _dot(canvas_rgb, draw_rgb, x, y, (0, 0, 255), True)

        for x, y in _waypoints[i]:
            _dot(canvas_rgb, draw_rgb, x, y, (255, 0, 0), True)

        loss_i = loss[i].sum()

        draw_rgb.text((5, 10), 'Loss: %.2f' % loss_i)

        k = min(canvas_rgb.size)
        canvas_map.thumbnail((k, k))
        canvas = np.hstack([np.array(x) for x in [canvas_rgb, canvas_map]])

        images.append((loss_i, torch.ByteTensor(np.uint8(canvas).transpose(2, 0, 1))))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images[:32]], nrow=4)
    # cv2.imshow('map', cv2.cvtColor(result.numpy(), cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)
    result = [wandb.Image(result.numpy().transpose(1, 2, 0))]

    return result

AUG_MAP_SIZE = 192
def net_eval(net, data, config):
    net.eval()
    MAE, RMSE, EVS = [], [], []
    converter = ConverterTorch()
    iterator = tqdm.tqdm(data, desc='val', total=len(data), position=1, leave=None)
    # wandb.run.summary['step'] = 0
    num_theta, total = 0, 0
    for i, (rgb_forward, rgb, mapview, waypoints, _) in enumerate(iterator):
        waypoints[..., 0] = AUG_MAP_SIZE - (waypoints[..., 0] + 1) * mapview.shape[-1] / 2
        waypoints[..., 1] = (waypoints[..., 1] + 1) * mapview.shape[-2] / 2

        points_cam = converter(waypoints)
        points_cam[..., 0] = (points_cam[..., 0] / converter.W) * 2 - 1
        points_cam[..., 1] = (points_cam[..., 1] / converter.H) * 2 - 1

        rgb_forward = rgb_forward.to(config['device'])
        rgb = rgb.to(config['device'])
        model_input = torch.cat((rgb_forward, rgb), 1)
        points_cam = points_cam.to(config['device'])
        _waypoints = net(model_input)

        mae = torch.abs(points_cam - _waypoints)
        mae_mean = mae.sum((1, 2)).mean()
        MAE.append(mae_mean.item())

        rmse = torch.pow((points_cam - _waypoints), 2)
        rmse_mean = rmse.sum((1, 2)).mean()
        RMSE.append(rmse_mean.item())

        y_true, y_pred = points_cam.cpu().numpy(), _waypoints.cpu().numpy()
        explained_variance_score = 1 - np.var(np.array(y_true) - np.array(y_pred)) / np.var(y_true)
        EVS.append(explained_variance_score)

        # points_cam = points_cam.cpu().numpy()
        # _waypoints = _waypoints.cpu().numpy()
        # for i in range(len(points_cam)):
        #     theta1 = np.degrees(np.arctan2(points_cam[i][4][0], points_cam[i][4][1]))
        #     theta2 = np.degrees(np.arctan2(_waypoints[i][4][0], _waypoints[i][4][1]))
        #     if abs(theta1-theta2) < 3:
        #         num_theta += 1
        # total += len(points_cam)

        # metrics = dict()
        # metrics['images'] = _log_visuals(rgb, mapview, loss,
        #             waypoints, points_cam, _waypoints)
        # wandb.run.summary['step'] += 1
        # wandb.log(
        #     {('%s/%s' % ('val', k)): v for k, v in metrics.items()},
        #     step=wandb.run.summary['step'])
    # print('-------------------')
    # print(num_theta, '   ', total)
    # print(num_theta * 1.0 / total)
    # print('-------------------')
    return np.mean(MAE), np.sqrt(np.mean(RMSE)), np.mean(EVS)

def main(config):
    # from thop import profile
    # net = Network(**config['model_args'])
    # input = torch.randn(1, 6, 160, 384)  # 模型输入的形状,batch_size=1
    # flops, params = profile(net, inputs=(input,))
    # print(flops / 1e9, 'G', params / 1e6)  # flops单位G，para单位M
    # aa


    data_train, data_val = get_dataset(config['source'])(**config['data_args'])
    net = Network(**config['model_args']).to(config['device'])
    net.load_state_dict(torch.load(str(config['model_path'])))

    # wandb.init(
    #     project='task-distillation-eval',
    #     config=config, id=config['run_name'], resume='auto')
    # wandb.save(str(Path(wandb.run.dir) / '*.t7'))

    with torch.no_grad():
        MAE_val, RMSE_val, EVS_val = net_eval(net, data_val, config)

    print(' MAE_val', MAE_val)
    print(' RMSE_val: ', RMSE_val)
    print(' EVS_val: ', EVS_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model args.
    parser.add_argument('--resnet_model', default='resnet18')
    parser.add_argument('--input_channels', type=int, default=6)
    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--model_path', type=Path, required=False)

    # Data args.
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=128)    #128
    parser.add_argument('--source', type=str, required=True, choices=SOURCES)

    parsed = parser.parse_args()

    keys = ['resnet_model', 'batch_size', 'temperature']
    run_name = 'stage2' + '_'.join(str(getattr(parsed, x)) for x in keys)

    config = {
            'run_name': run_name,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'source': parsed.source,
            'model_path': parsed.model_path,
            'model_args': {
                'temperature': parsed.temperature,
                'resnet_model': parsed.resnet_model,
                'input_channel': parsed.input_channels,
                },
            'data_args': {
                'dataset_dir': parsed.dataset_dir,
                'batch_size': parsed.batch_size,
                },
            }

    main(config)
