from pathlib import Path

import numpy as np
import torch
import torchvision

from scipy import interpolate

from models.network import Network
from utils.converter import ConverterTorch
from utils.common import load_yaml

Modular = False
###
if Modular:
    # Modularity and Abstract
    from PIL import Image
    from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
    from torchvision.transforms import ToTensor, ToPILImage
    from eval.erfnet import ERFNet
    from eval.transform import Relabel, Colorize
    NUM_CLASSES = 20

    image_transform = ToPILImage()
    input_transform_cityscapes = Compose([
        Resize((160,384),Image.BILINEAR),
        ToTensor(),
        #Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    cityscapes_trainIds2labelIds = Compose([
        Relabel(19, 255),
        Relabel(18, 33),
        Relabel(17, 32),
        Relabel(16, 31),
        Relabel(15, 28),
        Relabel(14, 27),
        Relabel(13, 26),
        Relabel(12, 25),
        Relabel(11, 24),
        Relabel(10, 23),
        Relabel(9, 22),
        Relabel(8, 21),
        Relabel(7, 20),
        Relabel(6, 19),
        Relabel(5, 17),
        Relabel(4, 13),
        Relabel(3, 12),
        Relabel(2, 11),
        Relabel(1, 8),
        Relabel(0, 7),
        Relabel(255, 0),
        ToPILImage(),
    ])

    model = ERFNet(NUM_CLASSES).to('cuda')
    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        return model
    model = torch.nn.DataParallel(model)
    model = load_my_state_dict(model, torch.load('save/erfnet_training1/model_best.pth'))
    model.eval()
###

def subsample(points, steps):
    dists = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(1))
    total_dist = dists.sum()

    result = [points[0]]
    cumulative = 0.0
    index = 0

    for i in range(1, steps):
        while index < len(dists) and cumulative < (i / steps) * total_dist:
            cumulative += dists[index]
            index += 1

        result.append(points[index])

    return np.array(result)


def spline(points, steps):
    t = np.linspace(0.0, 1.0, steps * 10)
    tck, u = interpolate.splprep(points.T, k=2, s=100.0)
    points = np.stack(interpolate.splev(t, tck, der=0), 1)

    return subsample(points, steps)


class Planner(object):
    def __init__(self, path_to_conf_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = torchvision.transforms.ToTensor()
        self.converter = ConverterTorch().to(self.device)

        self.target_index = 65
        self.speed_mult = 2.5

        path_to_conf_file = Path(path_to_conf_file)
        config = load_yaml(path_to_conf_file.parent / 'config.yaml')

        self.net = Network(**config['model_args']).to(self.device)
        self.net.load_state_dict(torch.load(path_to_conf_file))
        self.net.eval()

    @torch.no_grad()
    def run_step(self, rgb, rgb_forward, viz=None):
        if Modular:
            # Modularity and Abstract
            rgb = Image.fromarray(rgb).convert('RGB')
            img = input_transform_cityscapes(rgb)
            img = img.cuda().unsqueeze(0)
            rgb_forward = Image.fromarray(rgb_forward).convert('RGB')
            img_forward = input_transform_cityscapes(rgb_forward)
            img_forward = img_forward.cuda().unsqueeze(0)

            output = model(img)
            label = output[0].max(0)[1].byte().cpu().data
            label_color = Colorize()(label.unsqueeze(0))
            rgb = ToPILImage()(label_color)
            rgb.save('./seg.jpg')

            output = model(img_forward)
            label = output[0].max(0)[1].byte().cpu().data
            label_color = Colorize()(label.unsqueeze(0))
            rgb_forward = ToPILImage()(label_color)
            rgb_forward.save('./seg_2.jpg')

        img = self.transform(rgb).to(self.device).unsqueeze(0)
        img_forward = self.transform(rgb_forward).to(self.device).unsqueeze(0)

        # print(img_forward.shape)
        model_input = torch.cat((img_forward, img), 1)

        cam_coords = self.net(model_input)
        cam_coords[..., 0] = (cam_coords[..., 0] + 1) / 2 * img.shape[-1]   # rgb coords
        cam_coords[..., 1] = (cam_coords[..., 1] + 1) / 2 * img.shape[-2]

        map_coords = self.converter.cam_to_map(cam_coords).cpu().numpy().squeeze()
        world_coords = self.converter.cam_to_world(cam_coords).cpu().numpy().squeeze()

        target_speed = np.sqrt(((world_coords[:2] - world_coords[1:3]) ** 2).sum(1).mean())
        target_speed *= self.speed_mult

        theta1 = np.degrees(np.arctan2(world_coords[0][0], world_coords[0][1]))
        theta2 = np.degrees(np.arctan2(world_coords[4][0], world_coords[4][1]))
        # print(abs(theta2 - theta1))
        if abs(theta2 - theta1) < 2:
            target_speed *= self.speed_mult
        else:
            target_speed *= 1.2

        curve = spline(map_coords + 1e-8 * np.random.rand(*map_coords.shape), 100)
        target = curve[self.target_index]

        curve_world = spline(world_coords + 1e-8 * np.random.rand(*world_coords.shape), 100)
        target_world = curve_world[self.target_index]

        if viz:
            viz.planner_draw(cam_coords.cpu().numpy().squeeze(), map_coords, curve, target)

        return target_world, target_speed
