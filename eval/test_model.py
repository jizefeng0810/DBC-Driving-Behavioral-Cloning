# Code to produce colored segmentation output in Pytorch for all cityscapes subsets
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import os
import importlib

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from eval.dataset import cityscapes
from eval.erfnet import ERFNet
from eval.transform import Relabel, ToLabel, Colorize

import visdom

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((160, 384), Image.BILINEAR),
    ToTensor(),
    # Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform_cityscapes = Compose([
    Resize((160, 384), Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),  # ignore label to 19
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


def main(args):
    model = ERFNet(NUM_CLASSES)

    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(r'G:\CARLA-sequence-task-transfer\save\erfnet_training1\model_best.pth'))
    print("Model and weights LOADED successfully")

    model.eval()

    if (not os.path.exists(args.datadir)):
        print("Error: datadir could not be loaded")

    with open(r'G:\CARLA-sequence-task-transfer\seg_0.png', 'rb') as f:
        image = Image.open(f).convert('RGB')
    image = input_transform_cityscapes(image)

    images = image.cuda().unsqueeze(0)

    # inputs = Variable(images)
    # with torch.no_grad():
    outputs = model(images)

    label = outputs[0].max(0)[1].byte().cpu().data
    label_color = Colorize()(label.unsqueeze(0))

    label_save = ToPILImage()(label_color)
    label_save.save('./seggggggg_1.png')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../save/erfnet_training1/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # can be val, test, train, demoSequence

    parser.add_argument('--datadir', default="/datasets/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
