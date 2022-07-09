# DBC-Driving-Behavioral-Cloning

Prepared paper: Driving Behavioral Cloning for Autonomous Vehicle Using Task Distillation

## Approach
<div align=center><img src="img/framework.jpg"></div>

## How to use code
### Datasets
You can download dataset from: [BaiduPan](waiting)
### Training
#### Stage 1
Below script gives you an example of training a teacher model.
```
python train_stage1.py --dataset_dir ./CarlaDatasets/Route0_Noon --source stage1 --input_channels 4
```
#### Stage 2
Below script gives you an example of training a model with teacher model trained in stage 1.
```
python train_stage2.py --dataset_dir ./CarlaDatasets/Route0_Noon --source stage2 --input_channel 6 --teacher_path ./wandb/stage1_output_model.t7
```


## Simulator：Carla 0.9.9.2
Download [link](https://github.com/carla-simulator/carla)

## Results
The result is as follows:
<div align=center><img src="img/example-res.jpg"></div>
The video demonstrations of all the above experiments can be found at [Youtube](https://youtu.be/IPHR-7awYk0)

