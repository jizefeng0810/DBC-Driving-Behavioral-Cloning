from . import carla_dataset
from . import stage1_dataset
from . import stage2_dataset
from . import eval_dataset

SOURCES = {
        'carla': carla_dataset.get_dataset,
        'stage1': stage1_dataset.get_dataset,
        'stage2': stage2_dataset.get_dataset,
        'eval': eval_dataset.get_dataset,
        }


def get_dataset(source):
    return SOURCES[source]
