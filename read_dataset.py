
from config import Config
from torch.utils.data import DataLoader
from typing import Optional

from dataset.input_ksdd2 import Ksdd2Dataset


def get_dataset(mode, cfg) ->Optional[DataLoader]:
    if cfg.dataset == "ksdd2":
        dataset = Ksdd2Dataset(mode, cfg)
    else:
        raise Exception(f"{cfg.dataset}不存在")

    shuffle = mode=='train'
    batchsize = cfg.batch_size
    num_workers = 0
    drop_last = mode=='train'
    pin_memory = False

    return DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)


