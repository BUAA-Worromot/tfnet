import numpy as np
import pickle
import os
from dataset import Dataset
from ..config import Config


def read_split( kind: str):
    fn = "KSDD2/split_246.pyb"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == 'train':
            return train_samples
        elif kind == 'test':
            return test_samples
        else:
            raise Exception('Unknown')


class Ksdd2Dataset(Dataset):
    def __init__(self, mode:str, cfg:Config):
        super(Ksdd2Dataset, self).__init__(cfg.ksdd2_Dataset_dir, cfg, mode)
        self.read_contents()


    def read_contents(self):
        pos_samples, neg_samples = [], []
        data_points = read_split(self.mode)

        for part, is_seged in data_points:
            image_path = os.path.join(self.path, self.mode.lower(), f"{part}.png")
            seg_mask_path = os.path.join(self.path, self.mode.lower(), f"{part}_GT.png")

            image = self.read_img_resize(image_path,self.grayscale,self.img_size)
            seg_mask, positive = self.read_label_resize(seg_mask_path,self.img_size,self.cfg.dilation)

            if positive:
                image = self.to_tensor(image)
                seg_loss_mask = self.distance_transform(seg_mask,self.cfg.max_val,self.cfg.p)
                seg_loss_mask = self.to_tensor(self.downsize(seg_mask))
                pos_samples.append((image, seg_mask, seg_loss_mask, is_seged, image_path, seg_mask_path,part))

            else:
                image = self.to_tensor(image)
                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                neg_samples.append((image,seg_mask, seg_loss_mask, True, image_path,seg_mask_path,part))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2*len(pos_samples) if self.mode in ['train'] else len(pos_samples) + len(neg_samples)

        self.init_extra()