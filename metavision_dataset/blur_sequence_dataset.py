import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from metavision_dataset.inverse_my_gpu_corner_esim_v3 import My_GPUEBSimCorners



base_params = {
    'dataset_path': '/media/yuyang/Workspace/_Datasets_/my_corner_dataset/',
    'num_workers': 0,
    'batch_size': 6,
    'num_tbins': 5,     # BPTT
    'event_volume_depth': 10,
    'height': 180,
    'width': 240,
    'min_frames_per_video': 2, # 200,
    'max_frames_per_video': 3, # 6000,
    'raw_number_of_heatmaps_list': (19, 46, 82),    
    'randomize_noises': True,
    'data_device': 'cuda:0',

    'needed_number_of_heatmaps': 10, 
    'min_blur_ratio': 0.2,
    'max_blur_ratio': 0.6,
    'random_gamma_transform': True,
    'random_pepper_noise': True,
    'seed': 6,

    'epochs': 100,
}


class Blur_sequnece_dataloader(pl.LightningDataModule):
    """
    Simulation gives events + frames + corners
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.split_names = ['train', 'val']


    def get_dataloader(self, dataset_path, current_epoch):
        ## 
        len_list = len(self.hparams.raw_number_of_heatmaps_list)
        if current_epoch < 8 or current_epoch >= (self.hparams.epochs - 1):
            current_raw_number_of_heatmaps = sorted(self.hparams.raw_number_of_heatmaps_list)[len_list//2]
        else:
            current_raw_number_of_heatmaps = sorted(self.hparams.raw_number_of_heatmaps_list)[current_epoch%len_list]
        ## 
        current_seed = self.hparams.seed + current_epoch
        # print('current_raw_number_of_heatmaps: ', current_raw_number_of_heatmaps)
        # print('current_seed: ', current_seed)


        dataloader = My_GPUEBSimCorners.from_params(
            dataset_path,
            self.hparams.num_workers,
            self.hparams.batch_size,
            self.hparams.num_tbins,
            self.hparams.event_volume_depth,
            self.hparams.height,
            self.hparams.width,
            self.hparams.min_frames_per_video,
            self.hparams.max_frames_per_video,
            # self.hparams.raw_number_of_heatmaps_list,
            current_raw_number_of_heatmaps,
            self.hparams.randomize_noises,
            self.hparams.data_device,

            self.hparams.needed_number_of_heatmaps,
            self.hparams.min_blur_ratio,
            self.hparams.max_blur_ratio,
            self.hparams.random_gamma_transform,
            self.hparams.random_pepper_noise,
            # self.hparams.seed,
            current_seed,
        )
        return dataloader

    def train_dataloader(self):
        path = os.path.join(self.hparams.dataset_path, self.split_names[0])
        if self.trainer is not None:
            return self.get_dataloader(path, self.trainer.current_epoch)
        else:
            return self.get_dataloader(path, 0)

    def val_dataloader(self):
        path = os.path.join(self.hparams.dataset_path, self.split_names[1])
        if self.trainer is not None:
            return self.get_dataloader(path, self.trainer.current_epoch)
        else:
            return self.get_dataloader(path, 0)











