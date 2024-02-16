import sys
# sys.path.append('./')
sys.path.append("/usr/lib/python3/dist-packages/")

import argparse
import numpy as np
import os
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from metavision_core_ml.utils.train_utils import search_latest_checkpoint
from metavision_dataset.blur_sequence_dataset import Blur_sequnece_dataloader
from model.my_lightning_model_selfsupervise_step1 import CornerDetectionCallback, CornerDetectionLightningModel

torch.manual_seed(0)
np.random.seed(0)


def main(raw_args=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # dir params
    parser.add_argument('--root_dir', type=str, default='./train_logs_selfsupervise_step1/', help='logging directory')
    parser.add_argument('--dataset_path', type=str, default='/media/yuyang/Workspace/_Datasets_/my_corner_dataset/',
                        help='path of folder containing train and val folders containing images')

    # train params
    parser.add_argument('--lr', type=float, default=0.0003, help='base learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--checkpoint', type=str, default='', help='resume from specific checkpoint')
    parser.add_argument('--limit_train_batches', type=int, default=300, help='run training epoch for X batches')
    parser.add_argument('--limit_val_batches', type=int, default=50, help='run training epoch for X batches')
    parser.add_argument('--randomize_noises', action='store_true', help='randomize noises in the simulator')


    parser.add_argument('--demo_iter', type=int, default=50, help='run demo for X iterations')
    parser.add_argument('--precision', type=int, default=32, help='precision 32 or 16')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='accumulate gradient for more than a single batch')
    parser.add_argument('--demo_every', type=int, default=1, help='run demo every X epoch')
    parser.add_argument('--val_every', type=int, default=1, help='validate every X epochs')
    parser.add_argument('--save_every', type=int, default=1, help='save every X epochs')
    parser.add_argument('--just_test', action='store_true', help='launches demo video')
    parser.add_argument('--cpu', default=False, help='use cpu')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--mask_loss_no_events_yet', action='store_true', help='mask loss where no events')
    parser.add_argument('--data_device', type=str, default='cuda:0', help='run simulation on the cpu/gpu')
    parser.add_argument('--event_volume_depth', type=int, default=10, help='event volume depth')

    # data params
    parser.add_argument('--height', type=int, default=180, help='image height')
    parser.add_argument('--width', type=int, default=240, help='image width')
    parser.add_argument('--num_tbins', type=int, default=10, help="timesteps per batch tbppt")
    parser.add_argument('--min_frames_per_video', type=int, default=200, help='max frames per video')
    parser.add_argument('--max_frames_per_video', type=int, default=6000, help='max frames per video')
    parser.add_argument('--num_workers', type=int, default=4, help='number of threads')
    parser.add_argument('--raw_number_of_heatmaps_list', type=int, default=(19, 46, 82), help='number of raw corner heatmaps list')
    parser.add_argument('--needed_number_of_heatmaps', type=int, default=10, help='number of target corner heatmaps')
    parser.add_argument('--min_blur_ratio', type=float, default=0.2, help='min ratio for clear images to generate blurred image')
    parser.add_argument('--max_blur_ratio', type=float, default=0.6, help='max ratio for clear images to generate blurred image')
    parser.add_argument('--random_gamma_transform', default=True, help='Whether to perform random gamma transform on image intensity')
    parser.add_argument('--random_pepper_noise', default=True, help='Whether to add pepper noise on event representation')
    parser.add_argument('--seed', type=int, default=0, help='random seed for dataset')
    # parser.add_argument('--gradient_clip_val', type=float, default=10.0, help='The value at which to clip gradients')               # TODO: test
    parser.add_argument('--reload_dataloaders_every_n_epochs', type=int, default=1, help='Set to a non-negative integer to reload dataloaders every n epochs')    # TODO: test


    params, _ = parser.parse_known_args(raw_args)
    print('pl version: ', pl.__version__)
    params.cin = params.event_volume_depth
    params.cout = params.needed_number_of_heatmaps
    print(params)

    model = CornerDetectionLightningModel(params)
    if not params.cpu:
        model.cuda()
    else:
        params.data_device = "cpu"

    if params.resume:
        ckpt = search_latest_checkpoint(params.root_dir)
    elif params.checkpoint != "":
        ckpt = params.checkpoint
    else:
        ckpt = None
    print('ckpt: ', ckpt)

    tmpdir = os.path.join(params.root_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, every_n_epochs=params.save_every)

    logger = TensorBoardLogger(save_dir=os.path.join(params.root_dir, 'logs'))

    ## if ckpt is not None and params.just_test:
    if ckpt is not None:
        print('Loading checkpoint {}'.format(ckpt))
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu') if params.cpu else torch.device("cuda"))
        model.load_state_dict(checkpoint['state_dict'])

    ## Data Setup
    data = Blur_sequnece_dataloader(params)

    if params.just_test:
        if not params.cpu:
            model = model.cuda()
        model.video(data.val_dataloader(), -1)
    else:
        demo_callback = CornerDetectionCallback(data, params.demo_every)
        trainer = pl.Trainer(
            default_root_dir=params.root_dir,
            callbacks=[checkpoint_callback, demo_callback],
            logger=logger,
            accelerator="cpu" if params.cpu else "gpu",
            # gpus=0 if params.cpu else 1,
            precision=params.precision,
            accumulate_grad_batches=params.accumulate_grad_batches,
            max_epochs=params.epochs,
            # resume_from_checkpoint=ckpt,
            log_every_n_steps=5,
            limit_train_batches=params.limit_train_batches,
            limit_val_batches=params.limit_val_batches,

            # detect_anomaly=True,     # check anomaly
            # gradient_clip_val=params.gradient_clip_val,
            reload_dataloaders_every_n_epochs=params.reload_dataloaders_every_n_epochs
        )

        trainer.fit(model, data)


if __name__ == '__main__':
    main()