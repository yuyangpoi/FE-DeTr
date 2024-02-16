'''
将loss修改为自监督
Last edit: 20230727
'''
import torch
import torch.nn as nn
import pytorch_lightning as pl

import os
import argparse
import numpy as np

from tqdm import tqdm
from itertools import islice

from metavision_core_ml.utils.show_or_write import ShowWrite
from model.my_net_vv3 import FrameEventNet

from .self_supervise_loss_step1 import consistensy_loss, peaky_loss


class CornerDetectionCallback(pl.callbacks.Callback):
    """
    callbacks to our model
    """

    def __init__(self, data_module, video_result_every_n_epochs=2):
        super().__init__()
        self.data_module = data_module
        self.video_every = int(video_result_every_n_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        torch.save(pl_module.model,
                   os.path.join(trainer.log_dir, "whole-model-epoch-{}.ckpt".format(trainer.current_epoch)))
        if trainer.current_epoch==0 or not (trainer.current_epoch % self.video_every):
            pl_module.video(self.data_module.train_dataloader(), trainer.current_epoch, set="train")
            pl_module.video(self.data_module.val_dataloader(), trainer.current_epoch, set="val")
        ## Adjust peaky_loss alpha
        if trainer.current_epoch in pl_module.peaky_loss_alpha_milestones:
            pl_module.peaky_loss_alpha *= pl_module.peaky_loss_alpha_gamma


class CornerDetectionLightningModel(pl.LightningModule):
    """
    Corner Detection: Train your model to predict corners as a heatmap
    """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(hparams, logger=False)
        self.model = FrameEventNet(frame_cin=1, event_cin=self.hparams.cin, cout=self.hparams.cout)

        self.overlap_heat_map = True
        self.sigmoid_fn = torch.nn.Sigmoid()

        self.consistensy_loss_alpha = 1.0
        self.peaky_loss_alpha = 0.25

        self.peaky_loss_alpha_milestones = [6, 12, 18]
        self.peaky_loss_alpha_gamma = 2.0


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def compute_loss(self, frame, events, corner_mask, reset_mask,
                     rotation_vectors, translation_vectors,
                     camera_nts, camera_depths,
                     camera_Ks, camera_Kinvs,
                     origin_sizes,
                     masks
                     ):
        loss_dict = {}
        self.model.reset(reset_mask)
        self.model.detach()

        # print(events.shape)   # [T, B, C, H, W]
        pred_list = []
        for i in range(events.shape[0]):
            pred_list.append(self.sigmoid_fn(self.model(frame[i].float()/255, events[i].float())))   # [B, C, H, W]
        pred = torch.stack(pred_list, dim=0)
        # print(pred.shape)     # [T, B, C=10, H, W]
        loss_dict["consistensy_loss"] = self.consistensy_loss_alpha * consistensy_loss(pred,
                                                         rotation_vectors, translation_vectors,
                                                         camera_nts, camera_depths,
                                                         camera_Ks, camera_Kinvs,
                                                         origin_sizes, (self.hparams.height, self.hparams.width))
        loss_dict["peaky_loss"] = self.peaky_loss_alpha * peaky_loss(pred, valid_mask=None)

        return loss_dict


    def training_step(self, batch, batch_nb):
        # loss_dict = self.compute_loss(batch["blurred_images"], batch["events"], batch["corners"], batch["reset"], batch["homos"])
        loss_dict = self.compute_loss(batch["blurred_images"], batch["events"], batch["corners"], batch["reset"],
                                      batch["rotation_vectors"], batch['translation_vectors'],
                                      batch["camera_nts"], batch['camera_depths'],
                                      batch["camera_Ks"], batch['camera_Kinvs'],
                                      batch['origin_sizes'],
                                      None # batch['masks'],
                                      )
        loss = sum([v for k, v in loss_dict.items()])
        logs = {'loss': loss}
        logs.update({'train_' + k: v.item() for k, v in loss_dict.items()})

        learning_rate = self.optimizers().state_dict()['param_groups'][0]['lr']
        self.log('learning_rate', learning_rate)               
        self.log('peaky_loss_alpha', self.peaky_loss_alpha)    

        for k, v in loss_dict.items():
            print('{}: {}'.format(k, v))
        print('lr: {}'.format(learning_rate))
        print('alpha: {}'.format(self.peaky_loss_alpha))

        self.log('train_loss', loss)
        for k, v in loss_dict.items():
            self.log('train_' + k, v)

        return logs


    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[6, 12, 18], gamma=0.75, verbose=True)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict


    def make_heat_map_image(self, pred, divide_max=True):
        image = np.zeros((pred.shape[3], pred.shape[4]))
        for t in range(pred.shape[2]):
            pred_t = pred[0, 0, t]
            image = image + pred_t.cpu().numpy()
        if (image.max() != 0) and divide_max:
            image /= image.max()
        image *= 255
        image = np.concatenate([np.expand_dims(image, 2)] * 3, axis=2)
        return image.astype(np.uint8)

    def make_color_heat_map_image(self, pred, threshold=0.1):
        image = np.zeros((pred.shape[3], pred.shape[4], 3))
        pred = pred.cpu().numpy()
        for t in range(pred.shape[2]):
            pred_t = 1*(pred[0, 0, t] > threshold)
            image[pred_t != 0] = np.array([0, (pred.shape[2]-1-t)*(int(255/pred.shape[2])), 255])
        return image.astype(np.uint8)


    def image_from_events(self, events):
        events = events.sum(2).unsqueeze(2)
        events_as_image = 255 * (events > 0) + 0 * (events < 0) + 128 * (events == 0)
        return events_as_image


    def video(self, dataloader, epoch=0, set="val"):
        """

        Args:
            dataloader: data loader from train or val set
            epoch: epoch
            set: can be either train or val

        Returns:

        """
        print('Start Video on {} set **************************************************************'.format(set))

        self.model.eval()

        video_name = os.path.join(self.hparams.root_dir, 'videos', 'video_{}_{}.mp4'.format(set, epoch))
        dir = os.path.dirname(video_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        show_write = ShowWrite(False, video_name)

        with torch.no_grad():
            for batch in tqdm(islice(dataloader, self.hparams.demo_iter), total=self.hparams.demo_iter):
                frames = batch["blurred_images"].to(self.device)
                events = batch["events"].to(self.device)
                self.model.reset(batch["reset"])

                if 'masks' in batch.keys():
                    masks = batch["masks"].to(self.device)
                else:
                    masks = torch.ones_like(frames, device=self.device)


                # pred = self.model(events.float())
                pred_list = []
                for i in range(events.shape[0]):
                    pred_list.append(self.model(frames[i].float()/255, events[i].float()))
                pred = torch.stack(pred_list, dim=0)    # [T, B, C, H, W]

                # Draw GT corners on images
                ground_truth = batch["corners"]
                image = self.image_from_events(events)

                for t in range(pred.shape[0]):
                    heat_map_gt = self.make_color_heat_map_image(ground_truth[t, 0].unsqueeze(0).unsqueeze(1) / 255.)

                    pred_sigmoid = self.sigmoid_fn(pred[t, 0].unsqueeze(0).unsqueeze(1))
                    heat_map_image = self.make_color_heat_map_image(pred_sigmoid, threshold=0.8)

                    heatmaps = (pred_sigmoid[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    heatmap_first = np.stack([heatmaps[0]] * 3, axis=-1)
                    heatmap_mid_0 = np.stack([heatmaps[heatmaps.shape[0] // 2-2]] * 3, axis=-1)
                    heatmap_mid_1 = np.stack([heatmaps[heatmaps.shape[0] // 2]] * 3, axis=-1)
                    heatmap_mid_2 = np.stack([heatmaps[heatmaps.shape[0] // 2 + 2]] * 3, axis=-1)
                    heatmap_last = np.stack([heatmaps[-1]] * 3, axis=-1)
                    # print('last_heatmap.shape: ', last_heatmap.shape)

                    events_image = image[t, 0, 0].cpu().numpy().astype(np.uint8)
                    events_image = np.concatenate([np.expand_dims(events_image, 2)] * 3, axis=2)

                    blurred_image = frames[t, 0, 0].cpu().numpy().astype(np.uint8)
                    blurred_image = np.concatenate([np.expand_dims(blurred_image, 2)] * 3, axis=2)

                    valid_mask = (masks[t, 0, 0]*255).cpu().numpy().astype(np.uint8)
                    valid_mask = np.concatenate([np.expand_dims(valid_mask, 2)] * 3, axis=2)

                    processed_valid_mask = torch.nn.functional.max_pool2d(masks[t], kernel_size=7, stride=1, padding=3)
                    processed_valid_mask = (processed_valid_mask[0, 0]*255).cpu().numpy().astype(np.uint8)
                    processed_valid_mask = np.concatenate([np.expand_dims(processed_valid_mask, 2)] * 3, axis=2)



                    if self.overlap_heat_map:
                        heat_map_gt_mask = heat_map_gt.sum(2) == 0
                        heat_map_gt[heat_map_gt_mask] = events_image[heat_map_gt_mask]
                        heat_map_gt[~heat_map_gt_mask] = heat_map_gt[~heat_map_gt_mask]
                        heat_map_image_mask = heat_map_image.sum(2) == 0
                        heat_map_image[heat_map_image_mask] = events_image[heat_map_image_mask]
                        heat_map_image[~heat_map_image_mask] = heat_map_image[~heat_map_image_mask]
                    # cat = np.concatenate([events_image, heat_map_gt, heat_map_image], axis=1)   # [180, 240*n, 3]
                    cat_0 = np.concatenate([blurred_image, heat_map_gt, heat_map_image, valid_mask, processed_valid_mask], axis=1)  # [180, 240*4, 3]
                    cat_1 = np.concatenate([heatmap_first, heatmap_mid_0, heatmap_mid_1, heatmap_mid_2, heatmap_last], axis=1)  # [180, 240*4, 3]
                    cat = np.concatenate([cat_0, cat_1], axis=0)  # [180*2, 240*4, 3]
                    # event_image is an image created from events
                    # heatmap gt is the ground truth heatmap of corners overlaid with the events
                    # heatmap image is the predicted corners overlaid with the events
                    show_write(cat)

        self.model.train()
        print('Finish Video on {} set **************************************************************'.format(set))



