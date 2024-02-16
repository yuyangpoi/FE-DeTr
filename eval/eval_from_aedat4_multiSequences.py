'''
last edit: 20230714
Yuyang
'''
import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import numpy as np
import torch
import time
import os
import glob
import argparse
import pytorch_lightning as pl
import csv

from metavision_core_ml.preprocessing.event_to_tensor_torch import event_volume
from metavision_core_ml.corner_detection.corner_tracker import CornerTracker
from metavision_core_ml.corner_detection.utils import update_nn_tracker, save_nn_corners

from model.my_lightning_model_selfsupervise_step2_refine import CornerDetectionLightningModel
from davis_utils.buffer import FrameBuffer, EventBuffer


def parse_argument():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## dir params
    parser.add_argument('--file_dir', type=str, default='/media/yuyang/Workspace/_Datasets_/blurred_corners_dataset/Release/Normal/',
                        help='The name of a aedat4 dir')
    parser.add_argument('--save_path', type=str,
                        default='./5_eval_result_normal/',
                        help='The save path for result .csv')

    ## model params
    parser.add_argument('--cpu', default=False, help='use cpu')
    parser.add_argument('--checkpoint', type=str,
                        default='../final_checkpoints/epoch=0-step=125.ckpt',
                        help='checkpoint')
    parser.add_argument('--data_device', type=str, default='cuda:0', help='run simulation on the cpu/gpu')
    parser.add_argument('--event_volume_depth', type=int, default=10, help='event volume depth')
    parser.add_argument('--needed_number_of_heatmaps', type=int, default=10, help='number of target corner heatmaps')


    parser.add_argument('--show', default=True, help='if show detection and tracking process')

    return parser


def get_reader(file_path):
    assert os.path.exists(file_path), 'The file \'{}\' is not exist'.format(file_path)
    camera_reader = dv.io.MonoCameraRecording(file_path)

    return camera_reader



def init_model(params):
    print('pl version: ', pl.__version__)
    params.cin = params.event_volume_depth
    params.cout = params.needed_number_of_heatmaps
    print(params)

    model = CornerDetectionLightningModel(params)
    if not params.cpu:
        model.cuda()
    else:
        params.data_device = "cpu"

    ckpt = params.checkpoint
    print('ckpt: ', ckpt)

    checkpoint = torch.load(ckpt, map_location=torch.device('cpu') if params.cpu else torch.device("cuda"))
    model.load_state_dict(checkpoint['state_dict'])

    return model



from metavision_core_ml.core.temporal_modules import time_to_batch, batch_to_time
def clean_pred(pred, threshold):
    """
    Create a binary mask from a prediction between 0 and 1 after removal of local maximas
    Args:
        pred: prediction of the network after the sigmoid layer TxBxCxHxW
        threshold: Value of local maximas to consider corners

    Returns:
    Binary mask of corners locations.
    """
    def torch_nms(input_tensor, kernel_size):
        """
        runs non maximal suppression on square patches of size x size on the two last dimension
        Args:
            input_tensor: torch tensor of shape B, C, H, W
            kernel_size (int): size of the side of the square patch for NMS

        Returns:
            torch tensor where local maximas are unchanged and all other values are -inf
        """
        B, C, H, W = input_tensor.shape
        val, idx = torch.nn.functional.max_pool2d(input_tensor, kernel_size=kernel_size, return_indices=True)
        offsets = torch.arange(B * C, device=input_tensor.device) * H * W
        offsets = offsets.repeat_interleave(H // kernel_size).repeat_interleave(W // kernel_size).reshape(B, C,
                                                                                                          H // kernel_size,
                                                                                                          W // kernel_size)
        output_tensor = torch.ones_like(input_tensor) * float("-inf")
        output_tensor.view(-1)[idx + offsets] = val

        return output_tensor
    pred, batch_size = time_to_batch(pred)
    pred = torch_nms(pred, kernel_size=9)
    if pred.is_cuda:
        threshold = torch.tensor(threshold).to(pred.get_device())
    else:
        threshold = torch.as_tensor(threshold)
    pred_thresholded = 1 * (pred > threshold.view(-1, 1, 1, 1))
    pred = batch_to_time(pred_thresholded, batch_size)
    return pred



@torch.no_grad()
def model_pred_and_save(lightning_model, frame, event_representaion, ts, duration, tracker, csv_writer=None):
    '''
        frame: [T, B=1, 1, H, W]
        event_representaion: [T, B=1, C, H, W]
    '''
    lightning_model.model.eval()
    T, B, C, _, _ = event_representaion.shape
    assert B == 1
    delta_t = float(duration) / T

    channels_to_predict = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    print('ts begin: ', ts)
    for t in range(T):
        ts += delta_t
        pred = lightning_model.model(frame[t].float() / 255, event_representaion[t].float())
        pred = lightning_model.sigmoid_fn(pred)

        pred = clean_pred(pred.unsqueeze(0), threshold=0.95)
        num_predicted_time_steps = len(channels_to_predict)
        for index_channel_to_predict in channels_to_predict:
            interval_size = delta_t / 10
            num_interval = num_predicted_time_steps - 1 - index_channel_to_predict
            ts_from_multi_time_steps = ts - num_interval * interval_size
            # print('ts_from_multi_time_steps: ', ts_from_multi_time_steps)

            y, x = torch.where((pred[0, 0, index_channel_to_predict, :, :] > 0).squeeze())
            tracker = update_nn_tracker(tracker, x, y, ts_from_multi_time_steps)

            ## save result
            if csv_writer is not None:
                save_nn_corners(tracker, csv_writer, ts_from_multi_time_steps)


    print('ts end: ', ts)
    return None, ts



if __name__ == '__main__':
    ## Get params
    args, _ = parse_argument().parse_known_args(None)
    print(args)

    file_name_list = glob.glob(os.path.join(args.file_dir, '*.aedat4'))
    print(file_name_list)


    for file_name in file_name_list:
        ## Create aedat4 reader
        reader = get_reader(file_name)
        camera_name = reader.getCameraName()
        width, height = reader.getFrameResolution()
        print('Camera name: {}'.format(camera_name))

        ## init model
        args.height, args.width = height, width
        lightning_model = init_model(args)
        device = lightning_model.device
        with torch.no_grad():
            lightning_model.model.eval()
            print('Model is warming up...')
            for i in range(5):
                lightning_model.model(torch.rand((1, 1, height, width), device=device).float(),
                                      torch.rand((1, 10, height, width), device=device).float()) # warm up


        ## Initialize a visualizer for the overlay
        visualizer = dv.visualization.EventVisualizer(reader.getEventResolution(),
                                                      dv.visualization.colors.white(),
                                                      dv.visualization.colors.blue(),
                                                      dv.visualization.colors.red())

        ## Create a window for image display
        if args.show:
            cv.namedWindow("Preview", cv.WINDOW_NORMAL)

        ## Create buffer for images and events
        frame_buffer = FrameBuffer(60)


        ## Create tracker
        tracker = None
        ts_accumulated = 0  # timestamp accumulated from 0

        ## Create csv writer
        csv_save_path = os.path.join(args.save_path, os.path.basename(file_name).split('.')[0]+'.csv')
        print(csv_save_path)
        csv_file = open(csv_save_path, 'w')
        csv_writer = csv.writer(csv_file)


        ## Continue the loop while both cameras are connected
        while reader.isRunning():

            frame = reader.getNextFrame()
            if frame is not None:
                frame_buffer.push(frame)
                if len(frame_buffer) >= 2:

                    ## Init tracker
                    if tracker is None:
                        prev_frame = frame_buffer.get_prev_frame()
                        cur_frame = frame_buffer.get_cur_frame()
                        frame_interval = cur_frame.timestamp - prev_frame.timestamp
                        print('frame_interval: ', frame_interval)
                        tracker = CornerTracker(time_tolerance=frame_interval, distance_tolerance=5)  # time tolerance can be set to frame_interval or half


                    time_0 = time.time()

                    prev_frame = frame_buffer.get_prev_frame()
                    cur_frame = frame_buffer.get_cur_frame()

                    prev_exposure = prev_frame.exposure.microseconds
                    cur_exposure = cur_frame.exposure.microseconds

                    start_time, end_time = prev_frame.timestamp+prev_exposure, cur_frame.timestamp+cur_exposure

                    events_between = reader.getEventsTimeRange(start_time, end_time)
                    events_between_tensor, _, _ = EventBuffer.store_to_tensor(events_between, batch_idx=0, device=device)


                    ## 
                    frame_tensor = frame_buffer.frame_to_tensor(cur_frame, device=device) # [1, H, W]
                    event_tensor = events_between_tensor    # [N, 5]
                    start_times_for_representation = torch.FloatTensor([0]).view(1, ).to(device)
                    durations_for_representation = torch.FloatTensor([end_time - start_time]).view(1, ).to(device)
                    event_representation = event_volume(event_tensor, 1, height, width,
                                                        start_times_for_representation, durations_for_representation,
                                                        10,
                                                        'bilinear').squeeze(0)  # [C, H, W]


                    _, ts_accumulated = model_pred_and_save(lightning_model,
                                              frame_tensor.unsqueeze(0).unsqueeze(0),
                                              event_representation.unsqueeze(0).unsqueeze(0),
                                              ts_accumulated,
                                              end_time - start_time,
                                              tracker,
                                              csv_writer)
                    print('time： ', time.time() - time_0)



                    ## TODO: test show
                    if args.show:
                        cv.imshow('Preview', tracker.show(cur_frame.image))

                    # If escape button is pressed (code 27 is escape key), exit the program cleanly
                    if cv.waitKey(2) == 27:

                        exit(0)


                    print('end_time - start_time: ', end_time - start_time)
        csv_file.close()


