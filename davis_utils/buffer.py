'''
The multi stream slicer in Python only supports events as main stream, so we need to customize some buffers to implement specific functions
last edit: 20230711
Yuyang
'''
import dv_processing as dv
from datetime import timedelta
import numpy as np
import torch
import cv2
import time

import rospy
from sensor_msgs.msg import Imu as IMU_ROS
import geometry_msgs.msg



class FrameBuffer:
    def __init__(self, max_buffer_len: int):
        assert max_buffer_len >= 2
        self.frame_list = []
        self.max_buffer_len = max_buffer_len

    def __len__(self):
        return len(self.frame_list)

    # def push(self, image: np.ndarray, timestamp: int):
    def push(self, frame: dv.Frame):
        self.frame_list.append(frame)
        if len(self.frame_list) > self.max_buffer_len:
            self.frame_list.pop(0)

    def get_cur_frame(self):
        return self.frame_list[-1]

    def get_prev_frame(self):
        return self.frame_list[-2]


    @staticmethod
    def frame_to_ndarray(frame: dv.Frame):
        image = frame.image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image    # [H, W], 0~255


    @staticmethod
    def frame_to_tensor(frame: dv.Frame, device='cpu'):
        image_ndarray = FrameBuffer.frame_to_ndarray(frame)
        image_tensor = torch.as_tensor(image_ndarray, dtype=torch.float32).to(device).unsqueeze(0)
        return image_tensor # [C=1, H, W], 0~255





class EventBuffer:
    def __init__(self, max_buffer_time_duration_microsecond: int):
        '''
        max_buffer_time_duration: int
        '''
        self.event_store = dv.EventStore()
        self.max_buffer_time_duration = timedelta(microseconds=max_buffer_time_duration_microsecond)

    def __len__(self):
        return len(self.event_store)

    def push(self, events: dv.EventStore):
        self.event_store.add(events)

        if self.event_store.duration() > self.max_buffer_time_duration:
            # erase_start_time = self.event_store.getLowestTime()        # start time to be erased
            # erase_end_time = None       # start time to be erased
            # self.event_store.eraseTime(erase_start_time, erase_end_time)
            self.event_store.retainDuration(self.max_buffer_time_duration)

    def getLowestTime(self):
        return self.event_store.getLowestTime()

    def getHighestTime(self):
        return self.event_store.getHighestTime()


    def get_between(self, start_time, end_time):
        '''
        Slice event within time range, such as [12000; 16000); the end time is exclusive
        '''
        return self.event_store.sliceTime(start_time, end_time)


    @staticmethod
    def store_to_ndarray(events: dv.EventStore) -> (np.ndarray, int, int):
        # return events.numpy()
        coordinates = events.coordinates()                  # [N, 2(x,y)], dtype: int16
        polarities = events.polarities()[:, np.newaxis]     # [N, 1], dtype: uint8
        timestamps = events.timestamps()[:, np.newaxis]     # [N, 1], dtype: int64

        events_ndarray = np.concatenate([coordinates, polarities, timestamps], axis=1) # [N, 4(x, y, p(0,1), t(microseconds))], dtype: int64
        start_time = events.getLowestTime()
        duration = events.getHighestTime() - start_time
        return events_ndarray, start_time, duration

    @staticmethod
    def store_to_tensor(events: dv.EventStore, batch_idx=None, device='cpu') -> (torch.Tensor, int, int):
        events_ndarray, start_time, duration = EventBuffer.store_to_ndarray(events)
        events_ndarray[:, -1] -= start_time

        events_tensor = torch.as_tensor(events_ndarray.astype(np.float32))
        events_tensor[:, -2] = events_tensor[:, -2] * 2 - 1     # [N, 4(x, y, p(-1,1), t)], dtype: torch.float32

        if batch_idx is None:
            return events_tensor.to(device), 0, duration
        else:
            batch_idx_tensor = torch.ones((events_tensor.shape[0], 1), dtype=events_tensor.dtype) * batch_idx
            events_tensor_with_batch_idx = torch.cat([batch_idx_tensor, events_tensor], dim=1)  # [N, 5(batch_idx, x, y, p(-1,1), t)], dtype: torch.float32
            return events_tensor_with_batch_idx.to(device), 0, duration





# class ImuBuffer:
#     def __init__(self, max_imu_vector_list_len):
#         self.imu_vector_list = []
#         self.max_imu_vector_list_len = max_imu_vector_list_len
#
#     def __len__(self):
#         return len(self.imu_vector_list)
#
#
#     def push(self, imu_vector: dv.IMUPacket.ImuVector):
#         self.imu_vector_list.append(imu_vector)
#         self.imu_vector_list[0].append(imu_vector[0])
#         if len(self.imu_vector_list) > self.max_imu_vector_list_len:
#             self.imu_vector_list.pop(0)
#
#     def get_between(self, start_time, end_time):
#         pass


class ImuBuffer:
    def __init__(self, max_imu_len: int):
        self.imu_vector = dv.IMUPacket.ImuVector()
        self.timestamp_list = []
        self.max_imu_len = max_imu_len

    def __len__(self):
        return len(self.imu_vector)


    def push(self, input_imu_vector: dv.IMUPacket.ImuVector):
        for imu in input_imu_vector:
            self.imu_vector.append(imu)
            self.timestamp_list.append(imu.timestamp)

        if len(self.imu_vector) > self.max_imu_len:
            self.imu_vector = self.imu_vector[-1-self.max_imu_len:-1]
            self.timestamp_list = self.timestamp_list[-1-self.max_imu_len:-1]

    def getLowestTime(self):
        return self.imu_vector[0].timestamp

    def getHighestTime(self):
        return self.imu_vector[-1].timestamp


    def get_between(self, start_time, end_time) -> dv.IMUPacket.ImuVector:
        timestamp_array = np.array(self.timestamp_list)
        start_idx, end_idx = np.searchsorted(timestamp_array, [start_time, end_time])
        return self.imu_vector[start_idx: end_idx]


    @staticmethod
    def vector_to_ndarray(imu_vector: dv.IMUPacket.ImuVector) -> (np.ndarray, int, int):
        def np_vector_to_ndarray_base_func(imu: dv.IMU):
            angular_velocity = imu.getAngularVelocities()
            linear_acceleration = imu.getAccelerations()
            array = np.concatenate([angular_velocity, linear_acceleration], axis=0)  # [6, ]
            return array

        start_time = imu_vector[0].timestamp
        duration = imu_vector[-1].timestamp - start_time

        imu_array_list = []
        for i in range(len(imu_vector)):
            imu_array_list.append(np_vector_to_ndarray_base_func(imu_vector[i]))
        imu_ndarray = np.stack(imu_array_list, axis=0)   # [N, 6]
        return imu_ndarray, start_time, duration


    ## To ros message
    @staticmethod
    def vector_to_msg(imu_vector: dv.IMUPacket.ImuVector) -> (list, int, int):
        def np_vector_to_msg_base_func(imu: dv.IMU):
            imu_data_i = IMU_ROS()
            imu_data_i.header.stamp = rospy.Time(nsecs=imu.timestamp * 1000)  # microseconds to nanosecond
            imu_data_i.header.frame_id = 'base_link'

            angular_velocity = imu.getAngularVelocities()
            linear_acceleration = imu.getAccelerations()
            imu_data_i.angular_velocity = geometry_msgs.msg.Vector3(x=angular_velocity[0],
                                                                    y=angular_velocity[1],
                                                                    z=angular_velocity[2])
            imu_data_i.linear_acceleration = geometry_msgs.msg.Vector3(x=linear_acceleration[0],
                                                                       y=linear_acceleration[1],
                                                                       z=linear_acceleration[2])
            return imu_data_i

        start_time = imu_vector[0].timestamp    # microseconds
        duration = imu_vector[-1].timestamp - start_time

        imu_msg_list = []
        for i in range(len(imu_vector)):
            imu_msg_list.append(np_vector_to_msg_base_func(imu_vector[i]))
        return imu_msg_list, start_time, duration





