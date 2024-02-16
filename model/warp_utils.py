import kornia
from kornia.geometry.transform.imgwarp import homography_warp
import torch
import torch.nn.functional as F


def generate_homography_tensor(rvec, tvec, nt, depth):
    """
    Generates a single homography

    Args:
        rvec (tensor): rotation vector        [B, 3]
        tvec (tensor): translation vector     [B, 3]
        nt (tensor): normal to camera         [B, 1, 3]
        depth (tensor): depth to camera          [B, 1]
    """
    batch_size = rvec.shape[0]
    R = torch.transpose(kornia.geometry.conversions.angle_axis_to_rotation_matrix(rvec), 1, 2)   # [B, 3, 3]
    # print(R.shape)
    # print(torch.matmul(tvec.reshape(batch_size, 3, 1), nt).shape)
    # print(depth.unsqueeze(-1).shape)
    # print((torch.div(torch.matmul(tvec.reshape(batch_size, 3, 1), nt).permute(1, 2, 0), depth.unsqueeze(-1).permute(1, 2, 0))).shape)
    H = R - torch.div(torch.matmul(tvec.reshape(batch_size, 3, 1), nt).permute(1, 2, 0), depth.unsqueeze(-1).permute(1, 2, 0)).permute(2, 0, 1)           # [B, 3, 3]
    return H



def generate_image_homography_tensor(rvec, tvec, nt, depth, K, Kinv):
    """
    Generates a single image homography

    Args:
        rvec (tensor): rotation vector          [B, 3]
        tvec (tensor): translation vector       [B, 3]
        nt (tensor): normal to camera           [B, 1, 3]
        depth (tensor): depth to camera         [B, 1]
        K (tensor): intrisic matrix             [B, 3, 3]
        Kinv (tensor): inverse intrinsic matrix [B, 3, 3]
    """
    H = generate_homography_tensor(rvec, tvec, nt, depth)   # [B, 3, 3]
    G = torch.matmul(K, torch.matmul(H, Kinv))  # [B, 3, 3]
    # G /= G[2, 2]
    G = torch.div(G.permute(1, 2, 0), G[:, 2, 2]).permute(2, 0, 1)     # [B, 3, 3]
    return G


def get_image_transform_tensor(rvec1, tvec1, rvec2, tvec2, nt, depth, K, Kinv):
    """
    Get image Homography between 2 poses (includes cam intrinsics)

    Args:
        rvec1 (tensor): rotation vector 1         [B, 3]
        tvec1 (tensor): translation vector 1      [B, 3]
        rvec2 (tensor): rotation vector 2         [B, 3]
        tvec2 (tensor): translation vector 2      [B, 3]
        nt (tensor): plane normal                 [B, 1, 3]
        depth (tensor): depth from camera         [B, 1]
        K (tensor): intrinsic                     [B, 3, 3]
        Kinv (tensor): inverse intrinsic        [B, 3, 3]
    """
    H_0_1 = generate_image_homography_tensor(rvec1, tvec1, nt, depth, K, Kinv)
    H_0_2 = generate_image_homography_tensor(rvec2, tvec2, nt, depth, K, Kinv)
    # H_1_2 = torch.matmul(H_0_2, torch.linalg.inv(H_0_1))  # [B, 3, 3]
    H_1_2 = torch.matmul(H_0_2.float(), torch.linalg.inv(H_0_1.float()))  # [B, 3, 3] TODO: test
    return H_1_2



def scale_homography(homo, origin_size, target_size):
    '''

    :param homo: [B, 3, 3]
    :param origin_size: [B, 2(height, width)]
    :param target_size: [B, 2(height, width)]
    :return:
    '''
    batch_size = homo.shape[0]
    homo_scaled_list = []
    for b in range(batch_size):
        origin_height, origin_width = origin_size[b]
        target_height, target_width = target_size[b]
        scaling_matrix = torch.tensor([[target_width/origin_width, 0, 0],
                                     [0, target_height/origin_height, 0],
                                     [0, 0, 1.]], device=homo.device)
        # print('\nscaling_matrix: \n', scaling_matrix)
        homo_scaled = torch.matmul(scaling_matrix, torch.matmul(homo[b], torch.linalg.inv(scaling_matrix)))
        homo_scaled_list.append(homo_scaled)
    homo_scaled_out = torch.stack(homo_scaled_list, dim=0)
    return homo_scaled_out




def warp_perspective_tensor(image_tensor, M_tensor, dsize,
                            mode='bilinear', padding_mode='zeros',
                            normalized_homography=False):
    '''

    :param image_tensor: [B, C, H, W]
    :param M_tensor: homography [B, 3, 3]
    :param dsize: (height, width)
    :param mode:
    :return:
        image_tensor_warpped: [B, C, H, W]
    '''
    return kornia.geometry.transform.warp_perspective(image_tensor, M_tensor, dsize, mode=mode, padding_mode=padding_mode)
    # return homography_warp(image_tensor, M_tensor, dsize,
    #                        mode=mode, padding_mode=padding_mode,
    #                        normalized_coordinates=False,
    #                        normalized_homography=False,
    #                        )




def warp_perspective_tensor_by_flow(image_tensor, flow_tensor, dsize,
                            mode='bilinear', padding_mode='zeros'):
    '''

    :param image_tensor: [B, C, H, W]
    :param flow_tensor: [B, 2, H, W]
    :param dsize:
    :param mode:
    :param padding_mode:
    :return:
    '''
    ## TODO
    pass



