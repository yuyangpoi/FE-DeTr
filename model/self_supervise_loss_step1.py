import torch
import torch.nn.functional as F
from model.warp_utils import get_image_transform_tensor, scale_homography, warp_perspective_tensor



def consistensy_loss(pred_heatmaps_batch,
                     rotation_vector, translation_vectors,
                     camera_nts, camera_depths,
                     camera_Ks, camera_Kinvs,
                     origin_sizes,
                     target_size,
                     N=30,
                     mask=None):
    '''

    :param pred_heatmaps_batch: [T=5, B, C=10, H, W]
    :param pose_batch: [T, B, C=10, ...]
    :param target_size: tuple(height, width)
    :param N: int
    :param mask:
    :return:
    '''
    def extract_patches(sal):
        unfold = torch.nn.Unfold(N, padding=0, stride=N // 2)
        patches = unfold(sal).transpose(1, 2)  # flatten
        patches = F.normalize(patches, p=2, dim=2)  # norm [B, num, N*N], num=(height//(N//2)-1)*(width//(N//2)-1)
        return patches
    T, B, C, H, W = pred_heatmaps_batch.shape
    assert rotation_vector.shape[:3] == (T, B, C)
    assert target_size[0] == H and target_size[1] == W
    assert H % (N//2) == 0 and W % (N//2) == 0

    # intervals = sorted([1, 10, 20])
    intervals = sorted([1])

    loss = 0
    for i in range(T*C-1):
        heatmap_0 = pred_heatmaps_batch[i//C, :, i%C].unsqueeze(1)      # [B, 1, H, W]
        # homo = homos_batch[(i+1)//C, :, (i+1)%C]  # [B, 3, 3]
        rotation_vector_0 = rotation_vector[i//C, :, i%C].float()
        translation_vectors_0 = translation_vectors[i//C, :, i%C].float()
        camera_nts_0 = camera_nts[i//C, :, i%C].float()
        camera_depths_0 = camera_depths[i//C, :, i%C].float()
        camera_Ks_0 = camera_Ks[i//C, :, i%C].float()
        camera_Kinvs_0 = camera_Kinvs[i//C, :, i%C].float()
        origin_sizes_0 = origin_sizes[i//C, :, i%C].float()
        target_sizes_0 = torch.tensor(target_size).unsqueeze(0).repeat(B,1).float().to(origin_sizes_0.device)

        loss_interval = 0
        loss_interval_cnt = 0
        for interval in intervals:
            # print('i: ', i)
            # print(i+interval)
            # print(i+interval <= T*C-1)
            if i+interval <= T*C-1:
                # heatmap_1 = pred_heatmaps_batch[(i + 1) // C, :, (i + 1) % C].unsqueeze(1)  # [B, 1, H, W]
                # rotation_vector_1 = rotation_vector[(i+1)//C, :, (i+1)%C].float()
                # translation_vectors_1 = translation_vectors[(i+1)//C, :, (i+1)%C].float()
                heatmap_1 = pred_heatmaps_batch[(i+interval) // C, :, (i+interval) % C].unsqueeze(1)  # [B, 1, H, W]
                rotation_vector_1 = rotation_vector[(i+interval)//C, :, (i+interval)%C].float()
                translation_vectors_1 = translation_vectors[(i+interval)//C, :, (i+interval)%C].float()

                homo_0_1 = get_image_transform_tensor(rotation_vector_0, translation_vectors_0, rotation_vector_1, translation_vectors_1,
                                                      camera_nts_0, camera_depths_0,
                                                      camera_Ks_0, camera_Kinvs_0)
                homo_0_1_scaled = scale_homography(homo_0_1, origin_sizes_0, target_sizes_0)

                heatmap_0_warpped = warp_perspective_tensor(heatmap_0, homo_0_1_scaled, (H, W))

                heatmap_1_patches = extract_patches(heatmap_1)
                heatmap_0_warpped_patches = extract_patches(heatmap_0_warpped)


                ## cos sim
                cosim = (heatmap_1_patches * heatmap_0_warpped_patches).sum(dim=2)
                loss_interval += (1 - cosim.mean())
                loss_interval_cnt += 1


        if loss_interval_cnt > 0:
            loss += (loss_interval / loss_interval_cnt)

    return loss / (T*C-1)




def peaky_loss(pred_heatmaps_batch, N=30, valid_mask=None):
    '''

    :param pred_heatmaps_batch: [T, B, C, H, W]
    :param N: int
    :param valid_mask: [T, B, 1, H, W]
    :return:
    '''
    neg_loss_alpha = 0.1

    assert N % 2 == 0, 'N must be even!'
    T = pred_heatmaps_batch.shape[0]

    loss_sum = 0
    for t in range(T):
        heatmap = pred_heatmaps_batch[t]
        processed_heatmap = F.avg_pool2d(heatmap, kernel_size=3, stride=1, padding=1)   # [B, C, H, W]
        max_heatmap = F.max_pool2d(processed_heatmap, kernel_size=N + 1, stride=1, padding=N // 2)
        mean_heatmap = F.avg_pool2d(processed_heatmap, kernel_size=N + 1, stride=1, padding=N // 2)

        if valid_mask is None:
            loss = 1 - (max_heatmap - mean_heatmap).mean()
            loss_sum += loss

        else:
            pos_mask = valid_mask[t]    # [B, 1, H, W]
            processed_pos_mask = F.max_pool2d(pos_mask, kernel_size=7, stride=1, padding=3)


            ## positive area loss
            pos_loss = 1 - (max_heatmap - mean_heatmap)
            pos_loss = torch.divide(torch.sum(pos_loss * processed_pos_mask, dim=(1, 2, 3)),
                                    torch.sum(processed_pos_mask + 1e-6, dim=(1, 2, 3)))
            pos_loss = torch.mean(pos_loss)


            ## negative area loss
            neg_mask = 1 - pos_mask    # [B, 1, H, W]
            neg_loss = mean_heatmap
            neg_loss = torch.divide(torch.sum(neg_loss * neg_mask, dim=(1, 2, 3)),
                                    torch.sum(neg_mask + 1e-6, dim=(1, 2, 3)))
            neg_loss = torch.mean(neg_loss)

            loss = pos_loss + neg_loss_alpha * neg_loss

            loss_sum += loss

    return loss_sum / T

