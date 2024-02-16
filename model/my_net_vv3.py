
import torch
import torch.nn as nn
from torchvision.models import resnet

from mmcv.ops import DeformConv2d
from model.modules.recurrent_module import ConvRNN
from model.modules.cmt_module import CMTB



class CoarseFeatureExtractor(nn.Module):
    def __init__(self, input_channels, middle_channles, output_channels, bias=True):
        super().__init__()
        assert output_channels % 4 == 0

        self.input_layer = nn.Sequential(
            ## bias for zero input
            nn.Conv2d(input_channels, middle_channles, kernel_size=7, stride=1, padding=3, bias=bias),
        )
        self.extractor = nn.Sequential(
            resnet.Bottleneck(middle_channles, output_channels // 4, stride=1,
                              downsample=nn.Conv2d(middle_channles, output_channels, kernel_size=1, stride=1, padding=0)),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.extractor(x)
        return x



class ModalityFusionFeatureExtractor(nn.Module):
    def __init__(self, frame_coarse_feature_channels, event_coarse_feature_channels, outut_channels):
        super().__init__()
        self.frame_c = 32
        self.event_c = 32

        self.frame_CBR = nn.Sequential(
            nn.Conv2d(frame_coarse_feature_channels, self.frame_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.frame_c),
            nn.ReLU(),
        )
        self.event_CBR = nn.Sequential(
            nn.Conv2d(event_coarse_feature_channels, self.event_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.event_c),
            nn.ReLU(),
        )

        self.K_f_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(frame_coarse_feature_channels, self.event_c, kernel_size=3, stride=1, padding=1)
        )
        self.K_e_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(event_coarse_feature_channels, self.frame_c, kernel_size=3, stride=1, padding=1)
        )

        self.f_e_conv = nn.Conv2d(self.frame_c, self.frame_c, kernel_size=1, stride=1, padding=0)
        self.e_f_conv = nn.Conv2d(self.event_c, self.event_c, kernel_size=1, stride=1, padding=0)

        self.cmt_block = nn.Sequential(
            nn.Conv2d(self.frame_c+self.event_c, outut_channels, kernel_size=3, stride=1, padding=1),
            CMTB(dim=outut_channels, num_heads=8, sr_ratio=16, depth=1, pos_encoding=True),
        )

        self.output_BCR = nn.Sequential(
            nn.Conv2d(outut_channels, outut_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outut_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, frame_coarse_feature, event_coarse_feature):
        frame_feature = self.frame_CBR(frame_coarse_feature)        # [B, 32, H, W]
        event_feature = self.event_CBR(event_coarse_feature)        # [B, 72, H, W]

        K_f = self.K_f_generator(frame_coarse_feature)              # [B, 72, 1, 1]
        K_e = self.K_e_generator(event_coarse_feature)              # [B, 32, 1, 1]

        feature_f_e = frame_feature + frame_feature * K_e           # [B, 32, H, W], shortcut
        feature_e_f = event_feature + event_feature * K_f           # [B, 72, H, W], shortcut

        feature_f_e = self.f_e_conv(feature_f_e)                    # [B, 32, H, W]
        feature_e_f = self.e_f_conv(feature_e_f)                    # [B, 72, H, W]

        concat_feature = torch.cat([feature_f_e, feature_e_f], dim=1)   # [B, 32+72, H, W]

        fused_feature = self.cmt_block(concat_feature)              # [B, 32+72, H, W]

        fused_feature = self.output_BCR(fused_feature)              # [B, output_channels, H, W]

        return fused_feature



class Recurrent_layer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.recurrent_module = ConvRNN(input_channels, output_channels, cell='lstm')

    def forward(self, x):
        x = self.recurrent_module(x)
        return x

    def reset(self, mask):
        self.recurrent_module.reset(mask)

    def detach(self):
        self.recurrent_module.detach()





class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] -> [B, C, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MotionAwareSpatialChannelAttension(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MotionAwareSpatialChannelAttension, self).__init__()
        ## For spatial_pool
        self.conv_mask = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

        ## For attention_weight
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LayerNorm([input_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )

        ## For output channels adjust
        self.output_layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(output_channels),
            # nn.ReLU(inplace=True),
        )


    def spatial_pool(self, depth_feature):
        batch, channel, height, width = depth_feature.shape
        input_x = depth_feature # [N, C, H, W]
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(depth_feature)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        # context attention
        context = torch.matmul(input_x, context_mask)   # [N, 1, C, H*W] * [N, 1, H*W, 1] -> # [N, 1, C, 1]
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        attention_weight = self.spatial_pool(x)
        attention_weight = torch.sigmoid(self.channel_mul_conv(attention_weight))

        motion_feature = x * attention_weight
        motion_feature = self.output_layer(motion_feature)
        return motion_feature




class GroupBottleNeck(nn.Module):
    def __init__(self, input_channels, middle_channels, output_channels, groups):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, middle_channels, kernel_size=1, stride=1,
                               groups=groups)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels,
                               kernel_size=3, padding=1, stride=1,
                               groups=groups)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, output_channels, kernel_size=1, stride=1,
                               groups=groups)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


class MotionAwareHead(nn.Module):
    def __init__(self, fused_feature_channels, event_coarse_feature_channels, output_channels=10):
        super().__init__()
        self.output_channels = output_channels
        self.static_channels = 1
        self.dynamic_channels = output_channels - self.static_channels  # 9

        self.motion_c = 7
        self.base = 14
        assert (output_channels * self.base) % 2 == 0

        self.recurrent_layer = Recurrent_layer(fused_feature_channels, fused_feature_channels)

        ## static part
        self.static_conv = nn.Sequential(
            nn.Conv2d(fused_feature_channels, self.base, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.base),
            nn.ReLU(inplace=True),
        )

        ## dynamic part
        self.motion_inform_c = self.motion_c * self.dynamic_channels
        self.motion_extractor = MotionAwareSpatialChannelAttension(event_coarse_feature_channels,
                                                            self.dynamic_channels*self.motion_c)

        self.deformable_conv_offset_predictor_list = nn.ModuleList()
        self.deformable_conv_list = nn.ModuleList()
        self.deformable_BR_list = nn.ModuleList()
        for i in range(self.dynamic_channels):
            self.deformable_conv_offset_predictor_list.append(
                nn.Sequential(
                    nn.Conv2d(self.motion_c, 3*3*2, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(3*3*2, 3*3*2, kernel_size=1, stride=1, padding=0),
                )
            )
            self.deformable_conv_list.append(DeformConv2d(self.base, self.base, kernel_size=3, stride=1, padding=1))
            self.deformable_BR_list.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.base),
                    nn.ReLU(inplace=True),
                )
            )



        self.output_atten = SELayer(output_channels * self.base, reduction=self.base)

        ## 直接实现分组卷积
        groups = output_channels
        self.output_layer = nn.Sequential(
            GroupBottleNeck(output_channels * self.base,
                            output_channels * self.base // 2,
                            output_channels * self.base,
                            groups=groups),
            nn.Conv2d(output_channels * self.base, output_channels * self.base, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels * self.base, output_channels, kernel_size=1, stride=1, padding=0,
                      groups=groups),


            # nn.Sigmoid()  # TODO
        )


    def forward(self, fused_feature, event_coarse_feature):
        fused_feature = fused_feature + self.recurrent_layer(fused_feature)

        static_feature = self.static_conv(fused_feature)

        motion_inform = self.motion_extractor(event_coarse_feature)

        dynamic_feature_list = []
        dynamic_feature_i = static_feature
        for i in range(self.dynamic_channels):
            motion_inform_i = motion_inform[:, self.motion_inform_c-(i+1)*self.motion_c: self.motion_inform_c-i*self.motion_c]
            offset_i = self.deformable_conv_offset_predictor_list[i](motion_inform_i)
            # print('offset_i: ', offset_i)
            ## iterating on dynamic features
            dynamic_feature_i = self.deformable_conv_list[i](torch.cat([dynamic_feature_i], dim=1), offset_i)
            dynamic_feature_i = self.deformable_BR_list[i](dynamic_feature_i)
            dynamic_feature_list.append(dynamic_feature_i)

        static_dynamic_feature_list = [static_feature] + dynamic_feature_list
        static_dynamic_feature = torch.cat(static_dynamic_feature_list[::-1], dim=1)   # reverse list

        atten_static_dynamic_feature = self.output_atten(static_dynamic_feature)

        heatmaps = self.output_layer(atten_static_dynamic_feature)

        return heatmaps


    def reset(self, mask):
        self.recurrent_layer.reset(mask)

    def detach(self):
        self.recurrent_layer.detach()



class FrameEventNet(nn.Module):
    def __init__(self, frame_cin=1, event_cin=10, cout=10):
        super().__init__()
        self.zero_motion_cout = 1
        self.non_zero_motion_cout = cout - self.zero_motion_cout
        self.one_event_coarse_feature_channels = 4

        self.frame_coarse_feature_extractor = CoarseFeatureExtractor(frame_cin, 8, 8, True)
        self.event_coarse_feature_extractor = CoarseFeatureExtractor(event_cin, event_cin,
                                                                     self.non_zero_motion_cout*self.one_event_coarse_feature_channels,
                                                                     True)

        self.backbone = ModalityFusionFeatureExtractor(8,
                                                       self.non_zero_motion_cout*self.one_event_coarse_feature_channels,
                                                       16)

        self.head = MotionAwareHead(16, self.non_zero_motion_cout*self.one_event_coarse_feature_channels, cout)


    def forward(self, frame, event_representation):
        frame_coarse_feature = self.frame_coarse_feature_extractor(frame)  # [B, 16, H, W]
        event_coarse_feature = self.event_coarse_feature_extractor(event_representation)  # [B, (10-1)*4, H, W]

        fused_feature = self.backbone(frame_coarse_feature, event_coarse_feature)           # [B, 16, H, W]

        heatmaps = self.head(fused_feature, event_coarse_feature)                           # [B, 10, H, W]
        return heatmaps

    def reset(self, mask):
        self.head.reset(mask)

    def detach(self):
        self.head.detach()



if __name__ == '__main__':
    batch_size = 1
    height, width = 240, 320

    model = FrameEventNet(1, 10, 10).cuda()
    frame = torch.rand((batch_size, 1, height, width)).cuda()
    event = torch.rand((batch_size, 10, height, width)).cuda()
    output = model(frame, event)    # warm up
    print('output.shape: ', output.shape)

    import time
    with torch.no_grad():
        start_time = time.time()
        for i in range(200):
            output = model(frame, event)
            torch.cuda.synchronize()
        end_time = time.time()
        print((end_time - start_time)/200)

