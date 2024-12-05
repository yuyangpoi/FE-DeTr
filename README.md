# FE-DeTr: Keypoint Detection and Tracking in Low-quality Image Frames with Events

This is the Pytorch implementation of the ICRA 2024 paper [FE-DeTr: Keypoint Detection and Tracking in Low-quality Image Frames with Events](https://arxiv.org/abs/2403.11662). 

```bibtex
@inproceedings{wang2024fedetr,
    title={{FE-DeTr}: Keypoint Detection and Tracking in Low-quality Image Frames with Events}, 
    author={Xiangyuan Wang and Kuangyi Chen and Wen Yang and Lei Yu and Yannan Xing and Huai Yu},
    booktitle={IEEE International Conference on Robotics and Automation},
    year={2024},
    pages={14638--14644}
}
```


# Update
⭐ **[Extreme Corners Dataset](https://github.com/yuyangpoi/FF-KDT)** 

⭐ A better method that includes a **keypoint detector** and an **anypoint tracker**, both supporting the high temporal resolution: [Towards Robust Keypoint Detection and Tracking: A Fusion Approach with Event-Aligned Image Features](https://github.com/yuyangpoi/FF-KDT). 


# Introduction
FE-DeTr includes a novel keypoint detection network that fuses the textural and structural information from image frames with the high-temporal-resolution motion information from event streams. The network leverages a temporal response consistency for supervision, ensuring stable and efficient keypoint detection. Moreover, we use a spatio-temporal nearest-neighbor search strategy for robust keypoint tracking. 

<p align="center">
  <img src="figures/brief.png" width="80%">
</p>


# Network Architecture
<p align="center">
  <img src="figures/structure.png" width="90%">
</p>
