# CCVS-3ASeg: A Large-Scale Contextual Auxiliary Framework for Road Crack Semantic Segmentation in UAV Imagery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22812170.svg)](https://doi.org/10.5281/zenodo.22812170)

> Authors: Chongchong Yao, Nu Wen, Zhimin Zhang, Yachao Chang, Yong Fan.
![替代文本](./paperGraph/figure4.png)
![替代文本](./paperGraph/figure3.png)
![替代文本](./paperGraph/figure5.png)

>Code and Data of Paper: A Stage-Focused Strategy for Real-Time Small Crack Segmentation Leveraging High-Resolution UAV Road Imagery.
We will continue to update the data and code corresponding to the paper.

# Hardware environment for this technical experiment: 
> (1) Personal Computer (RTX4060), Based on NVIDIA Ada Lovelace architecture (TSMC 4N process), it has 3,072 CUDA cores (Compute Capability 8.9, requiring CUDA 12.0 or higher) and 8GB GDDR6 video memory (272 GB/s bandwidth). With 15.11 TFLOPS FP32 throughput and 242 AI TOPS, it handles AI training and testing tasks. <br>
> (2) Cloud Server (RTX3090), Based on NVIDIA Ampere architecture (8nm process), it has 10,496 CUDA cores (Compute Capability 8.6, requiring CUDA 11.0 or higher), 24GB GDDR6X memory (936 GB/s bandwidth), and 35.5 TFLOPS FP32 throughput — ideal for mid-high AI tasks.<br>
> (3) Edge Device (NVIDIA Jetson Xavier developer kit), featuring a Volta GPU with 384 CUDA cores and 48 Tensor Cores (compute capability sm_72), delivering up to 21 TOPS (INT8) depending on the power mode, and equipped with up to 16 GB 128-bit LPDDR4x memory — suitable for lightweight and efficient edge AI deployments.<br>
> (4) Drone (DJI M300RTK, Camara: ZenmuseH20T), Its video imaging system includes a 20MP zoom camera (1/1.7" CMOS) supporting 4K (3840×2160) @30fps and 1080P@30fps, a 12MP wide-angle camera (1/2.3" CMOS) with 1080P@30fps, and a thermal camera (640×512@30Hz). All record in MP4 (H.264).<br>

# Software environment for this technical experiment: 
>The required environment is Python 3.11.11 (all packages in the code are installed based on Python 3.11.11).<br>
>Pretrained model checkpoints and representative generated samples are included in this repository.


# Data Release
>our Custom-Dataset data: https://pan.baidu.com/s/1LQK4diDD0AxcO7prMDsMFA?pwd=8888 <br>
>public Crack Dataset: https://pan.baidu.com/s/15YEN-U0xII5Asu2j506TIA?pwd=8888 <br>
>Updates will continue after the paper is published.


# Future Work
> Although this study has effectively optimized the utilization of high-resolution UAV data and the real-time performance of inference, there are still some limitations that can be explored in future research. Firstly, the current research mainly focuses on the segmentation of road crack small objects; future work can extend the VRS-3ASeg Framework to other object segmentation scenarios in UAV imagery, such as small obstacles, road signs, and vegetation gaps, to further verify the strategy's generalization ability. Secondly, the VRoIS module may face performance degradation in extreme complex backgrounds (e.g., severe road pollution, heavy occlusion by vegetation), and future research can optimize the distribution fitting function and RoI extraction algorithm to improve the module's robustness. Finally, the adversarial loss function in the 3ASeg module can be further improved by introducing multi-scale feature constraint terms, to enhance the module's ability to adapt to small objects of different sizes and improve the overall segmentation precision.

