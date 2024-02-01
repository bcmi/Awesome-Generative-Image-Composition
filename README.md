# ![hot](https://bcmi.sjtu.edu.cn/~niuli/images/fire.png) Our lab has released the image composition toolbox [libcom](https://github.com/bcmi/libcom), which integrates all the functions related to image composition. The toolbox would be continuously upgraded. Welcome to visit and try :-)
---

# Awesome Generative Image Composition [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of resources including papers, datasets, and relevant links pertaining to generative image composition, which aims to generate plausible composite images based on a background image (optional bounding box) and a (*resp.*, a few) foreground image (*resp.*, images) of a specific object. 

## Contributing

Contributions are welcome.  If you wish to contribute, feel free to send a pull request. If you have suggestions for new sections to be included, please raise an issue and discuss before sending a pull request.

## Table of Contents
+ [Survey](#Survey)
+ [Papers](#Papers)
+ [Datasets](#Datasets)
+ [Other Resources](#Other-resources)

## Survey

A brief review on generative image composition is included in the following survey on image composition:

Li Niu, Wenyan Cong, Liu Liu, Yan Hong, Bo Zhang, Jing Liang, Liqing Zhang: "*Making Images Real Again: A Comprehensive Survey on Deep Image Composition.*" arXiv preprint arXiv:2106.14490 (2021). [[arXiv]](https://arxiv.org/pdf/2106.14490.pdf)

## Foreground: image;  Background: image

#### Object-to-Object
+ Vishnu Sarukkai, Linden Li, Arden Ma, Christopher Re, Kayvon Fatahalian: "*Collage Diffusion.*" WACV (2024) [[pdf]](https://openaccess.thecvf.com/content/WACV2024/papers/Sarukkai_Collage_Diffusion_WACV_2024_paper.pdf) [[code]](https://github.com/VSAnimator/collage-diffusion)
+ Ziyang Yuan, Mingdeng Cao, Xintao Wang, Zhongang Qi, Chun Yuan, Ying Shan: "*CustomNet: Zero-shot Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models.*" arXiv preprint arXiv:2310.19784 (2023) [[arXiv]](https://arxiv.org/pdf/2310.19784.pdf) [[code]](https://github.com/TencentARC/CustomNet)
+ Bo Zhang, Yuxuan Duan, Jun Lan, Yan Hong, Huijia Zhu, Weiqiang Wang, Li Niu: "*ControlCom: Controllable Image Composition using Diffusion Model.*" arXiv preprint arXiv:2308.10040 (2023) [[arXiv]](https://arxiv.org/pdf/2308.10040.pdf) [[code]](https://github.com/bcmi/ControlCom-Image-Composition)
+ Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, Hengshuang Zhao: "*AnyDoor: Zero-shot Object-level Image Customization.*" arXiv preprint arXiv:2307.09481 (2023) [[arXiv]](https://arxiv.org/pdf/2307.09481.pdf) [[code]](https://github.com/damo-vilab/AnyDoor)
+ Xin Zhang, Jiaxian Guo, Paul Yoo, Yutaka Matsuo, Yusuke Iwasawa: "*Paste, Inpaint and Harmonize via Denoising: Subject-Driven Image Editing with Pre-Trained Diffusion Model.*" arXiv preprint arXiv:2306.07596 (2023) [[arXiv]](https://arxiv.org/pdf/2306.07596.pdf) [[code]](https://sites.google.com/view/phd-demo-page)
+ Roy Hachnochi, Mingrui Zhao, Nadav Orzech, Rinon Gal, Ali Mahdavi-Amiri, Daniel Cohen-Or, Amit Haim Bermano: "*Cross-domain Compositing with Pretrained Diffusion Models.*" arXiv preprint arXiv:2302.10167 (2023) [[arXiv]](https://arxiv.org/pdf/2302.10167.pdf) [[code]](https://github.com/roy-hachnochi/cross-domain-compositing)
+ Shilin Lu, Yanzhu Liu, Adams Wai-Kin Kong: "*Tf-icon: Diffusion-based training-free cross-domain image composition.*" ICCV (2023) [[pdf]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_TF-ICON_Diffusion-Based_Training-Free_CrossDomain_Image_Composition_ICCV_2023_paper.pdf) [[code]](https://github.com/Shilin-LU/TF-ICON)
+ Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, Fang Wen: "*Paint by Example: Exemplar-based Image Editing with Diffusion Models.*" CVPR (2023) [[arXiv]](https://arxiv.org/pdf/2211.13227.pdf) [[code]](https://arxiv.org/pdf/2211.13227.pdf)
+ Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, Daniel Aliaga: "*ObjectStitch: Generative Object Compositing.*" CVPR (2023) [[arXiv]](https://arxiv.org/pdf/2212.00932.pdf)
+ Sumith Kulal, Tim Brooks, Alex Aiken, Jiajun Wu, Jimei Yang, Jingwan Lu, Alexei A. Efros, Krishna Kumar Singh: "*Putting People in Their Place: Affordance-Aware Human Insertion into Scenes.*" CVPR (2023) [[paper]](https://sumith1896.github.io/affordance-insertion/static/paper/affordance_insertion_cvpr2023.pdf) [[code]](https://github.com/adobe-research/affordance-insertion)



#### Token-to-Object

+ Lingxiao Lu, Bo Zhang, Li Niu: "*DreamCom: Finetuning Text-guided Inpainting Model for Image Composition.*" arXiv preprint arXiv:2309.15508 (2023) [[arXiv]](https://arxiv.org/pdf/2309.15508.pdf) [[code]](https://github.com/bcmi/DreamCom-Image-Composition)

+  Tianle Li, Max Ku, Cong Wei, Wenhu Chen: "*DreamEdit: Subject-driven Image Editing.*" arXiv preprint arXiv:2306.12624 (2023) [[arXiv]](https://arxiv.org/pdf/2306.12624.pdf) [[code]](https://github.com/DreamEditBenchTeam/DreamEdit)
 


## Foreground: 3D;  Background: image

+ Jinghao Zhou, Tomas Jakab, Philip Torr, Christian Rupprecht: "*Scene-Conditional 3D Object Stylization and Composition.*" arXiv preprint arXiv:2312.12419 (2023) [[arXiv]](https://arxiv.org/pdf/2312.12419.pdf) [[code]](https://github.com/shallowtoil/scene-cond-3d)

## Foreground: 3D;  Background: 3D
+ Rahul Goel, Dhawal Sirikonda, Saurabh Saini, PJ Narayanan: "*Interactive Segmentation of Radiance Fields.*" CVPR (2023) [[arXiv]](https://arxiv.org/abs/2212.13545) [[code]](https://github.com/rahul-goel/isrf_code)
+ Rahul Goel, Dhawal Sirikonda, Rajvi Shah, PJ Narayanan: "*FusedRF: Fusing Multiple Radiance Fields.*" CVPR Workshop (2023) [[arXiv]](https://arxiv.org/abs/2306.04180)
+ Verica Lazova, Vladimir Guzov, Kyle Olszewski, Sergey Tulyakov, Gerard Pons-Moll: "*Control-NeRF: Editable Feature Volumes for Scene Rendering and Manipulation.*" WACV (2023) [[arXiv]](https://arxiv.org/abs/2204.10850)
+ Jiaxiang Tang, Xiaokang Chen, Jingbo Wang, Gang Zeng: "*Compressible-composable nerf via rank-residual decomposition.*" NIPS (2022) [[arXiv]](https://arxiv.org/abs/2205.14870) [[code]](https://github.com/ashawkey/CCNeRF)
+ Bangbang Yang, Yinda Zhang, Yinghao Xu, Yijin Li, Han Zhou, Hujun Bao, Guofeng Zhang, Zhaopeng Cui: "*Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering.*" ICCV (2021) [[arXiv]](https://arxiv.org/abs/2109.01847) [[code]](https://github.com/zju3dv/object_nerf)

## Foreground: video;  Background: image

+ Boxiao Pan, Zhan Xu, Chun-Hao Paul Huang, Krishna Kumar Singh, Yang Zhou, Leonidas J. Guibas, Jimei Yang: "*ActAnywhere: Subject-Aware Video Background Generation.*" arXiv preprint arXiv:2401.10822 (2024) [[arXiv]](https://arxiv.org/pdf/2401.10822.pdf) 

## Foreground: video;  Background: video

+ Jiaqi Guo, Sitong Su, Junchen Zhu, Lianli Gao, Jingkuan Song: "*Training-Free Semantic Video Composition via Pre-trained Diffusion Model.*" arXiv preprint arXiv:2401.09195 (2024) [[arXiv]](https://arxiv.org/pdf/2401.09195v1.pdf)

+ Donghoon Lee, Tomas Pfister, Ming-Hsuan Yang: "*Inserting Videos into Videos.*" CVPR (2019) [[pdf]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Inserting_Videos_Into_Videos_CVPR_2019_paper.pdf)

## Other Resources

+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)

