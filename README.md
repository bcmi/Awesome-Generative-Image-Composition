
# Awesome Generative Image Composition [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of resources including papers, datasets, and relevant links pertaining to generative image composition, which aims to generate plausible composite images based on a background image (optional bounding box) and a (*resp.*, a few) foreground image (*resp.*, images) of a specific object. 

## Contributing

Contributions are welcome.  If you wish to contribute, feel free to send a pull request. If you have suggestions for new sections to be included, please raise an issue and discuss before sending a pull request.

## Table of Contents
+ [Survey](#Survey)
+ [Evaluation Metrics](#Evaluation-metrics)
+ [Test Set](#Test-set)
+ [Leaderboard](#Leaderboard)
+ [Papers](#Papers)
+ [Related Topics](#Related-topics)
+ [Other Resources](#Other-resources)

## Survey

A brief review on generative image composition is included in the following survey on image composition:

Li Niu, Wenyan Cong, Liu Liu, Yan Hong, Bo Zhang, Jing Liang, Liqing Zhang: "*Making Images Real Again: A Comprehensive Survey on Deep Image Composition.*" arXiv preprint arXiv:2106.14490 (2021). [[arXiv]](https://arxiv.org/pdf/2106.14490.pdf)

## Evaluation Metrics

+ [Composite-Image-Evaluation](https://github.com/bcmi/Composite-Image-Evaluation)

## Test Set

+ [COCOEE](https://github.com/Fantasy-Studio/Paint-by-Example?tab=readme-ov-file#test-benchmark) (within-domain, single-ref): 500 background images from MSCOCO validation set.  Each background image has a bounding box and a foreground image from MSCOCO training set.
+ [TF-ICON test benchmark](https://github.com/Shilin-LU/TF-ICON?tab=readme-ov-file#tf-icon-test-benchmark) (cross-domain, single-ref): 332 samples. Each sample consists of a background image, a foreground image, a
user mask, and a text prompt.
+ [FOSCom](https://github.com/bcmi/ControlCom-Image-Composition) (within-domain, single-ref): 640 background images from Internet. Each background image has a manually annotated bounding box and a foreground image from MSCOCO training set.
+ [DreamEditBench](https://huggingface.co/datasets/tianleliphoebe/DreamEditBench) (within-domain, multi-ref): 220 background images and 30 unique foreground objects from 15 categories. 
+ [MureCom](https://github.com/bcmi/DreamCom-Image-Composition?tab=readme-ov-file#our-murecom-dataset) (within-domain, multi-ref): 640 background images and 96 unique foreground objects from 32 categories.

## Leaderboard

The training set is open. The test set is [COCOEE](https://github.com/Fantasy-Studio/Paint-by-Example?tab=readme-ov-file#test-benchmark) benchmark. 
<table class="tg">
  <tr>
    <th class="tg-0pky" rowspan="2" align="center">Method</th>
    <th class="tg-0pky" colspan="3" align="center">Foreground</th>
    <th class="tg-0pky" colspan="2" align="center">Background</th>
    <th class="tg-0pky" colspan="2" align="center">Overall</th>
  </tr>
  <tr>
    <th class="tg-0pky" align="center">CLIP&uarr;</th>
    <th class="tg-0pky" align="center">DINO&uarr;</th>
    <th class="tg-0pky" align="center">FID&darr;</th>
    <th class="tg-0pky" align="center">LSSIM&uarr;</th>    
    <th class="tg-0pky" align="center">LPIPS&darr;</th>
    <th class="tg-0pky" align="center">FID&darr;</th>
    <th class="tg-0pky" align="center">QS&uarr;</th>
  </tr>
<tr>
  <th class="tg-0pky" align="center">Inpaint&Paste</th>
    <th class="tg-0pky" align="center">-</th>
  <th class="tg-0pky" align="center">-</th>
    <th class="tg-0pky" align="center">8.0</th>
    <th class="tg-0pky" align="center">-</th>    
    <th class="tg-0pky" align="center">-</th>
    <th class="tg-0pky" align="center">3.64</th>
    <th class="tg-0pky" align="center">72.07</th>
  </tr>
  <tr>
  <th class="tg-0pky" align="center"><a href="https://arxiv.org/pdf/2108.01073.pdf">SDEdit</a> </th>
    <th class="tg-0pky" align="center">85.02</th>
    <th class="tg-0pky" align="center">55.38</th>
    <th class="tg-0pky" align="center">9.77</th>
    <th class="tg-0pky" align="center">0.630</th>    
    <th class="tg-0pky" align="center">0.344</th>
    <th class="tg-0pky" align="center">6.42</th>
    <th class="tg-0pky" align="center">75.20</th>
  </tr>   
  <th class="tg-0pky" align="center"><a href="https://arxiv.org/pdf/2211.13227.pdf">PBE</a> </th>
    <th class="tg-0pky" align="center">84.84</th>
  <th class="tg-0pky" align="center">52.52</th>
    <th class="tg-0pky" align="center">6.24</th>
    <th class="tg-0pky" align="center">0.823</th>    
    <th class="tg-0pky" align="center">0.116</th>
    <th class="tg-0pky" align="center">3.18</th>
    <th class="tg-0pky" align="center">77.80</th>
  </tr>   
  <th class="tg-0pky" align="center"><a href="https://arxiv.org/pdf/2212.00932.pdf">ObjectStitch</a></th>
    <th class="tg-0pky" align="center">85.97</th>
    <th class="tg-0pky" align="center">61.12</th>
    <th class="tg-0pky" align="center">6.86</th>
    <th class="tg-0pky" align="center">0.825</th>    
    <th class="tg-0pky" align="center">0.116</th>
    <th class="tg-0pky" align="center">3.35</th>
    <th class="tg-0pky" align="center">76.86</th>
  </tr>  
  
  <th class="tg-0pky" align="center"><a href="https://arxiv.org/pdf/2307.09481.pdf">AnyDoor</a></th>
    <th class="tg-0pky" align="center">89.7</th>
    <th class="tg-0pky" align="center">70.16</th>
    <th class="tg-0pky" align="center">10.5</th>
    <th class="tg-0pky" align="center">0.870</th>    
    <th class="tg-0pky" align="center">0.109</th>
    <th class="tg-0pky" align="center">3.60</th>
    <th class="tg-0pky" align="center">76.18</th>
  </tr>
  <th class="tg-0pky" align="center"><a href="https://arxiv.org/pdf/2308.10040.pdf">ControlCom</a></th>
    <th class="tg-0pky" align="center">88.31</th>
    <th class="tg-0pky" align="center">63.67</th>
    <th class="tg-0pky" align="center">6.28</th>
    <th class="tg-0pky" align="center">0.826</th>    
    <th class="tg-0pky" align="center">0.114</th>
    <th class="tg-0pky" align="center">3.19</th>
    <th class="tg-0pky" align="center">77.84</th>
  </tr>
</table>

### Evaluating Your Results

1. **Install Dependencies**:
   - Begin by installing the dependencies listed in [requirements.txt](./requirements.txt).
   - Additionally, install [Segment Anything](https://github.com/facebookresearch/segment-anything).

2. **Clone Repository and Download Pretrained Models**:
   - Clone this repository and ensure you have a `checkpoints` folder.
   - Download the following pretrained models into the `checkpoints` folder:
     - [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32): Used for CLIP score and FID score calculations.
     - [ViT-H SAM model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints): Utilized to estimate foreground masks for reference images and generated composites.
     - [facebook/dino-vits16](https://huggingface.co/facebook/dino-vits16): Employed in DINO score computation.
     - [coco2017_gmm_k20](https://github.com/Fantasy-Studio/Paint-by-Example#qs-score): Utilized to compute the overall quality score.

   The resulting folder structure should resemble the following:
   ```shell
   checkpoints/
   ├── clip-vit-base-patch32
   ├── coco2017_gmm_k20
   ├── dino-vits16
   └── sam_vit_h_4b8939.pth
   ```

<!-- 3. **Download Cache File for FID Scores**:
   - Download the cache file from [Google Drive](https://drive.google.com/file/d/1m5EXLb2fX95uyl2dYtQUudjnFsGhN5dU/view?usp=sharing) used for computing FID scores.
   - Unzip the cache file to a `cache` folder as follows:
     ```shell
     cache/
     ├── coco2017_test.pth
     └── cocoee_gtfg.pth
     ```
   Alternatively, you can download the test set of [COCO2017](http://images.cocodataset.org/zips/test2017.zip) in advance and unzip it to a `data` folder. -->

3. **Prepare COCOEE Benchmark and Your Results**:
   - Prepare the [COCOEE benchmark](https://github.com/Fantasy-Studio/Paint-by-Example?tab=readme-ov-file#test-benchmark) alongside your generated composite results. Ensure that your composite images have filenames corresponding to the background images of the COCOEE dataset, as illustrated below:
      ```shell
      results/
      ......
      ├── 000002228519_GT.png
      ├── 000002231413_GT.png
      ├── 900100065455_GT.png
      └── 900100376112_GT.png
      ```
   - Modify the paths accordingly in the `run.sh` file. If you have downloaded the cache file mentioned earlier, please ignore `cocodir`.
   - Execute the following command:
     ```shell
     sh run.sh
     ```
   Then, wait for the results of all metrics to be computed.


## Papers

#### Object-to-Object
+ Vishnu Sarukkai, Linden Li, Arden Ma, Christopher Re, Kayvon Fatahalian: "*Collage Diffusion.*" WACV (2024) [[pdf]](https://openaccess.thecvf.com/content/WACV2024/papers/Sarukkai_Collage_Diffusion_WACV_2024_paper.pdf) [[code]](https://github.com/VSAnimator/collage-diffusion) 
+ Ziyang Yuan, Mingdeng Cao, Xintao Wang, Zhongang Qi, Chun Yuan, Ying Shan: "*CustomNet: Zero-shot Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models.*" arXiv preprint arXiv:2310.19784 (2023) [[arXiv]](https://arxiv.org/pdf/2310.19784.pdf) [[code]](https://github.com/TencentARC/CustomNet)
+ Bo Zhang, Yuxuan Duan, Jun Lan, Yan Hong, Huijia Zhu, Weiqiang Wang, Li Niu: "*ControlCom: Controllable Image Composition using Diffusion Model.*" arXiv preprint arXiv:2308.10040 (2023) [[arXiv]](https://arxiv.org/pdf/2308.10040.pdf) [[code]](https://github.com/bcmi/ControlCom-Image-Composition) [[demo]](https://bcmi.sjtu.edu.cn/home/niuli/demo_image_composition/)
+ Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, Hengshuang Zhao: "*AnyDoor: Zero-shot Object-level Image Customization.*" CVPR (2024) [[arXiv]](https://arxiv.org/pdf/2307.09481.pdf) [[code]](https://github.com/damo-vilab/AnyDoor) [[demo]](https://huggingface.co/spaces/xichenhku/AnyDoor-online)
+ Xin Zhang, Jiaxian Guo, Paul Yoo, Yutaka Matsuo, Yusuke Iwasawa: "*Paste, Inpaint and Harmonize via Denoising: Subject-Driven Image Editing with Pre-Trained Diffusion Model.*" arXiv preprint arXiv:2306.07596 (2023) [[arXiv]](https://arxiv.org/pdf/2306.07596.pdf) [[code]](https://sites.google.com/view/phd-demo-page)
+ Roy Hachnochi, Mingrui Zhao, Nadav Orzech, Rinon Gal, Ali Mahdavi-Amiri, Daniel Cohen-Or, Amit Haim Bermano: "*Cross-domain Compositing with Pretrained Diffusion Models.*" arXiv preprint arXiv:2302.10167 (2023) [[arXiv]](https://arxiv.org/pdf/2302.10167.pdf) [[code]](https://github.com/roy-hachnochi/cross-domain-compositing)
+ Shilin Lu, Yanzhu Liu, Adams Wai-Kin Kong: "*TF-ICON: Diffusion-based Training-free Cross-domain Image Composition.*" ICCV (2023) [[pdf]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_TF-ICON_Diffusion-Based_Training-Free_Cross-Domain_Image_Composition_ICCV_2023_paper.pdf) [[code]](https://github.com/Shilin-LU/TF-ICON)
+ Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, Fang Wen: "*Paint by Example: Exemplar-based Image Editing with Diffusion Models.*" CVPR (2023) [[arXiv]](https://arxiv.org/pdf/2211.13227.pdf) [[code]](https://arxiv.org/pdf/2211.13227.pdf) [[demo]](https://huggingface.co/spaces/Fantasy-Studio/Paint-by-Example)
+ Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, Daniel Aliaga: "*ObjectStitch: Generative Object Compositing.*" CVPR (2023) [[arXiv]](https://arxiv.org/pdf/2212.00932.pdf) [[code]](https://github.com/bcmi/ObjectStitch-Image-Composition)
+ Sumith Kulal, Tim Brooks, Alex Aiken, Jiajun Wu, Jimei Yang, Jingwan Lu, Alexei A. Efros, Krishna Kumar Singh: "*Putting People in Their Place: Affordance-Aware Human Insertion into Scenes.*" CVPR (2023) [[paper]](https://sumith1896.github.io/affordance-insertion/static/paper/affordance_insertion_cvpr2023.pdf) [[code]](https://github.com/adobe-research/affordance-insertion)



#### Token-to-Object

+ Lingxiao Lu, Bo Zhang, Li Niu: "*DreamCom: Finetuning Text-guided Inpainting Model for Image Composition.*" arXiv preprint arXiv:2309.15508 (2023) [[arXiv]](https://arxiv.org/pdf/2309.15508.pdf) [[code]](https://github.com/bcmi/DreamCom-Image-Composition)

+  Tianle Li, Max Ku, Cong Wei, Wenhu Chen: "*DreamEdit: Subject-driven Image Editing.*" TMLR (2023) [[arXiv]](https://arxiv.org/pdf/2306.12624.pdf) [[code]](https://github.com/DreamEditBenchTeam/DreamEdit)




## Related Topics

#### Foreground: 3D;  Background: image

+ Jinghao Zhou, Tomas Jakab, Philip Torr, Christian Rupprecht: "*Scene-Conditional 3D Object Stylization and Composition.*" arXiv preprint arXiv:2312.12419 (2023) [[arXiv]](https://arxiv.org/pdf/2312.12419.pdf) [[code]](https://github.com/shallowtoil/scene-cond-3d)

#### Foreground: 3D;  Background: 3D
+ Mohamad Shahbazi, Liesbeth Claessens, Michael Niemeyer, Edo Collins, Alessio Tonioni, Luc Van Gool, Federico Tombari: "*InseRF: Text-Driven Generative Object Insertion in Neural 3D Scenes.*" arXiv preprint arXiv:2401.05335 (2024) [[arXiv]](https://arxiv.org/pdf/2401.05335.pdf)
+ Rahul Goel, Dhawal Sirikonda, Saurabh Saini, PJ Narayanan: "*Interactive Segmentation of Radiance Fields.*" CVPR (2023) [[arXiv]](https://arxiv.org/abs/2212.13545) [[code]](https://github.com/rahul-goel/isrf_code)
+ Rahul Goel, Dhawal Sirikonda, Rajvi Shah, PJ Narayanan: "*FusedRF: Fusing Multiple Radiance Fields.*" CVPR Workshop (2023) [[arXiv]](https://arxiv.org/abs/2306.04180)
+ Verica Lazova, Vladimir Guzov, Kyle Olszewski, Sergey Tulyakov, Gerard Pons-Moll: "*Control-NeRF: Editable Feature Volumes for Scene Rendering and Manipulation.*" WACV (2023) [[arXiv]](https://arxiv.org/abs/2204.10850)
+ Jiaxiang Tang, Xiaokang Chen, Jingbo Wang, Gang Zeng: "*Compressible-composable NeRF via Rank-residual Decomposition.*" NIPS (2022) [[arXiv]](https://arxiv.org/abs/2205.14870) [[code]](https://github.com/ashawkey/CCNeRF)
+ Bangbang Yang, Yinda Zhang, Yinghao Xu, Yijin Li, Han Zhou, Hujun Bao, Guofeng Zhang, Zhaopeng Cui: "*Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering.*" ICCV (2021) [[arXiv]](https://arxiv.org/abs/2109.01847) [[code]](https://github.com/zju3dv/object_nerf)

#### Foreground: video;  Background: image

+ Boxiao Pan, Zhan Xu, Chun-Hao Paul Huang, Krishna Kumar Singh, Yang Zhou, Leonidas J. Guibas, Jimei Yang: "*ActAnywhere: Subject-Aware Video Background Generation.*" arXiv preprint arXiv:2401.10822 (2024) [[arXiv]](https://arxiv.org/pdf/2401.10822.pdf) 

#### Foreground: video;  Background: video

+ Jiaqi Guo, Sitong Su, Junchen Zhu, Lianli Gao, Jingkuan Song: "*Training-Free Semantic Video Composition via Pre-trained Diffusion Model.*" arXiv preprint arXiv:2401.09195 (2024) [[arXiv]](https://arxiv.org/pdf/2401.09195v1.pdf)

+ Donghoon Lee, Tomas Pfister, Ming-Hsuan Yang: "*Inserting Videos into Videos.*" CVPR (2019) [[pdf]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Inserting_Videos_Into_Videos_CVPR_2019_paper.pdf)

## Other Resources

+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)

