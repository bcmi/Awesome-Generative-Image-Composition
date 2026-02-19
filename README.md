
# Awesome Generative Image Composition [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of resources including papers, datasets, and relevant links pertaining to generative image composition (object insertion). **Generative image composition aims to generate plausible composite images based on a background image (optional bounding box) and a (*resp.*, a few) foreground image (*resp.*, images) of a specific object.** For more complete resources on general image composition ([object insertion](https://github.com/bcmi/Awesome-Object-Insertion)), please refer to [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion).

<p align='center'>  
  <img src='./figures/task.jpg'  width=90% />
</p>

## Contributing

Contributions are welcome.  If you wish to contribute, feel free to send a pull request. If you have suggestions for new sections to be included, please raise an issue and discuss before sending a pull request.

## Table of Contents
+ [Survey](#Survey)
+ [Online Demo](#Online-demo)
+ [Evaluation Metrics](#Evaluation-metrics)
+ [Datasets](#Datasets)
+ [Papers](#Papers)
+ [Other Resources](#Other-resources)

## Survey

A brief review on generative image composition is included in the following survey on image composition:

Li Niu, Wenyan Cong, Liu Liu, Yan Hong, Bo Zhang, Jing Liang, Liqing Zhang: "*Making Images Real Again: A Comprehensive Survey on Deep Image Composition.*" arXiv preprint arXiv:2106.14490 (2021). [[arxiv]](https://arxiv.org/pdf/2106.14490.pdf)  [[slides]](https://www.ustcnewly.com/download/Image_composition_tutorial.pdf)

## Online Demo

Try this [online demo](http://libcom.ustcnewly.com/) for image composition and have fun!

[![]](https://github.com/user-attachments/assets/87416ec5-2461-42cb-9f2d-5030b1e1b5ec)

## Evaluation Metrics

+ [Composite-Image-Evaluation](https://github.com/bcmi/Composite-Image-Evaluation)


## Datasets

+ [COCOEE](https://github.com/Fantasy-Studio/Paint-by-Example?tab=readme-ov-file#test-benchmark) (within-domain, single-ref): 500 background images from MSCOCO validation set.  Each background image has a bounding box and a foreground image from MSCOCO training set.
+ [TF-ICON test benchmark](https://github.com/Shilin-LU/TF-ICON?tab=readme-ov-file#tf-icon-test-benchmark) (cross-domain, single-ref): 332 samples. Each sample consists of a background image, a foreground image, a
user mask, and a text prompt.
+ [DreamEditBench](https://huggingface.co/datasets/tianleliphoebe/DreamEditBench) (within-domain, multi-ref): 220 background images and 30 unique foreground objects from 15 categories. 
+ [MureCom](https://github.com/bcmi/DreamCom-Image-Composition?tab=readme-ov-file#our-murecom-dataset) (within-domain, multi-ref): 640 background images and 96 unique foreground objects from 32 categories.
+ [SAM-FB](https://github.com/KaKituken/affordance-aware-any) (within-domain, single-ref): built upon SA-1B (SAM dataset). 3,160,403 images with 3,439 foreground categories.
+ [Subjects 200K](https://github.com/Yuanshi9815/Subjects200K) (within-domain, double-ref): 200,000 paired images. Each pair has the same subject yet various scene contexts.
+ [ORIDa](https://hello-jinwoo.github.io/orida/) (within-domain, multi-ref): 200 unique foreground objects. Each object is placed in an average of 50 diverse scenes. In each scene, one object is placed at 1~4 different positions.
+ [AnyInsertion](https://huggingface.co/datasets/WensongSong/AnyInsertion_V1) (within-domain, single-ref): The training set includes 136,385 samples across two prompt types: 58,188 mask-prompt image pairs and 78,197 text-prompt image pairs;the test set includes 158 data pairs: 120 mask-prompt pairs and 38 text-prompt pairs.


## Papers

### Training-free
+ Shilin Lu, Zhuming Lian, Zihan Zhou, Shaocong Zhang, Chen Zhao, Adams Wai-Kin Kong: "*Does FLUX Already Know How to Perform Physically Plausible Image Composition?*" ICLR (2026) [[arxiv]](https://arxiv.org/pdf/2509.21278)
+ Haowen Li, Zhenfeng Fan, Zhang Wen, Zhengzhou Zhu, Yunjin Li: "*AIComposer: Any Style and Content Image Composition via Feature Integration.*" (**+text**)  ICCV (2025) [[arxiv]](https://arxiv.org/pdf/2507.20721) [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_AIComposer_Any_Style_and_Content_Image_Composition_via_Feature_Integration_ICCV_2025_paper.pdf) [[code]](https://github.com/sherlhw/AIComposer)
+ Pengzhi Li, Qiang Nie, Ying Chen, Xi Jiang, Kai Wu, Yuhuan Lin, Yong Liu, Jinlong Peng, Chengjie Wang, Feng Zheng: "*Tuning-Free Image Customization with Image and Text Guidance.*" (**+text**) ECCV (2024) [[arxiv]](https://arxiv.org/pdf/2403.12658) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09769.pdf) [[code]](https://github.com/zrealli/TIGIC)
+ Yibin Wang, Weizhong Zhang, Jianwei Zheng, Cheng Jin: "*PrimeComposer: Faster Progressively Combined Diffusion for Image Composition with Attention Steering.*" (**+text**) ACM MM (2024) [[arxiv]](https://arxiv.org/pdf/2403.05053) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3664647.3680848) [[code]](https://github.com/CodeGoat24/PrimeComposer)
+ Shilin Lu, Yanzhu Liu, Adams Wai-Kin Kong: "*TF-ICON: Diffusion-based Training-free Cross-domain Image Composition.*" (**+text**) ICCV (2023) [[arxiv]](https://arxiv.org/pdf/2403.05053) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_TF-ICON_Diffusion-Based_Training-Free_Cross-Domain_Image_Composition_ICCV_2023_paper.pdf) [[code]](https://github.com/Shilin-LU/TF-ICON)
+ Roy Hachnochi, Mingrui Zhao, Nadav Orzech, Rinon Gal, Ali Mahdavi-Amiri, Daniel Cohen-Or, Amit Haim Bermano: "*Cross-domain Compositing with Pretrained Diffusion Models.*" arXiv preprint arXiv:2302.10167 (2023) [[arxiv]](https://arxiv.org/pdf/2302.10167.pdf) [[code]](https://github.com/roy-hachnochi/cross-domain-compositing)

### Training-based

####  Free from object-centric finetuning
+ Wensong Song, Hong Jiang, Zongxing Yang, Ruijie Quan, Yi Yang: "*Insert Anything: Image Insertion via In-Context Editing in DiT.*" AAAI (2026) [[arxiv]](https://arxiv.org/pdf/2504.15009) [[code]](https://github.com/song-wensong/insert-anything)
+ Dong Liang, Jinyuan Jia, Yuhao Liu, Rynson W.H. Lau: "*HOComp: Interaction-Aware Human-Object Composition.*" NeurIPS (2025)  [[arxiv]](https://arxiv.org/pdf/2507.16813) [[paper]](https://openreview.net/pdf/c6b791fd625ad535761f9071136be59309a1bde2.pdf) [[code]](https://github.com/dliang293/HOComp)
+ Qi Zhang, Guanyu Xing, Mengting Luo, Jianwei Zhang, Yanli Liu: "*Inserting Objects into Any Background Images via Implicit Parametric Representation.*" IEEE Transactions on Visualization and Computer Graphics (2025) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10753453)
+ Lu Yang, Yuanhao Wang, Yicheng Liu, Enze Wang, Ziyang Zhao, Yanqi He, Zexian Song, Hao Lua: "*UNICOM: Unified, foreground-aware, and context-realistic deep image composition with diffusion model.*" Neurocomputing (2025) [[paper]](https://www.sciencedirect.com/science/article/pii/S0925231225018016)
+ Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, Hengshuang Zhao: "*AnyDoor: Zero-shot Image Customization with Region-to-region Reference.*"  T-PAMI (2025) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10976616)
+ Jinwoo Kim, Sangmin Han, Jinho Jeong, Jiwoo Choi, Dongyeong Kim, Seon Joo Kim: "*ORIDa: Object-centric Real-world Image Composition Dataset.*" CVPR (2025) [[arxiv]](https://arxiv.org/pdf/2506.08964) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_ORIDa_Object-centric_Real-world_Image_Composition_Dataset_CVPR_2025_paper.pdf) 
+ Gemma Canet Tarrés, Zhe Lin, Zhifei Zhang, He Zhang, Andrew Gilbert, John Collomosse, Soo Ye Kim: "*Multitwine: Multi-Object Compositing with Text and Layout Control.*" (**+text**) CVPR (2025)  [[arxiv]](https://arxiv.org/pdf/2502.05165) [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Tarres_Multitwine_Multi-Object_Compositing_with_Text_and_Layout_Control_CVPR_2025_paper.pdf)
+ Junjia Huang, Pengxiang Yan, Jiyang Liu, Jie Wu, Zhao Wang, Yitong Wang, Liang Lin, Guanbin Li: "*DreamFuse: Adaptive Image Fusion with Diffusion Transformer.*" ICCV (2025) (**+text**) [[arxiv]](https://arxiv.org/pdf/2504.08291) [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_DreamFuse_Adaptive_Image_Fusion_with_Diffusion_Transformer_ICCV_2025_paper.pdf) [[code]](https://github.com/LL3RD/DreamFuse-Code) 
+ Haoxuan Wang, Jinlong Peng, Qingdong He, Hao Yang, Ying Jin, Jiafu Wu, Xiaobin Hu, Yanjie Pan, Zhenye Gan, Mingmin Chi, Bo Peng, Yabiao Wang: "*UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer.*" ICCV (2025) [[arxiv]](https://arxiv.org/pdf/2503.09277) [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_UniCombine_Unified_Multi-Conditional_Combination_with_Diffusion_Transformer_ICCV_2025_paper.pdf) [[code]](https://github.com/Xuan-World/UniCombine)
+ Daniel Winter, Asaf Shul, Matan Cohen, Dana Berman, Yael Pritch, Alex Rav-Acha, Yedid Hoshen: "*ObjectMate: A Recurrence Prior for Object Insertion and Subject-Driven Generation.*" ICCV (2025) [[arxiv]](https://arxiv.org/pdf/2412.08645) [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Winter_ObjectMate_A_Recurrence_Prior_for_Object_Insertion_and_Subject-Driven_Generation_ICCV_2025_paper.pdf)
+ Yongsheng Yu, Ziyun Zeng, Haitian Zheng, Jiebo Luo: "*OmniPaint: Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting.*" ICCV (2025) [[arxiv]](https://arxiv.org/pdf/2503.08677) [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_OmniPaint_Mastering_Object-Oriented_Editing_via_Disentangled_Insertion-Removal_Inpainting_ICCV_2025_paper.pdf) [[code]](https://github.com/yeates/OmniPaint)
+ Zitian Zhang, Frederic Fortier-Chouinard, Mathieu Garon, Anand Bhattad, Jean-Francois Lalonde: "*ZeroComp: Zero-shot Object Compositing from Image Intrinsics via Diffusion.*" WACV (2025) [[arxiv]](https://arxiv.org/pdf/2410.08168) [[paper]](https://openaccess.thecvf.com/content/WACV2025/papers/Zhang_ZeroComp_Zero-Shot_Object_Compositing_from_Image_Intrinsics_via_Diffusion_WACV_2025_paper.pdf) [[code]](https://github.com/lvsn/ZeroComp)
+ Jixuan He, Wanhua Li, Ye Liu, Junsik Kim, Donglai Wei, Hanspeter Pfister: "*Affordance-Aware Object Insertion via Mask-Aware Dual Diffusion.*" arXiv preprint arXiv:2412.14462 (2024)  [[arxiv]](https://arxiv.org/pdf/2412.14462) [[code]](https://github.com/KaKituken/affordance-aware-any)
+ Weijing Tao, Xiaofeng Yang, Biwen Lei, Miaomiao Cui, Xuansong Xie, Guosheng Lin: "*MotionCom: Automatic and Motion-Aware Image Composition with LLM and Video Diffusion Prior.*" arXiv preprint arXiv:2409.10090 (2024) [[arxiv]](https://arxiv.org/pdf/2409.10090.pdf) [[code]](https://github.com/weijing-tao/MotionCom)
+ Daniel Winter, Matan Cohen, Shlomi Fruchter, Yael Pritch, Alex Rav-Acha, Yedid Hoshen: "*ObjectDrop: Bootstrapping Counterfactuals for Photorealistic Object Removal and Insertion.*"  ECCV (2024) [[arxiv]](https://arxiv.org/pdf/2403.18818) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09857.pdf)
+ Gemma Canet Tarrés, Zhe Lin, Zhifei Zhang, Jianming Zhang, Yizhi Song, Dan Ruta, Andrew Gilbert, John Collomosse, Soo Ye Kim："*Thinking Outside the BBox: Unconstrained Generative Object Compositing.*" ECCV (2024) [[arxiv]](https://arxiv.org/pdf/2409.04559) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07965.pdf)
+ Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, He Zhang, Wei Xiong, Daniel Aliaga: "*IMPRINT: Generative Object Compositing by Learning Identity-Preserving Representation.*" CVPR (2024) [[arxiv]](https://arxiv.org/pdf/2403.10701) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Song_IMPRINT_Generative_Object_Compositing_by_Learning_Identity-Preserving_Representation_CVPR_2024_paper.pdf)
+ Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, Hengshuang Zhao: "*AnyDoor: Zero-shot Object-level Image Customization.*" CVPR (2024) [[arxiv]](https://arxiv.org/pdf/2307.09481) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_AnyDoor_Zero-shot_Object-level_Image_Customization_CVPR_2024_paper.pdf) [[code]](https://github.com/damo-vilab/AnyDoor) 
+ Vishnu Sarukkai, Linden Li, Arden Ma, Christopher Re, Kayvon Fatahalian: "*Collage Diffusion.*" WACV (2024) [[arxiv]](https://arxiv.org/pdf/2303.00262) [[paper]](https://openaccess.thecvf.com/content/WACV2024/papers/Sarukkai_Collage_Diffusion_WACV_2024_paper.pdf) [[code]](https://github.com/VSAnimator/collage-diffusion) 
+ Ziyang Yuan, Mingdeng Cao, Xintao Wang, Zhongang Qi, Chun Yuan, Ying Shan: "*CustomNet: Zero-shot Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models.*" ACM MM (2024) [[arxiv]](https://arxiv.org/pdf/2310.19784) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3664647.3681396) [[code]](https://github.com/TencentARC/CustomNet) 
+ Bo Zhang, Yuxuan Duan, Jun Lan, Yan Hong, Huijia Zhu, Weiqiang Wang, Li Niu: "*ControlCom: Controllable Image Composition using Diffusion Model.*" arXiv preprint arXiv:2308.10040 (2023) [[arxiv]](https://arxiv.org/pdf/2308.10040.pdf) [[code]](https://github.com/bcmi/ControlCom-Image-Composition)
+ Xin Zhang, Jiaxian Guo, Paul Yoo, Yutaka Matsuo, Yusuke Iwasawa: "*Paste, Inpaint and Harmonize via Denoising: Subject-Driven Image Editing with Pre-Trained Diffusion Model.*" arXiv preprint arXiv:2306.07596 (2023) [[arxiv]](https://arxiv.org/pdf/2306.07596.pdf) 
+ Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, Fang Wen: "*Paint by Example: Exemplar-based Image Editing with Diffusion Models.*" CVPR (2023) [[arxiv]](https://arxiv.org/pdf/2211.13227.pdf) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Paint_by_Example_Exemplar-Based_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.pdf) [[code]](https://arxiv.org/pdf/2211.13227.pdf) 
+ Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, Daniel Aliaga: "*ObjectStitch: Generative Object Compositing.*" CVPR (2023) [[arxiv]](https://arxiv.org/pdf/2212.00932.pdf) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_ObjectStitch_Object_Compositing_With_Diffusion_Model_CVPR_2023_paper.pdf) [[code]](https://github.com/bcmi/ObjectStitch-Image-Composition)
+ Sumith Kulal, Tim Brooks, Alex Aiken, Jiajun Wu, Jimei Yang, Jingwan Lu, Alexei A. Efros, Krishna Kumar Singh: "*Putting People in Their Place: Affordance-Aware Human Insertion into Scenes.*" CVPR (2023) [[arxiv]](https://arxiv.org/pdf/2304.14406) [[paper]](https://sumith1896.github.io/affordance-insertion/static/paper/affordance_insertion_cvpr2023.pdf) [[code]](https://github.com/adobe-research/affordance-insertion)

####  Require object-centric finetuning
+ Jiaxuan Chen, Bo Zhang, Qingdong He, Jinlong Peng, Li Niu: "*CareCom: Generative Image Composition with Calibrated Reference Features*", AAAI (2026) [[arxiv]](https://www.arxiv.org/pdf/2511.11060) [[project]](https://www.ustcnewly.com/project_pages/CareCom.html)
+ Jiaxuan Chen, Bo Zhang, Qingdong He, Jinlong Peng, Li Niu: "*MureObjectStitch: Multi-reference Image Composition.*" arXiv preprint arXiv:2411.07462 (2025) [[arxiv]](https://arxiv.org/pdf/2411.07462) [[code]](https://github.com/bcmi/MureObjectStitch-Image-Composition)
+ Nataniel Ruiz, Yuanzhen Li, Neal Wadhwa, Yael Pritch, Michael Rubinstein, David E. Jacobs, Shlomi Fruchter: "*Magic Insert: Style-Aware Drag-and-Drop.*" ICCV (2025) [[arxiv]](https://arxiv.org/pdf/2407.02489) [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Ruiz_Magic_Insert_Style-Aware_Drag-and-Drop_ICCV_2025_paper.pdf)
+ Zhekai Chen, Wen Wang, Zhen Yang, Zeqing Yuan, Hao Chen, Chunhua Shen: "*FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior.*" ECCV (2024) [[arxiv]](https://arxiv.org/pdf/2407.04947) [[paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02529.pdf) [[code]](https://github.com/aim-uofa/FreeCompose)
+ Lingxiao Lu, Bo Zhang, Li Niu: "*DreamCom: Finetuning Text-guided Inpainting Model for Image Composition.*" arXiv preprint arXiv:2309.15508 (2023) [[arxiv]](https://arxiv.org/pdf/2309.15508.pdf) [[code]](https://github.com/bcmi/DreamCom-Image-Composition)
+ Tianle Li, Max Ku, Cong Wei, Wenhu Chen: "*DreamEdit: Subject-driven Image Editing.*" TMLR (2023) [[arxiv]](https://arxiv.org/pdf/2306.12624.pdf) [[paper]](https://openreview.net/pdf?id=P9haooN9v2) [[code]](https://github.com/DreamEditBenchTeam/DreamEdit)



## Other Resources

+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)

