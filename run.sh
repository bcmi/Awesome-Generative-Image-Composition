#!/bin/bash

cocoee="data/COCOEE"
testdir="/data2/zhangbo/generative_composition/baseline_results/Ours/cocoee/results01/images"
cocodir="data/coco/test2017"

# running with single gpu
CUDA_VISIBLE_DEVICES=0 python data_preprocess/sam_on_cocoee.py --cocoee $cocoee
CUDA_VISIBLE_DEVICES=0 python data_preprocess/sam_on_results.py --cocoee $cocoee --testdir $testdir

# running with multiple gpus
# CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 data_preprocess/sam_on_cocoee.py --cocoee $cocoee
# CUDA_VISIBLE_DEVICES=1,3,2 torchrun --nproc_per_node=3 data_preprocess/sam_on_results.py --cocoee $cocoee --testdir $testdir

# compute all metrics
python compute_metrics.py --cocoee $cocoee --testdir $testdir --gpu 0 --batchsize 16 --cocodir $cocodir