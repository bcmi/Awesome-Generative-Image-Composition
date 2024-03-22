
from locale import normalize
from tkinter.tix import IMAGE
import torch
import sys, os
import argparse
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
from metrics.clip_score import measure_foreground_clipscore
from metrics.dino_score import measure_foreground_dinoscore
from metrics.fid_score  import measure_foreground_fid, measure_overall_fid
from metrics.ssim_and_lpips import measure_background_lpips, measure_background_ssim
from metrics.quality_score  import measure_overall_quality
import random
import numpy as np

def seed_all(seed_value=0):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def write_to_csv(results, save_path, metrics=['CLIP_fg', 'DINO_fg', 'FID_fg', 'SSIM_bg', 'LPIPS_bg', 'FID', 'QS']):
    '''
    Save evaluation results to a file
    '''
    rows = [metrics]
    rows.append([results[m] for m in metrics])
    import csv
    with open(save_path, 'w') as f:
        cw = csv.writer(f)
        cw.writerows(rows)
    print('Saving evaluation results to ', save_path)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cocoee", default=os.path.join(curdir, 'data/COCOEE'), type=str)
    parser.add_argument("--testdir", default=os.path.join(curdir, 'data/eval_results/composite'), type=str)
    parser.add_argument("--ckptdir", default=os.path.join(curdir, "checkpoints"), type=str)
    parser.add_argument("--cachedir", default=os.path.join(curdir, "cache"), type=str)
    parser.add_argument("--cocodir", default=os.path.join(curdir, "data/coco/test2017"), type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batchsize", default=16, type=int)
    opt = parser.parse_args()
    assert os.path.exists(opt.testdir), opt.testdir
    assert os.path.exists(opt.cocoee), opt.cocoee
    assert os.path.exists(opt.ckptdir), opt.ckptdir
    os.makedirs(opt.cachedir, exist_ok=True)
    return opt

if __name__ == '__main__':
    seed_all()
    opt = parse_argument()
    device = torch.device(f"cuda:{opt.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    
    result_dir = opt.testdir
    mask_dir   = os.path.join(result_dir, '../mask')
    batch_size = opt.batchsize
    cocoee_dir = opt.cocoee
    cache_dir  = opt.cachedir
    cocotest_dir = opt.cocodir
    clip_path  = os.path.join(opt.ckptdir, "clip-vit-base-patch32")
    dino_path  = os.path.join(opt.ckptdir, 'dino-vits16')
    gmm_path   = os.path.join(opt.ckptdir, 'coco2017_gmm_k20')
    
    results    = {}
    results['CLIP_fg'] = measure_foreground_clipscore(result_dir, mask_dir, cocoee_dir, 
                                                      device, bs=batch_size,
                                                      clip_path=clip_path)
    results['DINO_fg'] = measure_foreground_dinoscore(result_dir, mask_dir, cocoee_dir, 
                                                      device, bs=batch_size,
                                                      dino_path=dino_path)
    results['FID_fg']  = measure_foreground_fid(result_dir,
                                                cocoee_dir, 
                                                device, 
                                                bs=batch_size, 
                                                clip_path=clip_path, 
                                                cache_dir=cache_dir)
    results['LPIPS_bg']= measure_background_lpips(result_dir,
                                                  cocoee_dir, 
                                                  device, 
                                                  bs=batch_size)
    results['SSIM_bg'] = measure_background_ssim(result_dir, 
                                                 cocoee_dir,
                                                 device, 
                                                 bs=batch_size)
    results['FID']     = measure_overall_fid(result_dir, 
                                             cocotest_dir, 
                                             device, 
                                             bs=batch_size, 
                                             clip_path=clip_path, 
                                             cache_dir=cache_dir)
    results['QS']      = measure_overall_quality(result_dir, 
                                                 cocoee_dir,
                                                 device, 
                                                 bs=batch_size,
                                                 gmm_path=gmm_path,
                                                 clip_path=clip_path)
    
    res_file = os.path.join(result_dir, '../metrics.csv')
    print('='*20)
    for k,v in results.items():
        print(k, v)
    print('='*20)
    write_to_csv(results, res_file)
