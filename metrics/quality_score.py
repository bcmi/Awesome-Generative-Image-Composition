import torch
from torchmetrics import Metric
from transformers import CLIPModel as _CLIPModel

from copy import deepcopy
from PIL  import Image
import numpy as  np
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import _compute_fid
import os, sys
import pickle
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
from tqdm import tqdm 
from utils import get_image_list
from utils import get_bbox_for_foreground, clip_preprocess
import warnings
warnings.filterwarnings("ignore")

def get_model(model_name_or_path):
    model = _CLIPModel.from_pretrained(model_name_or_path).eval()
    return model

class PairedDataset(Dataset):
    def __init__(self, image_dir, cocoee_dir, preprocess_func):
        self.image_list   = get_image_list(image_dir)
        self.bbox_mask_dir= os.path.join(cocoee_dir, 'Mask_bbox_3500')
        assert os.path.exists(self.bbox_mask_dir), self.bbox_mask_dir
        self.gt_dir       = os.path.join(cocoee_dir, 'GT_3500')
        assert os.path.exists(self.gt_dir), self.gt_dir
        self.process_func = preprocess_func
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        img_path  = self.image_list[index]
        img_name  = os.path.splitext(os.path.basename(img_path))[0].split('_')[0]
        pre_comp  = Image.open(img_path).convert('RGB')
        bg_mask,_ = get_bbox_for_foreground(img_name, pre_comp.size, self.bbox_mask_dir)
        
        pre_bg  = np.array(pre_comp) * (np.array(bg_mask)[:,:,None] < 127)
        pre_bg  = Image.fromarray(pre_bg)
        
        gt_path = os.path.join(self.gt_dir, img_name+'_GT.png')
        assert os.path.exists(gt_path), gt_path
        gt_comp = Image.open(gt_path).convert('RGB')
        gt_bg  = np.array(gt_comp) * (np.array(bg_mask)[:,:,None] < 127)
        gt_bg  = Image.fromarray(gt_bg)
        bg_mask = torch.from_numpy(np.array(bg_mask.resize((224, 224)))).float() / 255
        
        return {'gt_comp': self.process_func(gt_comp),
                'gt_bg': self.process_func(gt_bg),
                'pre_comp': self.process_func(pre_comp),
                'pre_bg': self.process_func(pre_bg),
                'bg_mask': bg_mask,
                'img_path': img_path}

@torch.no_grad()
def measure_overall_quality(pre_dir, cocoee_dir, device, bs=16, 
                            gmm_path='checkpoints/coco2017_gmm_k20', 
                            clip_path="openai/clip-vit-base-patch32"):
    pca_gmm    = pickle.load(open(gmm_path, "rb"))
    clip       = get_model(clip_path).to(device)
    dataset    = PairedDataset(pre_dir, 
                               cocoee_dir, 
                               clip_preprocess)
    dataloader = DataLoader(dataset, 
                            num_workers=8, 
                            drop_last=False, 
                            shuffle=False, 
                            batch_size=bs)
    min_number = 0
    max_number = 300

    def compute_qs(img):
        feat  = clip.get_image_features(img.to(device))
        feat  = feat.cpu().numpy()
        score = pca_gmm.score_samples(feat)
        score = (score - min_number) / (max_number - min_number)
        score = np.clip(score, 0, 1) * 100
        return score

    score_list = []
    for batch in tqdm(dataloader):
        comp_score = compute_qs(batch['pre_comp'])
        img_paths  = batch['img_path']
        for idx,img_path in enumerate(img_paths):
            this_comp = float(comp_score[idx])
            score_list.append(this_comp)
    
    score_arr  = np.array(score_list).reshape(-1)
    comp_mean  = np.round(score_arr.mean(), 2)
    print('QualityScore={:.2f}'.format(comp_mean))
    return comp_mean