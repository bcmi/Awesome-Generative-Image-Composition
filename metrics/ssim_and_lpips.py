import torch
from copy import deepcopy
from PIL  import Image
import numpy as  np
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import _compute_fid
import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
from tqdm import tqdm 
from utils import get_image_list
from utils import get_bbox_for_foreground

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
        bg_mask,box  = get_bbox_for_foreground(img_name, pre_comp.size, self.bbox_mask_dir)
        
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

def lpips_preprocess(image, input_size=(512, 512)):
    # LPIPS needs the images to be in the [-1, 1] range.
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    image = image.resize(input_size)
    image = np.array(image) / 255.0
    image = 2 * image - 1
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image).permute(2, 0, 1).float()
    return image

def ssim_preprocess(image, input_size=(512, 512)):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    image = image.resize(input_size)
    image = np.array(image) / 255.0
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image).permute(2, 0, 1).float()
    return image

@torch.no_grad()
def measure_background_lpips(pre_dir, cocoee_dir, device, bs=4):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips      = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction="sum").to(device)
    total, count = 0,0
    dataset    = PairedDataset(pre_dir,
                               cocoee_dir, 
                               lpips_preprocess)
    dataloader = DataLoader(dataset, 
                            num_workers=8, 
                            drop_last=False, 
                            shuffle=False, 
                            batch_size=bs)
    for batch in tqdm(dataloader):
        pre_t = batch['pre_bg'].to(device)
        gt_t  = batch['gt_bg'].to(device)
        total += lpips(pre_t, gt_t).detach().cpu().item()
        count += pre_t.shape[0]
    avg = np.round(total / count, 3)
    print('LPIPS_bg={:.3f}'.format(avg))
    return avg

def measure_background_ssim(pre_dir, cocoee_dir, device, bs=4):
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="sum").to(device)
    total, count = 0,0
    dataset    = PairedDataset(pre_dir, 
                               cocoee_dir,
                               ssim_preprocess)
    dataloader = DataLoader(dataset, 
                            num_workers=16, 
                            drop_last=False, 
                            shuffle=False, 
                            batch_size=bs)
    for batch in tqdm(dataloader):
        pre_t = batch['pre_bg'].to(device)
        gt_t  = batch['gt_bg'].to(device)
        total += ssim(pre_t, gt_t).detach().cpu().item()
        count += pre_t.shape[0]
    avg = np.round(total / count, 3)
    print('SSIM_bg={:.3f}'.format(avg))
    return avg
