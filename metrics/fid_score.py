import torch
from torchmetrics import Metric
from transformers import CLIPModel as _CLIPModel
from copy import deepcopy
from PIL  import Image
import numpy as  np
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import _compute_fid
from torchvision import transforms
import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
from tqdm import tqdm 
from utils import get_bbox_for_foreground, get_image_list, clip_preprocess, gt_clip_preprocess

def get_model(model_name_or_path):
    model = _CLIPModel.from_pretrained(model_name_or_path)
    return model.eval()

class ImageDataset(Dataset):
    def __init__(self, image_dir, preprocess_func):
        self.image_list = get_image_list(image_dir)
        self.process_func = preprocess_func
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index]
        im_t  = self.process_func(image)
        return im_t

class CLIPFID(Metric):
    def __init__(self, model_name_or_path, num_features=512):
        super().__init__()
        self.clip   = get_model(model_name_or_path)
        mx_nb_feets = (num_features, num_features)
        self.add_state("real_features_sum", torch.zeros(num_features).float(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(mx_nb_feets).float(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(num_features).float(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(mx_nb_feets).float(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def encode_image(self, images):
        img_features = self.clip.get_image_features(images)
        return img_features

    @torch.no_grad()
    def update(self, imgs, real: bool) -> None:  # type: ignore
        """Update the state with extracted features."""
        features = self.encode_image(imgs)

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self):
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)


class ForegroundPatchDataset(Dataset):
    def __init__(self, image_dir, cocoee_dir, preprocess_func):
        self.image_list = get_image_list(image_dir)
        self.mask_dir   = os.path.join(cocoee_dir, 'Mask_bbox_3500')
        assert os.path.exists(self.mask_dir), self.mask_dir
        self.process_func = preprocess_func
        self.image_size = (224, 224)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image  = self.image_list[index]
        img    = Image.open(image).resize(self.image_size)
        img_np = np.array(img)
        prefix = os.path.splitext(os.path.basename(image))[0].split('_')[0]
        _,box  = get_bbox_for_foreground(prefix, img.size, self.mask_dir)
        x1,y1,x2,y2 = box
        fg_img = Image.fromarray(img_np[y1:y2, x1:x2])
        im_t  = self.process_func(fg_img)
        return im_t
    
def crop_cocoee_foregrounds(cocoee_dir, cache_dir):
    image_dir  = os.path.join(cocoee_dir, 'GT_3500')
    mask_dir   = os.path.join(cocoee_dir, 'Mask_bbox_3500')
    image_list = get_image_list(image_dir)
    gtfg_dir   = os.path.join(cache_dir, 'COCOEE_GTFG')
    os.makedirs(gtfg_dir, exist_ok=True)
    
    for image in image_list:
        img    = Image.open(image)
        img_np = np.array(img)
        prefix = os.path.splitext(os.path.basename(image))[0].split('_')[0]
        _,box  = get_bbox_for_foreground(prefix, img.size, mask_dir)
        x1,y1,x2,y2 = box
        fg_img = Image.fromarray(img_np[y1:y2, x1:x2])
        fg_img.save(os.path.join(gtfg_dir, os.path.basename(image)))
    print('crop {} gt foreground patches and save to {}'.format(
        len(os.listdir(gtfg_dir)), gtfg_dir
    ))
    return gtfg_dir
    
def register_fid_real_images(fid, device, real_dir, processor, bs, dataname, cache_dir, is_foreground=False):
    cache_path = os.path.join(cache_dir, f'{dataname}.pth')
    if os.path.exists(cache_path):
        print('load fid cache from ', cache_path)
        cache_dict = torch.load(cache_path)
        fid.real_features_sum = cache_dict['real_features_sum'].to(device)
        fid.real_features_cov_sum = cache_dict['real_features_cov_sum'].to(device)
        fid.real_features_num_samples = cache_dict['real_features_num_samples'].to(device)
    else:
        if is_foreground:
            real_dir = crop_cocoee_foregrounds(real_dir, cache_dir) 
        realset = ImageDataset(real_dir, processor)
        realloader  = DataLoader(realset, shuffle=False, drop_last=False,
                                num_workers=8, batch_size=bs)
        for real_t in tqdm(realloader, desc="real images"):
            fid.update(real_t.to(device), real=True)
        
        cache_dict = {'real_features_sum': fid.real_features_sum,
                     'real_features_cov_sum': fid.real_features_cov_sum,
                     'real_features_num_samples': fid.real_features_num_samples}
        torch.save(cache_dict, cache_path)
        print('save fid cache to ', cache_path)
    return fid


@torch.no_grad()
def measure_foreground_fid(fake_dir, cocoee_dir, device, bs=16, 
                           clip_path="openai/clip-vit-base-patch32", cache_dir="cache"):
    fid        = CLIPFID(clip_path).to(device)
    fid        = register_fid_real_images(fid, 
                                          device, 
                                          cocoee_dir, 
                                          clip_preprocess, 
                                          bs, 
                                          dataname='cocoee_gtfg',
                                          cache_dir=cache_dir,
                                          is_foreground=True)
    
    fakeset    = ForegroundPatchDataset(fake_dir, 
                                        cocoee_dir, 
                                        clip_preprocess)
    fakeloader = DataLoader(fakeset, 
                            shuffle=False, 
                            drop_last=False,
                            num_workers=8, 
                            batch_size=bs)
    for fake_t in tqdm(fakeloader, desc="fake images"):
        fid.update(fake_t.to(device), real=False)
    avg_fid = fid.compute().detach().cpu().item()
    result = np.round(avg_fid,2)
    print('FID_fg={:.2f}'.format(result))
    return result


@torch.no_grad()
def measure_overall_fid(fake_dir, real_dir, device, bs=16, 
                        clip_path="openai/clip-vit-base-patch32", cache_dir="cache"):
    fid        = CLIPFID(clip_path).to(device)
    fid        = register_fid_real_images(fid, 
                                          device, 
                                          real_dir, 
                                          gt_clip_preprocess, 
                                          bs, 
                                          dataname='coco2017_test',
                                          cache_dir=cache_dir)
    fakeset    = ImageDataset(fake_dir, 
                              clip_preprocess)
    fakeloader = DataLoader(fakeset, 
                            shuffle=False, 
                            drop_last=False,
                            num_workers=8, 
                            batch_size=bs)
    for fake_t in tqdm(fakeloader, desc="fake images"):
        fid.update(fake_t.to(device), real=False)
    avg_fid = fid.compute().detach().cpu().item()
    result = np.round(avg_fid,2)
    print('FID={:.2f}'.format(result))
    return result
