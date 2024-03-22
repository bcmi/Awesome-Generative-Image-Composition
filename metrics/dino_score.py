import torch
from torchmetrics import Metric
from transformers import ViTImageProcessor
from transformers import ViTModel
from torchvision import transforms
from copy import deepcopy
from PIL  import Image
import numpy as  np
from torch.utils.data import Dataset, DataLoader
import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
from utils import get_bbox_for_foreground, get_image_list
from tqdm import tqdm

def get_model(model_name_or_path):
    model = ViTModel.from_pretrained(model_name_or_path)
    return model.eval()

def get_processor(model_name_or_path):
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    return processor

class DINOScoreMetric(Metric):
    def __init__(self,num_score,DINO_path):
        super().__init__()
        self.num_score = num_score
        for i in range(num_score):
            self.add_state(f"score{i}", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
            self.add_state(f"num{i}",   default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.model = get_model(DINO_path)
        self.image_size = (224, 224)

    def update(self, input1, input2, index=0, mask1=None, bbox1=None, mask2=None, bbox2=None):
        device = input1.device if isinstance(input1, torch.Tensor) else input1[0].device
        feature1 = self.input2feature(input1, device, mask1, bbox1)
        feature2 = self.input2feature(input2, device, mask2, bbox2)
        # cosine similarity between feature vectors
        per_score = 100 * (feature1 * feature2).sum(axis=-1)
        # print(score) if mask1 == None else print('mask', score)
        score = getattr(self, f'score{index}')
        num   = getattr(self, f'num{index}')
        score += torch.sum(per_score)
        num += per_score.numel()
        
    def compute(self, index=-1):
        if index < 0:
            all_val = []
            for i in range(self.num_score):
                score = getattr(self, f'score{i}')
                num   = getattr(self, f'num{i}')
                val   = score / num.float() if num > 0 else 0.
                all_val.append(val)
            return all_val if len(all_val) > 1 else all_val[0]
        else:
            score = getattr(self, f'score{index}')
            num   = getattr(self, f'num{index}')
            val   = score / num.float() if num > 0 else 0.
            return val
        
    def input2feature(self, input, device, mask, bbox):
        if mask != None:
            input = self.crop_by_mask(input, mask)
        elif bbox != None:
            input = self.crop_by_bbox(input, bbox)
        if input.shape[-2:] != self.image_size:
            input = transforms.Resize(self.image_size)(input)
        feature = self.encode_image(input, device)
        return feature

    def crop_by_bbox(self, image, bbox):
        out_size = self.image_size
        h, w = image.shape[-2:]
        bbox_int = deepcopy(bbox)
        bbox_int[:, 0::2] *= w
        bbox_int[:, 1::2] *= h
        bbox_int = bbox_int.int()
        bbox_int[:,2:] = torch.maximum(bbox_int[:,2:], bbox_int[:,:2] + 1)
        crops = []
        for i in range(bbox_int.shape[0]):
            x1,y1,x2,y2 = bbox_int[i]
            if x1 < x2 and y1 < y2:
                crop = image[i:i+1, :, y1:y2, x1:x2]
            else:
                print('invalid coordinates generated from mask (x1,y1,x2,y2)', x1,y1,x2,y2)
                crop = image
            crop = transforms.Resize(out_size)(crop)
            crops.append(crop)
        crops = torch.cat(crops, dim=0)
        return crops

    def crop_by_mask(self, image, mask):
        out_size = self.image_size
        thresh = 0.5
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        assert mask.ndim == 4 and image.ndim == 4, f'{mask.ndim} vs {image.ndim}'
        if mask.shape[-2:] != image.shape[-2:]:
            mask = transforms.Resize(image.shape[-2:])(mask)
        crops = []
        for i in range(mask.shape[0]):
            y, x = torch.where(mask[i,0] > thresh)
            x1 = torch.min(x).item()
            y1 = torch.min(y).item()
            x2 = torch.max(x).item()
            y2 = torch.max(y).item()
            if x1 < x2 and y1 < y2:
                crop = image[i:i+1, :, y1:y2, x1:x2]
            else:
                print('invalid coordinates generated from mask (x1,y1,x2,y2)', x1,y1,x2,y2)
                crop = image
            crop = transforms.Resize(out_size)(crop)
            crops.append(crop)
        crops = torch.cat(crops, dim=0)
        return crops
    
    def encode_image(self, images, device):
        outputs      = self.model(images)
        img_features = outputs.pooler_output
        # print('image{}, features {}'.format(images.shape, img_features.shape))
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
        return img_features
    
class ForegroundObjectDataset(Dataset):
    def __init__(self, image_dir, mask_dir, cocoee_dir, preprocess_func):
        self.pre_image_dir = image_dir 
        self.image_list   = get_image_list(image_dir)
        self.pre_mask_dir = mask_dir
        self.gt_image_dir = os.path.join(cocoee_dir, 'Ref_3500')
        assert os.path.exists(self.gt_image_dir), self.gt_image_dir
        self.gt_mask_dir  = os.path.join(cocoee_dir, 'ref_mask')
        assert os.path.exists(self.gt_mask_dir), self.gt_mask_dir
        self.bbox_mask_dir= os.path.join(cocoee_dir, 'Mask_bbox_3500')
        assert os.path.exists(self.bbox_mask_dir), self.bbox_mask_dir
        self.process_func = preprocess_func
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        pre_comp_path = self.image_list[index]
        comp_name = os.path.basename(pre_comp_path)
        prefix    = os.path.splitext(comp_name)[0].split('_')[0]
        pre_comp  = Image.open(pre_comp_path).convert('RGB')
        # crop synthesis foreground from composite image
        _,box     = get_bbox_for_foreground(prefix, pre_comp.size, self.bbox_mask_dir)
        x1,y1,x2,y2 = box
        pre_fg    = np.array(pre_comp)[y1:y2, x1:x2]
        # crop foreground mask
        pre_mask_path = os.path.join(self.pre_mask_dir, os.path.splitext(comp_name)[0] + '.png')
        if not os.path.exists(pre_mask_path):
            pre_mask_path = os.path.join(self.pre_mask_dir, prefix + '_GT.png')
        pre_mask = Image.open(pre_mask_path).convert('L').resize(pre_comp.size)
        pre_mask = np.array(pre_mask)[y1:y2, x1:x2, None]
        # fill the non-foreground area of real/generated foreground patches.
        pre_fg   = pre_fg * (pre_mask > 127)
        
        gt_fg_path   = os.path.join(self.gt_image_dir,  prefix + '_ref.png')
        gt_mask_path = os.path.join(self.gt_mask_dir, prefix + '_ref.png')
        gt_fg   = Image.open(gt_fg_path).convert('RGB').resize((pre_fg.shape[1], pre_fg.shape[0]))
        gt_fg   = np.array(gt_fg)
        gt_mask = Image.open(gt_mask_path).convert('L').resize((pre_fg.shape[1], pre_fg.shape[0]))
        gt_mask = np.array(gt_mask)[:,:,None]
        gt_fg   = gt_fg * (gt_mask > 127)
        
        pre_fg_t = self.process_func(Image.fromarray(pre_fg), return_tensors="pt")['pixel_values'][0]
        gt_fg_t  = self.process_func(Image.fromarray(gt_fg), return_tensors="pt")['pixel_values'][0]
        return {'pre_fg': pre_fg_t, 
                'gt_fg': gt_fg_t}
        
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
def measure_foreground_dinoscore(pre_dir, mask_dir, cocoee_dir, device, bs=4, dino_path="checkpoints/dino-vits16"):
    DINO_score  = DINOScoreMetric(num_score=1, DINO_path=dino_path).to(device)
    processor   = get_processor(dino_path)
    dataset     = ForegroundObjectDataset(pre_dir, mask_dir, cocoee_dir, processor)
    dataloader  = DataLoader(dataset, 
                            num_workers=8, 
                            drop_last=False, 
                            shuffle=False, 
                            batch_size=bs)
    for batch in tqdm(dataloader):
        DINO_score.update(batch['pre_fg'].to(device), batch['gt_fg'].to(device), 0)
    fg_score = DINO_score.compute(0)
    fg_score = np.round(fg_score.cpu().item(), 2)
    print('DINO_fg={:.2f}'.format(
        fg_score
    ))
    return fg_score