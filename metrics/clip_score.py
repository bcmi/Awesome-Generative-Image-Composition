import torch
from torchmetrics import Metric
from transformers import CLIPModel as _CLIPModel
from transformers import CLIPProcessor as _CLIPProcessor
from torchvision import transforms
from copy import deepcopy
from PIL  import Image
import numpy as  np
from torch.utils.data import Dataset, DataLoader
import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
from utils import get_bbox_for_foreground
from tqdm import tqdm 


def _get_model_and_processor(model_name_or_path):
    '''
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14-336",
        "openai/clip-vit-large-patch14",
    ] = "openai/clip-vit-large-patch14",
    '''
    model = _CLIPModel.from_pretrained(model_name_or_path).eval()
    processor = _CLIPProcessor.from_pretrained(model_name_or_path)
    return model, processor

transform = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
def clip_preprocess(image, input_size=(224, 224), normalize=True):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    image = image.resize(input_size)
    image = np.array(image) / 255.0
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image).permute(2, 0, 1).float()
    if normalize:
        image = transform(image)
    return image

class CLIPScoreMetric(Metric):
    def __init__(self,num_score,clip_path):
        super().__init__()
        self.num_score = num_score
        for i in range(num_score):
            self.add_state(f"score{i}", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
            self.add_state(f"num{i}",   default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.model, self.processor = _get_model_and_processor(clip_path)
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
        img_features = self.model.get_image_features(images)
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
        return img_features
    
class ForegroundObjectDataset(Dataset):
    def __init__(self, image_dir, mask_dir, cocoee_dir, preprocess_func):
        self.pre_image_dir = image_dir 
        self.image_list   = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
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
        comp_name = self.image_list[index]
        pre_comp_path = os.path.join(self.pre_image_dir, comp_name)
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
        gt_mask = Image.open(gt_mask_path).convert('L').resize((pre_fg.shape[1], pre_fg.shape[0]))
        gt_mask = np.array(gt_mask)[:,:,None]
        gt_fg   = np.array(gt_fg) * (gt_mask > 127)

        return {'pre_fg': self.process_func(Image.fromarray(pre_fg)), 
                'gt_fg': self.process_func(Image.fromarray(gt_fg))}
    

@torch.no_grad()
def measure_foreground_clipscore(pre_dir, mask_dir, cocoee_dir, device, bs=4, clip_path="openai/clip-vit-base-patch32"):
    clip_score  = CLIPScoreMetric(num_score=1, clip_path=clip_path).to(device)
    dataset     = ForegroundObjectDataset(pre_dir, mask_dir, cocoee_dir, clip_preprocess)
    dataloader  = DataLoader(dataset, 
                            num_workers=8, 
                            drop_last=False, 
                            shuffle=False, 
                            batch_size=bs)
    for batch in tqdm(dataloader):
        clip_score.update(batch['pre_fg'].to(device), batch['gt_fg'].to(device), 0)
    fg_score = clip_score.compute(0)
    fg_score = np.round(fg_score.cpu().item(), 2)
    print('CLIP_fg={:.2f}'.format(
        fg_score
    ))
    return fg_score