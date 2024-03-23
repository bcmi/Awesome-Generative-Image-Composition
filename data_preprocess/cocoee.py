import os
from torch.utils import data
import numpy as np
import torch
from PIL import Image
import copy

def check_dir(dir):
    assert os.path.exists(dir), dir
    return dir

def mask2bbox(mask):
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1)
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [x1, y1, x2, y2]

class ForegroundDataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 image_size=1024,
                 keep_aspect_ratio=True):
        self.name = "COCOEE_ref"
        self.dataset_dir = dataset_dir
        assert os.path.exists(dataset_dir), dataset_dir
        self.image_dir  = check_dir(os.path.join(dataset_dir, 'Ref_3500'))
        self.image_list = os.listdir(self.image_dir)
        self.length=len(self.image_list)

        self.bbox_dir = os.path.join(dataset_dir, 'ref_box')
        os.makedirs(self.bbox_dir, exist_ok=True)
        self.mask_dir = os.path.join(dataset_dir, 'ref_mask')
        os.makedirs(self.mask_dir, exist_ok=True)
        self.keep_aspect_ratio = keep_aspect_ratio
        self.image_size = image_size
        from segment_anything.utils.transforms import ResizeLongestSide
        self.transform  = ResizeLongestSide(image_size)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        image_p    = Image.open(image_path).convert("RGB")
        image      = np.array(image_p)
        original_size = image.shape[:2]
        ori_w, ori_h = original_size
        bboxes    = np.array([20, 20, ori_w-20, ori_h-20]).reshape((1,4))
        bbox_file = image_name.replace('.png', '.txt')
        bbox_path = os.path.join(self.bbox_dir, bbox_file)
        mask_path = os.path.join(self.mask_dir, image_name)
        input_image = self.transform.apply_image(image)
        input_bbox  = self.transform.apply_boxes(copy.deepcopy(bboxes), original_size)
        original_image_torch = torch.as_tensor(image).permute(2, 0, 1)
        original_bbox_torch  = torch.from_numpy(bboxes).float()
        
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1)
        input_bbox_torch  = torch.from_numpy(input_bbox).float()
        return_dict = {
            'original_image':  original_image_torch,
            'original_bboxes': original_bbox_torch, 
            'image':  input_image_torch,
            'bboxes': input_bbox_torch,
            'image_path': image_path,
            'original_size': original_size,
            'bbox_path': bbox_path,
            'mask_path': mask_path
        }
        return return_dict

class CompositeDataset(data.Dataset):
    def __init__(self,
                 image_dir,
                 cocoee_dir,
                 out_dir=None,
                 image_size=1024,
                 keep_aspect_ratio=True):
        self.name = "composite_results"
        self.image_dir     = image_dir
        assert os.path.exists(image_dir), image_dir
        self.mask_bbox_dir = check_dir(os.path.join(cocoee_dir, 'Mask_bbox_3500'))
        self.image_list    = os.listdir(self.image_dir)
        self.length        = len(self.image_list)

        if out_dir == None:
            out_dir = image_dir
        self.bbox_dir = os.path.join(out_dir, '../box')
        os.makedirs(self.bbox_dir, exist_ok=True)
        self.mask_dir = os.path.join(out_dir, '../mask')
        os.makedirs(self.mask_dir, exist_ok=True)
        self.keep_aspect_ratio = keep_aspect_ratio
        self.image_size = image_size
        from segment_anything.utils.transforms import ResizeLongestSide
        self.transform  = ResizeLongestSide(image_size)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        im_name    = os.path.splitext(image_name)[0].split('_')[0]
        image_p    = Image.open(image_path).convert("RGB")
        image      = np.array(image_p)
        original_size = image.shape[:2]
        
        # get bounding box from mask
        mask_bbox_path = os.path.join(self.mask_bbox_dir, im_name + '_mask.png')
        assert os.path.exists(mask_bbox_path), mask_bbox_path
        mask_bbox = Image.open(mask_bbox_path).convert("L")
        mask_bbox = mask_bbox.resize((image_p.width, image_p.height))
        bboxes = np.array(mask2bbox(np.asarray(mask_bbox))).reshape((1,4))

        # # get bounding box from txt file
        # origin_bbox_file = os.path.join(self.origin_bbox_dir, im_name + '_GT.txt')
        # assert os.path.exists(origin_bbox_file), origin_bbox_file
        # bboxes = self.load_bbox_file(origin_bbox_file) 

        bbox_file = image_name.replace('.png', '.txt')
        bbox_path = os.path.join(self.bbox_dir, bbox_file)
        mask_path = os.path.join(self.mask_dir, image_name)
        input_image = self.transform.apply_image(image)
        input_bbox  = self.transform.apply_boxes(copy.deepcopy(bboxes), original_size)
        original_image_torch = torch.as_tensor(image).permute(2, 0, 1)
        original_bbox_torch  = torch.from_numpy(bboxes).float()
        
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1)
        input_bbox_torch  = torch.from_numpy(input_bbox).float()
        return_dict = {
            'original_image': original_image_torch,
            'original_bboxes': original_bbox_torch, 
            'image':  input_image_torch,
            'bboxes': input_bbox_torch,
            'image_path': image_path,
            'original_size': original_size,
            'bbox_path': bbox_path,
            'mask_path': mask_path
        }
        return return_dict