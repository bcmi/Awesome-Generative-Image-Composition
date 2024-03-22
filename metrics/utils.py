import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def get_bbox_for_foreground(img_name, img_size, bbox_mask_dir):
    # get the bounding box of the foreground region
    mask_name = img_name + '_mask.png'
    mask_path = os.path.join(bbox_mask_dir, mask_name)
    assert os.path.exists(mask_path), mask_path
    mask_img = Image.open(mask_path).convert('L')
    mask_img = mask_img.resize((img_size[0], img_size[1]))
    bbox = mask2bbox(np.array(mask_img))
    return mask_img, bbox

def mask2bbox(mask):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1)
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [x1, y1, x2+1, y2+1]

def get_image_list(image_dir):
    image_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    return image_list

crop_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512)])
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

def gt_clip_preprocess(image, input_size=(224, 224), normalize=True):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    image = crop_transform(image)
    image = image.resize(input_size)
    image = np.array(image) / 255.0
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image).permute(2, 0, 1).float()
    if normalize:
        image = transform(image)
    return image