from logging import raiseExceptions
import os, sys, time
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2
from tqdm import tqdm
import datetime
import argparse
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
from cocoee import ForegroundDataset as SegmentDataset
from sam_model import SAMModel, tensor2mask

use_ddp = False
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    use_ddp = True
    import torch.distributed as dist
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
else:
    local_rank = 0
    torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cocoee", default=os.path.join(curdir, '../data/COCOEE'), type=str)
    parser.add_argument("--ckpt", default=os.path.join(curdir, "../checkpoints/sam_vit_h_4b8939.pth"), type=str)
    opt = parser.parse_args()
    assert os.path.exists(opt.cocoee), opt.cocoee
    assert os.path.exists(opt.ckpt), opt.ckpt
    return opt    

def main():
    opt     = parse_argument()
    model   = SAMModel(opt, device)
    dataset = SegmentDataset(opt.cocoee)
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=1, 
                            shuffle=False,
                            num_workers=8,
                            drop_last=False,
                            sampler=sampler)
    if local_rank == 0:
        print('{} samples = {} bs x {} batches'.format(
            len(dataset), dataloader.batch_size, len(dataloader)
        ))

    total_time = 0
    if local_rank == 0:
        bar = tqdm(total=len(dataloader))
        bar.set_description('{}'.format(dataset.name))
    else:
        bar = None

    for index,batch in enumerate(dataloader):
        if local_rank == 0:
            bar.update(1)
        res_txt    = batch['bbox_path'][0]
        if os.path.exists(res_txt):
            continue
        start      = time.time()
        mask,boxes = model(batch)
        boxes      = boxes.cpu().numpy()
        mask_np    = tensor2mask(mask, denorm=True)        
        all_boxes  = []
        
        for i in range(boxes.shape[0]):
            mask_path = batch['mask_path'][i]
            mask_name = os.path.basename(mask_path)
            cv2.imwrite(mask_path, mask_np[i])
            box = boxes[i].tolist()
            all_boxes.append(box + [mask_name])
            
        with open(res_txt, 'w') as f:
            for box in all_boxes:
                print(box[0], box[1], box[2], box[3], box[4], file=f)
        
        if local_rank == 0:
            per_time = time.time() - start
            total_time += per_time
            avg_time = total_time / (index+1)
            end_time = avg_time * (len(dataloader) - index - 1)
            last_time = str(datetime.timedelta(seconds=int(end_time)))
            bar.set_postfix({'last_time': last_time,
                            'avg_time': '{:.1f}s'.format(avg_time)})
    
if __name__ == "__main__":
    main()