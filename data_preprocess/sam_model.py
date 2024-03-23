import torch
import numpy as np
from torch import nn
from segment_anything import SamPredictor, sam_model_registry
from torchvision import ops

def select_mask_with_highscore(masks, scores):
    '''
    masks:  b,3,h,w
    scores: b,3
    return  b,1,h,w
    '''
    highscore = torch.argmax(scores, dim=1)
    batch_idx = torch.arange(masks.shape[0]).reshape(-1)
    selected  = masks[batch_idx, highscore].unsqueeze(1) # b,1,h,w
    return selected
    
def get_mask_union(masks):
    '''
    masks:  b,3,h,w
    return: b,1,h,w
    '''
    union = torch.max(masks, dim=1)[0].unsqueeze(1)
    return union

def tensor2mask(t, denorm=False):
    t = t.permute(0,2,3,1)
    t = t.detach().cpu().numpy()
    if denorm:
        mask = np.clip(t * 255, 0, 255).astype(np.uint8)
    else:
        mask = np.clip(t, 0, 1).astype(np.float32)
    return mask

def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out

class SAMModel(nn.Module):
    def __init__(self, opt, device, max_num_boxes=16, mask_fuse='highscore'):
        super().__init__()
        self.device = device
        sam = sam_model_registry["vit_h"](checkpoint=opt.ckpt)
        sam.to(device=device)
        self.prompt = 'box'
        self.predictor = SamPredictor(sam)
        self.mask_fuse = mask_fuse
        self.max_num_boxes = max_num_boxes
        self.score_thresh  = 0.88
    
    def get_input(self, batch):
        inputs = {}
        inputs['image'] = batch['image'].to(self.device)
        boxes  = batch['bboxes'][0].to(self.device)
        num_bboxes = boxes.shape[0]
        if num_bboxes > 1:
            bbox_area = ops.box_area(boxes).float() / (inputs['image'].shape[-1] * inputs['image'].shape[-2])
            keep_idx  = ops.batched_nms(boxes, 
                                        bbox_area,
                                        idxs=torch.ones_like(bbox_area).int(),
                                        iou_threshold=0.7)
            boxes = boxes[keep_idx]
        if boxes.shape[0] > self.max_num_boxes:
            boxes = boxes[:self.max_num_boxes]
        inputs['boxes'] = boxes
        inputs['original_size']  = (batch['original_size'][0].item(), batch['original_size'][1].item())
        return inputs
    
    @torch.no_grad()
    def forward(self, batch):
        inputs = self.get_input(batch)
        boxes = inputs['boxes']
        N = boxes.shape[0]
        self.predictor.set_torch_image(inputs['image'],
                                       inputs['original_size'])
        
        if self.prompt == 'point':
            point_coords = torch.zeros((N,2), device=self.device, dtype=torch.float32)
            point_coords[:,1] = boxes[:,2] - boxes[:,0]
            point_coords[:,0] = boxes[:,3] - boxes[:,1]
            point_coords = point_coords.int().unsqueeze(1)
            point_labels = torch.ones((N,1), device=self.device)
            if self.mask_fuse in ['highscore', 'union']:
                masks, scores, _ = self.predictor.predict_torch(point_coords=point_coords,
                                                                point_labels=point_labels,
                                                                boxes=None,
                                                                multimask_output=True)
                
                if self.mask_fuse == 'highscore':
                    final_mask = select_mask_with_highscore(masks, scores)
                else:
                    final_mask = get_mask_union(masks)
            else:
                final_mask,_,_ = self.predictor.predict_torch(point_coords=point_coords,
                                                            point_labels=point_labels,
                                                            boxes=None,
                                                            multimask_output=False)
        elif self.prompt == 'box':
            if self.mask_fuse in ['highscore', 'union']:
                masks, scores, _ = self.predictor.predict_torch(point_coords=None,
                                                                point_labels=None,
                                                                boxes=inputs['boxes'],
                                                                multimask_output=True)
                # filter out the mask with low score
                if masks.shape[0] > 1:
                    highscored_index = torch.max(scores, dim=-1)[0] > self.score_thresh
                    if highscored_index.sum() > 0:
                        masks  = masks[highscored_index]
                        scores = scores[highscored_index] 
                if self.mask_fuse == 'highscore':
                    final_mask = select_mask_with_highscore(masks, scores)
                else:
                    final_mask = get_mask_union(masks)
            else:
                final_mask,_,_ = self.predictor.predict_torch(point_coords=None,
                                                            point_labels=None,
                                                            boxes=inputs['boxes'],
                                                            multimask_output=False)
        else:
            raise Exception("undefined prompt type: " + self.prompt)
        final_box = batched_mask_to_box(final_mask.squeeze(1))
        return final_mask, final_box