import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from tqdm import tqdm
import random
from glob import glob
from typing import List, Tuple, Dict

dataset = 'SD_Test_Dataset/'


# Test Class labels for now --> I will train off a regular human dataset while I get the Deer dataset perfected, but this will do as a preliminary one for the unit testins.
class_labels = {
    0: "Human",
    1: "Female 🤓" # NO MOVIES! I'm LOCKED IN, I'M GETTING RICH. 
}

# Get the bounding box from mask:
def mask_to_boxes(mask_path: str) -> List[Tuple[int, int, int, int]]:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w + x, h + y))
    
    
    return boxes 
    
def write_yolo_label(txt_path: str, boxes: List[Tuple[int, int, int, int]], img_w:int, img_h:int, class_id:int=0):
    lines = []
    for (x1, y1, x2, y2) in boxes:
        center_x = ((x1 + x2)/2.0)/img_w
        center_y = ((y1 + y2)/2.0)/img_h
        w = (x2 - x1)/img_w
        h = (y2 - y1)/img_h
        lines.append(f"{class_id} {center_x} {center_y} {w} {h}\n")
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))
    
    
    
    return


class LoadDataset(Dataset):
    """
    New loader for your structure:
      SD_Test_Dataset/
        Men I/
          img/
          masks/
        Women I/
          img/
          masks/
        ...
    It finds all images under each subfolder's `img/` and matches a mask with the same basename
    under that subfolder's `masks/`. The class id is derived from the parent folder name (Men/Women).
    """

    def __init__(self, base_dir, img_subfolder='img', mask_subfolder='masks', img_size=640, augment=False, verbose=True):
        self.base_dir = base_dir
        self.img_subfolder = img_subfolder
        self.mask_subfolder = mask_subfolder
        self.img_size = img_size
        self.augment = augment
        self.samples = []  # list of tuples: (img_path, mask_path, class_id)

        def get_class_from_folder(folder_name: str) -> int:
            # simple heuristic: 'men' -> 0, 'woman' or 'women' -> 1
            low = folder_name.lower()
            if "men" in low:
                return 0
            if "woman" in low or "women" in low:
                return 1
            # default fallback
            return 0

        # iterate immediate subfolders of base_dir
        for entry in sorted(os.listdir(base_dir)):
            group_dir = os.path.join(base_dir, entry)
            if not os.path.isdir(group_dir):
                continue
            img_dir = os.path.join(group_dir, img_subfolder)
            mask_dir = os.path.join(group_dir, mask_subfolder)
            if not os.path.isdir(img_dir):
                continue
            class_id = get_class_from_folder(entry)
            # collect images
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for img_path in glob(os.path.join(img_dir, ext)):
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    # try multiple mask extensions
                    found_mask = None
                    for mext in (".png", ".jpg", ".jpeg", ".bmp"):
                        candidate = os.path.join(mask_dir, base + mext)
                        if os.path.exists(candidate):
                            found_mask = candidate
                            break
                    if found_mask is None:
                        # warn and skip
                        if verbose:
                            print(f"⚠️ No mask for {os.path.basename(img_path)} in {entry}")
                        continue
                    self.samples.append((img_path, found_mask, class_id))

        if len(self.samples) == 0:
            # helpful message if nothing found
            raise ValueError(f"No image+mask pairs found under {base_dir}. Check folder names and 'img'/'masks' subfolders.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, class_id = self.samples[idx]
        img = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
        if img is None:
            raise RuntimeError(f"Failed to read image {img_path}")
        h0, w0 = img.shape[:2]

        # Read mask and get bounding boxes in absolute pixel coords
        boxes = []
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # shouldn't happen because we checked, but safe-guard
            return torch.from_numpy(np.transpose(img.astype(np.float32)/255.0, (2,0,1))), np.zeros((0,5), dtype=np.float32)

        _, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([class_id, float(x1), float(y1), float(x2), float(y2)])

        # apply letterbox resize/pad (same function you already have)
        img_resized, ratio, pad = letterbox(img, new_shape=self.img_size)
        if boxes:
            boxes_np = np.array(boxes, dtype=np.float32)
            # scale boxes by ratio (same factor for both x and y because letterbox keeps ratio)
            boxes_np[:, 1:] = boxes_np[:, 1:] * ratio
            # add padding: pad = (left, top)
            boxes_np[:, [1, 3]] += pad[0]  # x1, x2
            boxes_np[:, [2, 4]] += pad[1]  # y1, y2
        else:
            boxes_np = np.zeros((0, 5), dtype=np.float32)

        # horizontal flip augmentation (same logic you used)
        if self.augment and random.random() < 0.5:
            img_resized = img_resized[:, ::-1, :]
            if len(boxes_np):
                x1 = boxes_np[:, 1].copy()
                x2 = boxes_np[:, 3].copy()
                boxes_np[:, 1] = self.img_size - x2
                boxes_np[:, 3] = self.img_size - x1

        img_resized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_resized, (2, 0, 1)))
        target = boxes_np  # numpy (N,5): [class,x1,y1,x2,y2] in resized coordinates

        return img_tensor, target


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR) 
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    

    return img, r, (left, top)

def autopad(k, p=None): # Pad Kernel
    if p is None:
        p = k//2
    return p

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() # Swish activation function
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

class C3(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        mid = out_channels//2
        self.conv1 = Conv(in_channels, mid, kernel_size=1)
        self.conv2 = Conv(in_channels, mid, kernel_size=1)
        self.m = nn.Sequential(*[Bottleneck(mid, mid) for _ in range(n)])
        self.conv3 = Conv(2*mid, out_channels, kernel_size=1)

    def forward(self, x):
        y1 = self.m(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat((y1, y2), dim=1))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1)
        self.conv2 = Conv(in_channels, out_channels, kernel_size=3)
    
    
    def forward(self, x):
        return (self.conv2(self.conv1(x)) + x)


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, in_channels//2, 1)
        self.pool1 = nn.MaxPool2d(5, 1, 2)
        self.pool2 = nn.MaxPool2d(9, 1, 4)
        self.pool3 = nn.MaxPool2d(13, 1, 6)
        self.conv2 = Conv(in_channels * 2, out_channels, 1)
    

    def forward(self, x):
        x1 = self.conv1(x)
        y = torch.cat([x1, self.pool1(x1), self.pool2(x1), self.pool3(x1)], dim=1)
        return self.conv2(y)
    


class Detect(nn.Module):
    def __init__(self, num_classes=2, channels=[128, 256, 512], strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 5 + num_classes # --> Object + bbox(4) + classes
        self.strides = strides
        self.m = nn.ModuleList([nn.Conv2d(c, self.num_outputs, kernel_size=1) for c in channels])
        self.grid = ([torch.zeros(1)] * len(channels)) # Make sure that this is anchros-free! 

    def forward(self, x):
        out = []
        for i, xi in enumerate(x):
            predict = self.m[i](xi)
            B, _, H, W = predict.shape
            predict = predict.view(B, self.num_outputs, H, W).permute(0, 2, 3, 1).contiguous()
            out.append(predict)
        return out
    
# Now... it is time...
# FOR THE MINI YOLO


class MiniYOLO(nn.Module):
    def __init__(self, num_classes=2):
        super(MiniYOLO, self).__init__()
        self.stem = nn.Sequential(
            Conv(3, 32, kernel_size=3, stride=2),
            Conv(32, 64, kernel_size=3, stride=2)
        )

        self.b1 = nn.Sequential(Conv(64, 128, kernel_size=3, stride=2), C3(128, 128, n=1))
        self.b2 = nn.Sequential(Conv(128, 256, kernel_size=3, stride=2), C3(256, 256, n=2))
        self.b3 = nn.Sequential(Conv(256, 512, kernel_size=3, stride=2), C3(512, 512, n=2))
        self.spp = SPP(512, 512)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_conv = Conv(512 + 256, 256, kernel_size=1)
        self.p4 = C3(256, 256, n=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_conv = Conv(256 + 128, 128, kernel_size=1)
        self.p3 = C3(128, 128, n=1)
        

        self.down1 = Conv(128, 256, kernel_size=3, stride=2)
        self.pan4 = C3(256 + 256, 256, n=1)
        self.down2 = Conv(256, 512, kernel_size=3, stride=2)
        self.pan5 = C3(512 + 512, 512, n=1)
        

        self.detect = Detect(num_classes, channels=[128, 256, 512], strides=[8, 16, 32])

    def forward(self, x):
        x = self.stem(x)
        p1 = self.b1(x)
        p2 = self.b2(p1)
        p3 = self.b3(p2)
        p5 = self.spp(p3)

        #top->down
        up4 = self.up1(p5)
        cat4 = torch.cat([up4, p2], dim=1)
        p4 = self.p4(self.p4_conv(cat4))
        up3 = self.up2(p4)
        cat3 = torch.cat([up3, p1], dim=1)
        p3_out = self.p3(self.p3_conv(cat3))

        #bottom->up
        down4 = self.down1(p3_out)
        cat4b = torch.cat([down4, p4], dim=1)
        pan4 = self.pan4(cat4b)
        down5 = self.down2(pan4)
        cat5b = torch.cat([down5, p5], dim=1)
        pan5 = self.pan5(cat5b)

        predictions = self.detect([p3_out, pan4, pan5]) # Here, p3_out is fine, pan4 is mid, and pan5 is coarse images.
        return predictions 
    
    def sigmoid(x):
        return 1/(1+torch.exp(-x))
    
    
    
def get_loss(predictions, targets, device, img_size=640, lambda_box=5.0, lambda_obj=1.0, lambda_class=1.0):
    batch_size = predictions[0].shape[0]
    device = predictions[0].device
    total_box_loss = torch.tensor(0.0, device=device)
    total_obj_loss = torch.tensor(0.0, device=device)
    total_class_loss = torch.tensor(0.0, device=device)
    strides = [8, 16, 32]
    
    for b in range(batch_size):
        t = targets[b]
        if t.shape[0] == 0:
            for si, prediction in enumerate(predictions):
                obj_predict = prediction[b][..., 0]
                total_obj_loss += F.binary_cross_entropy_with_logits(obj_predict, torch.zeros_like(obj_predict), reduction='sum')
            continue
        for gt in t:
            clss = int(gt[0].item())
            x1, y1, x2, y2 = gt[1:].tolist()
            gx = (x1 + x2)/2.0
            gy = (y1 + y2)/2.0
            gw = (x2 - x1) # GAMES WORKSHOP???
            gh = (y2 - y1)

            scale_idx = 0 
            area = gw * gh
            if area > (img_size/32)**2:
                scale_idx = 2
            elif area > (img_size/16)**2:
                scale_idx = 1
            else:
                scale_idx = 0

            prediction = predictions[scale_idx]
            B, H, W, _ = prediction.shape
            stride = strides[scale_idx]
            cx_cell = (gx/stride)
            cy_cell = (gy/stride)

            ix = int(cx_cell)
            iy = int(cy_cell)
            ix = max(0, min(W - 1, ix))
            iy = max(0, min(H - 1, iy))

            out = prediction[b, iy, ix, :]
            obj_target = torch.tensor(1.0, device=device)
            total_obj_loss += F.binary_cross_entropy_with_logits(out[0], obj_target)

            # Build the box as well (eh uh... eh uh... eh uh... pulling out the coupe at the lot, told em fuck 12, fuck SWAT)
            # I wonder if anyone will read this random line... or any of this code. How about this:
            # If you read this, please go in the group chat and type:
            # "Did you guys know Bogdan likes potatoes?" --> I'll know.

            px = (torch.sigmoid(out[1]) + ix) * stride
            py = (torch.sigmoid(out[2]) + iy) * stride
            pw = torch.exp(out[3]) * stride
            ph = torch.exp(out[4]) * stride
            px1 = (px - pw)/2
            py1 = (py - ph)/2 
            px2 = (px + pw)/2
            py2 = (py + ph)/2
            pbox = torch.stack([px1, py1, px2, py2]).unsqueeze(0)
            gbox = torch.tensor([x1,y1,x2,y2], device=device).unsqueeze(0)
            ciou = bbox_iou_ciou(pbox, gbox)
            total_box_loss += (1.0 - ciou).sum() * lambda_box

            class_predict = out[5:]
            target_class = torch.zeros_like(class_predict); target_class[clss] = 1.0
            total_class_loss += F.binary_cross_entropy_with_logits(class_predict, target_class)

            loss = total_box_loss + lambda_obj * total_obj_loss + lambda_class * total_class_loss
            return loss, {'box': total_box_loss.item(), 'obj': total_obj_loss.item(),'cls': total_class_loss.item()}


def sigmoid(x):
    return 1/(1+torch.exp(-x))
        
def bbox_iou_ciou(box1, box2, eps=1e-7):
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.max(box1[:, 2], box2[:, 2])
    y2 = torch.max(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) # Intersection
        

    area1 = (box1[:,2]-box1[:,0]).clamp(0) * (box1[:,3]-box1[:,1]).clamp(0)
    area2 = (box2[:,2]-box2[:,0]).clamp(0) * (box2[:,3]-box2[:,1]).clamp(0)
    union = area1 + area2 - inter + eps
    iou = inter / union

    cx1 = (box1[:,0] + box1[:,2]) / 2
    cy1 = (box1[:,1] + box1[:,3]) / 2
    cx2 = (box2[:,0] + box2[:,2]) / 2
    cy2 = (box2[:,1] + box2[:,3]) / 2
    center_dist = (cx1-cx2)**2 + (cy1-cy2)**2

    enclose_x1 = torch.min(box1[:,0], box2[:,0])
    enclose_y1 = torch.min(box1[:,1], box2[:,1])
    enclose_x2 = torch.max(box1[:,2], box2[:,2])
    enclose_y2 = torch.max(box1[:,3], box2[:,3])
    enclose_diag = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + eps

    w1 = (box1[:,2]-box1[:,0]).clamp(min=eps)
    h1 = (box1[:,3]-box1[:,1]).clamp(min=eps)
    w2 = (box2[:,2]-box2[:,0]).clamp(min=eps)
    h2 = (box2[:,3]-box2[:,1]).clamp(min=eps)

    v = (4/(np.pi**2)) * (torch.atan(w2/h2) - torch.atan(w1/h1))**2
    with torch.no_grad():
        alpha = v/((1 - iou) + v + eps)
    ciou = iou - (center_dist/enclose_diag) - alpha * v
    return ciou.clamp(-1.0, 1.0)
    

def decode_predictions(preds, conf_thresh=0.25, img_size=640, strides=[8,16,32]):
    """
    preds: list of tensors (B,H,W,C) raw logits
    Return list per image of detections: [ (x1,y1,x2,y2, conf, cls) ... ]
    """
    results = []
    B = preds[0].shape[0]
    for b in range(B):
        boxes = []
        for si, pred in enumerate(preds):
            H,W,_ = pred.shape[1:]
            stride = strides[si]
            p = pred[b].detach()
            # p: H,W,C
            obj = torch.sigmoid(p[...,0])
            bx = torch.sigmoid(p[...,1])  # offset
            by = torch.sigmoid(p[...,2])
            bw = torch.exp(p[...,3])
            bh = torch.exp(p[...,4])
            cls_logits = torch.sigmoid(p[...,5:])  # for multiple classes

            # build positions grid
            grid_y = torch.arange(H, device=p.device).view(H,1).expand(H,W)
            grid_x = torch.arange(W, device=p.device).view(1,W).expand(H,W)
            cx = (bx + grid_x) * stride
            cy = (by + grid_y) * stride
            w = bw * stride
            h = bh * stride
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2

            # flatten
            x1 = x1.reshape(-1); y1 = y1.reshape(-1); x2 = x2.reshape(-1); y2 = y2.reshape(-1)
            conf = obj.reshape(-1)
            cls_scores = cls_logits.reshape(-1, cls_logits.shape[-1])  # (N, nc)
            # get best class
            best_conf, best_cls = torch.max(cls_scores, dim=1)
            final_conf = conf * best_conf  # combine objectness and class prob
            mask = final_conf > conf_thresh
            for xi, yi, xa, xb, cc, cl in zip(x1[mask].cpu().numpy(), y1[mask].cpu().numpy(), x2[mask].cpu().numpy(), y2[mask].cpu().numpy(), final_conf[mask].cpu().numpy(), best_cls[mask].cpu().numpy()):
                boxes.append([xi, yi, xa, xb, float(cc), int(cl)])
        results.append(boxes)
    return results

def nms(boxes, iou_threshold=0.45):
    """
    Simple NMS: boxes is list of [x1,y1,x2,y2,conf,cls]
    """
    if not boxes:
        return []
    boxes = np.array(boxes)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    scores = boxes[:,4]
    classes = boxes[:,5].astype(np.int32)
    keep = []
    # perform per-class NMS
    unique_cls = np.unique(classes)
    for c in unique_cls:
        idxs = np.where(classes==c)[0]
        xs = x1[idxs]; ys = y1[idxs]; xe = x2[idxs]; ye = y2[idxs]
        sc = scores[idxs]
        order = sc.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            keep.append(idxs[i])
            xx1 = np.maximum(xs[i], xs[order[1:]])
            yy1 = np.maximum(ys[i], ys[order[1:]])
            xx2 = np.minimum(xe[i], xe[order[1:]])
            yy2 = np.minimum(ye[i], ye[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (xe[i]-xs[i]) * (ye[i]-ys[i])
            area_others = (xe[order[1:]]-xs[order[1:]]) * (ye[order[1:]]-ys[order[1:]])
            union = area_i + area_others - inter
            iou = inter / (union + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds+1]
    return boxes[keep].tolist()

# -------------------------
# Training loop
# -------------------------
def train(
    model,
    train_loader,
    optimizer,
    device,
    epochs=10,
    val_loader=None,
    img_size=640
):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        running_loss = 0.0
        for i, (imgs, targets) in pbar:
            imgs = imgs.to(device)
            
            # Convert targets from numpy arrays to tensors
            packed_targets = [torch.from_numpy(t).float().to(device) for t in targets]

            optimizer.zero_grad()
            preds = model(imgs)  # list per scale
            loss, loss_items = get_loss(preds, packed_targets, device, img_size=img_size)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{epochs} Loss {running_loss/(i+1):.4f} box{loss_items['box']:.3f} obj{loss_items['obj']:.3f} cls{loss_items['cls']:.3f}")
    # finished
    return model


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]  # keep as numpy arrays for conversion in train()
    return imgs, targets



if __name__ == "__main__":


    print("STARTING TRAINING!!!")
    if(torch.cuda.is_available()):
        device = torch.device('cuda')
        print(f"Found GPU!! Using {device}")
    else:
        device = torch.device('CPU')
        print(f"Did not find GPU... :( --> Using a stinky CPU") 

    # paths (update these)
    train_images = "SD_Test_Dataset"   # root that contains Men I, Women I, etc.
    # Create dataset and dataloader
    dataset = LoadDataset(train_images, img_size=640, augment=True)
    print("Found", len(dataset), "image+mask pairs")

 

    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MiniYOLO(num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    trained = train(model, loader, optimizer, device, epochs=20, img_size=640)

    # Save weights
    torch.save(trained.state_dict(), "mini_yolo_test.pth")

    model.eval()
    img = cv2.imread("test_image.jpg")[:,:,::-1]
    img0 = img.copy()
    img, ratio, pad = letterbox(img, new_shape=640)
    x = img.astype(np.float32) / 255.0
    x = torch.from_numpy(np.transpose(x, (2,0,1))).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(x)
        dets = decode_predictions(preds, conf_thresh=0.25, img_size=640)
        dets = nms(dets[0], iou_threshold=0.45)
    # draw detections
    for d in dets:
        x1,y1,x2,y2,conf,cls = d
        x1 = int((x1 - pad[0]) / ratio); y1 = int((y1 - pad[1]) / ratio)
        x2 = int((x2 - pad[0]) / ratio); y2 = int((y2 - pad[1]) / ratio)
        cv2.rectangle(img0, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img0, f"{cls}:{conf:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.imwrite("out_detect.jpg", img0[:,:,::-1])
    print("done - output saved to out_detect.jpg")
