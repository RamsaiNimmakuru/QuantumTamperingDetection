import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    inter = (pred * target).sum()
    return 1 - ((2.*inter + smooth)/(pred.sum()+target.sum()+smooth))

def dice_coeff(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    return (2.*inter + 1e-6)/(pred.sum()+target.sum()+1e-6)

def iou_score(pred, target, thr=0.5):
    pred = (torch.sigmoid(pred) > thr).float()
    inter = (pred * target).sum()
    union = pred.sum()+target.sum()-inter
    return (inter+1e-6)/(union+1e-6)

def pixel_acc(pred, target, thr=0.5):
    pred = (torch.sigmoid(pred) > thr).float()
    return (pred==target).float().mean()

def precision_recall_f1(pred, target, thr=0.5):
    pred = (torch.sigmoid(pred) > thr).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

def compute_all_metrics(pred, target):
    iou = iou_score(pred, target)
    dice = dice_coeff(pred, target)
    p, r, f1 = precision_recall_f1(pred, target)
    acc = pixel_acc(pred, target)
    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(f1),
        "PixelAcc": float(acc)
    }
