"""Depth estimation metrics"""
import torch


def compute_depth_metrics(pred, gt, min_depth=0.1, max_depth=50):
    """
    Compute depth metrics (adapted from Monodepth2).

    Args:
        pred: predicted depth [B, 1, H, W]
        gt: ground truth depth [B, 1, H, W]

    Returns:
        dict with keys: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    """
    # Resize if needed
    if pred.shape != gt.shape:
        pred = torch.nn.functional.interpolate(pred, gt.shape[2:], mode='bilinear', align_corners=False)

    # Valid mask
    mask = (gt > min_depth) & (gt < max_depth)
    if mask.sum() == 0:
        return {k: float('nan') for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']}

    gt = gt[mask].clamp(min_depth, max_depth)
    pred = pred[mask].clamp(min_depth, max_depth)

    # Threshold accuracy
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    # Error metrics
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2))

    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'a1': a1.item(),
        'a2': a2.item(),
        'a3': a3.item()
    }


def disparity_to_depth(disp, baseline=0.54, focal=721):
    """Convert disparity to depth: depth = (baseline * focal) / disp"""
    return (baseline * focal) / disp.abs()
