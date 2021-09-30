import numpy as np


def depth_to_disp(depth, baseline, focal_length):
    mask = depth > 0
    disp = baseline * focal_length / depth
    return disp, mask


def disp_to_depth(disp, baseline, focal_length):
    mask = disp > 0
    depth = baseline * focal_length / disp
    return depth, mask


def depth_error(gt_depth, pred_depth, mask):
    total_pixel = np.sum(mask)

    diff = np.abs(gt_depth - pred_depth)
    diff = mask * diff
    diff = np.nan_to_num(diff, nan=0, posinf=0, neginf=0)
    l1 = np.sum(diff) / total_pixel
    rmse = np.square(diff)
    rmse = np.sum(rmse)
    rmse = rmse / total_pixel
    rmse = np.sqrt(rmse)

    ratio1 = pred_depth / gt_depth
    ratio2 = gt_depth / pred_depth
    ratio = np.maximum(ratio1, ratio2)
    a1 = ratio < 1.25
    a2 = ratio < 1.25 ** 2
    a3 = ratio < 1.25 ** 3
    a1 = np.sum(mask * a1) / total_pixel
    a2 = np.sum(mask * a2) / total_pixel
    a3 = np.sum(mask * a3) / total_pixel
    return l1, rmse, a1, a2, a3


def disp_error(gt_disp, pred_disp, mask):
    total_pixel = np.sum(mask)

    diff = np.abs(gt_disp - pred_disp)
    diff = np.nan_to_num(diff, nan=0, posinf=0, neginf=0)
    diff = mask * diff
    epe = np.sum(diff) / total_pixel
    bad3 = diff > 3.0
    bad3 = np.sum(bad3) / total_pixel * 100.0
    return epe, bad3
