import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import depth_to_disp, disp_to_depth, depth_error, disp_error

parser = argparse.ArgumentParser(description="Error Evaluation Tools")
parser.add_argument('--file_name', type=str, default="32",
                    help="File name for the image to be evaluated, all imgs should be placed in imgs/ w/ the same name")
parser.add_argument('--model', type=str, default="stereonet", help="The model to be evaluated")
parser.add_argument('--gt_d_ext', type=str, default=".csv", help="Extension for ground truth", choices=[".csv", ".exr"])
parser.add_argument('--pred_disp', type=bool, default=True,
                    help="If set to True, the prediction is disparity. Otherwise, it is depth")
parser.add_argument('--gt_disp', type=bool, default=False,
                    help="If set to True, the prediction is disparity. Otherwise, it is depth")
parser.add_argument('--focal_length', type=float, default=385.88, help="Focal length of the camera in px")
parser.add_argument('--baseline', type=float, default=0.05015, help="Baseline of the camera in m")
parser.add_argument('--max_depth', type=float, default=10.0, help="Maximum depth in m")
parser.add_argument('--max_disp', type=int, default=128, help="Maximum disparity in px")
args = parser.parse_args()

coord = []

def disp_depth_conversion(pred, gt):
    if args.pred_disp:
        pred_disp = pred
        pred_depth, _ = disp_to_depth(pred, args.baseline, args.focal_length)
    else:
        pred_depth = pred
        pred_disp, _ = depth_to_disp(pred, args.baseline, args.focal_length)

    if args.gt_disp:
        gt_disp = gt
        disp_mask = gt_disp > 0
        gt_depth, depth_mask = disp_to_depth(gt, args.baseline, args.focal_length)
    else:
        gt_depth = gt
        depth_mask = gt_depth > 0
        gt_disp, disp_mask = depth_to_disp(gt, args.baseline, args.focal_length)

    gt_depth = np.clip(gt_depth, a_min=0, a_max=args.max_depth)
    pred_depth = np.clip(pred_depth, a_min=0, a_max=args.max_depth)

    gt_disp = np.clip(gt_disp, a_min=0, a_max=args.max_disp)
    pred_disp = np.clip(pred_disp, a_min=0, a_max=args.max_disp)

    return gt_depth, gt_disp, pred_depth, pred_disp, depth_mask, disp_mask


def SGBM_disp_depth(l_rgb, r_rgb):
    stereo = cv2.StereoSGBM_create(numDisparities=args.max_disp, blockSize=5)
    disp = stereo.compute(l_rgb, r_rgb)
    disp = disp / 16.0
    depth, _ = disp_to_depth(disp, args.baseline, args.focal_length)
    return depth, disp


def get_ROI_coord(img):
    def onclick(event):
        x = event.xdata
        y = event.ydata
        x = int(round(x))
        y = int(round(y))
        global coord
        if len(coord) == 0:
            coord.append((x, y))
        elif len(coord) == 1:
            if coord[0][0] == x or coord[0][1] == y:
                print("Selected points cannot form a bounding box. Select the second point again")
            else:
                coord.append((x, y))
                fig.canvas.mpl_disconnect(cid)
                plt.close('all')
        else:
            raise RuntimeError
        return None
    print("Select two points on the image to specify the region of interest")
    fig = plt.figure()
    plt.imshow(img)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def get_img_patch(img, tl_x, tl_y, br_x, br_y):
    dim = len(np.shape(img))
    if dim == 3:
        img_patch = img[tl_y:br_y, tl_x:br_x, :]
    elif dim == 2:
        img_patch = img[tl_y:br_y, tl_x:br_x]
    else:
        print("Input is not an img")
        raise RuntimeError
    return img_patch


def main():
    left_path = os.path.join("imgs/left", "%s.png" % args.file_name)
    right_path = os.path.join("imgs/right", "%s.png" % args.file_name)
    pred_path = os.path.join("imgs/pred_depth", args.model, "%s.npy" % args.file_name)
    gt_path = os.path.join("imgs/gt_depth", "%s%s" % (args.file_name, args.gt_d_ext))

    left_im = cv2.imread(left_path)
    right_im = cv2.imread(right_path)
    pred = np.load(pred_path)
    if args.gt_d_ext == ".csv":
        gt = np.genfromtxt(gt_path, delimiter=',')
    elif args.gt_d_ext == ".exr":
        gt = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        gt = gt[:, :, 1]
    else:
        raise RuntimeError

    gt_depth, gt_disp, pred_depth, pred_disp, depth_mask, disp_mask = disp_depth_conversion(pred, gt)
    sgbm_depth, sgbm_disp = SGBM_disp_depth(left_im, right_im)

    get_ROI_coord(left_im)
    TL_x = min(coord[0][0], coord[1][0])
    TL_y = min(coord[0][1], coord[1][1])
    BR_x = max(coord[0][0], coord[1][0])
    BR_y = max(coord[0][1], coord[1][1])

    rgb_patch = get_img_patch(left_im, TL_x, TL_y, BR_x, BR_y)
    gt_depth_patch = get_img_patch(gt_depth, TL_x, TL_y, BR_x, BR_y)
    gt_disp_patch = get_img_patch(gt_disp, TL_x, TL_y, BR_x, BR_y)
    pred_depth_patch = get_img_patch(pred_depth, TL_x, TL_y, BR_x, BR_y)
    pred_disp_patch = get_img_patch(pred_disp, TL_x, TL_y, BR_x, BR_y)
    sgbm_depth_patch = get_img_patch(sgbm_depth, TL_x, TL_y, BR_x, BR_y)
    sgbm_disp_patch = get_img_patch(sgbm_disp, TL_x, TL_y, BR_x, BR_y)
    depth_mask_patch = get_img_patch(depth_mask, TL_x, TL_y, BR_x, BR_y)
    disp_mask_patch = get_img_patch(disp_mask, TL_x, TL_y, BR_x, BR_y)

    l1, rmse, a1, a2, a3 = depth_error(gt_depth_patch, pred_depth_patch, depth_mask_patch)
    epe, bad3 = disp_error(gt_disp_patch, pred_disp_patch, disp_mask_patch)

    sgbm_l1, sgbm_rmse, sgbm_a1, sgbm_a2, sgbm_a3 = depth_error(gt_depth_patch, sgbm_depth_patch, depth_mask_patch)
    sgbm_epe, sgbm_bad3 = disp_error(gt_disp_patch, sgbm_disp_patch, disp_mask_patch)

    print("Depth error for region of interest is: ")
    print("StereoNet: L1=%.4f, RMSE=%.4f, (delta<1.25)=%.4f, (delta<1.25^2)=%.4f, (delta<1.25^3)=%.4f" % (l1, rmse, a1, a2, a3))
    print("SGBM: L1=%.4f, RMSE=%.4f, (delta<1.25)=%.4f, (delta<1.25^2)=%.4f, (delta<1.25^3)=%.4f" % (sgbm_l1, sgbm_rmse, sgbm_a1, sgbm_a2, sgbm_a3))
    print("Disparity error for region of interest is: ")
    print("StereoNet: EPE=%.4f, Bad3 = %.4f" % (epe, bad3))
    print("SGBM: EPE=%.4f, Bad3 = %.4f" % (sgbm_epe, sgbm_bad3))
    
    plt.figure()
    plt.subplot(332)
    plt.title("Left IR")
    plt.imshow(rgb_patch)
    plt.subplot(334)
    plt.title("GT Disp")
    plt.imshow(gt_disp_patch, vmin=0, vmax=args.max_disp)
    plt.subplot(335)
    plt.title("StereoNet Disp")
    plt.imshow(pred_disp_patch, vmin=0, vmax=args.max_disp)
    plt.subplot(336)
    plt.title("SGBM Disp")
    plt.imshow(sgbm_disp_patch, vmin=0, vmax=args.max_disp)
    plt.subplot(337)
    plt.title("GT Depth")
    plt.imshow(gt_depth_patch, vmin=0, vmax=args.max_depth)
    plt.subplot(338)
    plt.title("StereoNet Depth")
    plt.imshow(pred_depth_patch, vmin=0, vmax=args.max_depth)
    plt.subplot(339)
    plt.title("SGBM Depth")
    plt.imshow(sgbm_depth_patch, vmin=0, vmax=args.max_depth)
    plt.show()


if __name__ == "__main__":
    main()
