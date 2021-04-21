import os
import torch.utils.data as data
from torchvision import transforms
from PIL import Image


class DepthEstDataset(data.Dataset):
    def __init__(self, model_name):
        super(DepthEstDataset, self).__init__()
        data_path = "imgs/"
        self.left_im_path = os.path.join(data_path, "left")
        self.right_im_path = os.path.join(data_path, "right")
        self.pred_depth_path = os.path.join(data_path, "pred_depth", model_name)
        if not os.path.exists(self.pred_depth_path):
            os.makedirs(self.pred_depth_path)
        self.data_list = os.listdir(self.left_im_path)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        file_name = self.data_list[item]
        left_im = Image.open(os.path.join(self.left_im_path, file_name))
        right_im = Image.open(os.path.join(self.right_im_path, file_name))
        left_im = self.to_tensor(left_im)
        right_im = self.to_tensor(right_im)

        depth_filename = file_name.split('.')[0] + ".npy"
        pred_depth_filename = os.path.join(self.pred_depth_path,
                                           depth_filename)  # path to save the predicted depth file as a np array
        return left_im, right_im, pred_depth_filename
