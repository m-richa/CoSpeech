from operator import gt
import os
from datasets.base_dataset import BaseDataset
import os.path
from pathlib import Path
import torch
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from PIL import Image, ImageMorph, ImageFilter
import bisect
import numpy as np
import io
import cv2
import h5py
import albumentations as A
import albumentations.pytorch as AP
import glob
import PIL
import torchvision.transforms as transforms
from skimage.color import rgb2gray
import pdb


class CustomDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)
        self.state = "Train" if self.opt.isTrain else "Test"
        self.dataset_name = opt.dataset_names[0]
        self.opt = opt
        # only load in train mode
        self.dataset_root = os.path.join(self.root, self.dataset_name)

        gt_img_path = glob.glob(self.root + "/images/*.png")
        gt_img_path = [path for path in gt_img_path if "[No_Face]" not in path]
        gt_img_path = sorted(
            gt_img_path, key=lambda x: int(x.split("/")[-1].split(".")[0])
        )
        gt_img_path = gt_img_path
        # gt_feature_path = glob.glob(self.root+'/FaceMesh/*.png')
        # gt_feature_path = sorted(gt_feature_path, key = lambda x: int(x.split('/')[-1].split('.')[0]))
        # gt_feature_path = [path.replace('Origin', 'FaceMesh') for path in gt_img_path]
        # gt_feature_path = [path.replace('origin', 'facemesh_shoulder') for path in gt_img_path]
        gt_feature_path = [path.replace("images", "keypoints") for path in gt_img_path]
        split = int(len(gt_img_path) * (80 / 100))
        split_t = int(len(gt_img_path) * (70 / 100))

        # pdb.set_trace()
        if self.opt.isTrain and not self.opt.is_val:
            self.gt_img_path = gt_img_path[:split]
            self.gt_feature_path = gt_feature_path[:split]
        # if self.opt.isTrain and not self.opt.is_val:
        #     self.gt_img_path = gt_img_path[:split_t]
        #     self.gt_feature_path = gt_feature_path[:split_t]
        # elif self.opt.isTrain and self.opt.is_val:
        #     self.gt_img_path = gt_img_path[split_t:split]
        #     self.gt_feature_path = gt_feature_path[split_t:split]
        else:
            self.gt_img_path = gt_img_path[split:]
            self.gt_feature_path = gt_feature_path[split:]

        self.transform = A.Compose(
            [
                A.Resize(self.opt.loadSize, self.opt.loadSize),
                AP.transforms.ToTensor(
                    normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
                ),
            ]
        )

        # tmp = []
        # for j in range(4):
        #     output = imread(self.root+'/candidates/normalized_full_{}.png'.format(j))
        #     # output = self.transform(output)
        #     output = AP.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)})(image=output)['image']
        #     tmp.append(output)
        # self.cand = torch.cat(tmp)

        # print(len(self.gt_img_path))
        # print(len(self.gt_feature_path))

    def __len__(self):
        return len(self.gt_img_path)

    def name(self):
        return "CustomDataset"

    def __getitem__(self, ind):

        data_index = ind * self.opt.frame_jump + np.random.randint(self.opt.frame_jump)
        target_ind = data_index
        # extra
        if target_ind < 2:
            target_ind += 2
        elif target_ind + 2 >= len(self.gt_img_path) - 1:
            target_ind -= 2

        # transform
        # self.image_transforms = A.Compose([A.Resize(np.int32(self.opt.loadSize), np.int32(self.opt.loadSize))])
        # try:
        gt_file_path = self.gt_img_path[target_ind]
        gt_image = imread(gt_file_path)
        gt_image = resize(gt_image, (self.opt.loadSize, self.opt.loadSize))
        feature_file_path = self.gt_feature_path[target_ind]
        feature_map = imread(feature_file_path)
        feature_map = resize(feature_map, (self.opt.loadSize, self.opt.loadSize))
        feature_map = rgb2gray(feature_map)
        feature_map = np.array(feature_map).astype(np.float32)
        # pdb.set_trace()

        # make edge maps thicker for vis
        mask = Image.open(feature_file_path).convert("L")
        mask = mask.filter(ImageFilter.FIND_EDGES)
        mask = mask.filter(ImageFilter.MaxFilter)
        mask = mask.resize((self.opt.loadSize, self.opt.loadSize), Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)
        mask = mask.reshape(1, self.opt.loadSize, self.opt.loadSize)
        # transforms
        # gt_image = self.transform(gt_image)
        gt_image = AP.transforms.ToTensor(
            normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        )(image=gt_image)["image"]
        feature_map = feature_map.reshape(1, self.opt.loadSize, self.opt.loadSize)
        feature_map = torch.from_numpy(feature_map)

        # extra
        # cand_image = self.cand
        tmp = []
        for i in range(-2, 3):
            if i == 0:
                continue
            output = imread(self.gt_feature_path[target_ind - i])
            output = resize(output, (self.opt.loadSize, self.opt.loadSize))
            output = rgb2gray(output)
            output = np.array(output).astype(np.float32)
            output = output.reshape(1, self.opt.loadSize, self.opt.loadSize)
            output = torch.from_numpy(output)
            tmp.append(output)
        cand_image = torch.cat(tmp)
        # print(feature_map.min())
        # print(feature_map.max())

        return_list = {
            "feature_map": feature_map,
            "cand_image": cand_image,
            "tgt_image": gt_image,
            "weight_mask": gt_image,
            "cand_feature": gt_image,
            "mask": mask,
        }

        return return_list
