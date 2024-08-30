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


class Custom3DDataset(BaseDataset):
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
        # gt_img_path = [path for path in gt_img_path if "[No_Face]" not in path]
        gt_img_path = sorted(gt_img_path)
        # gt_img_path = sorted(
        #     gt_img_path, key=lambda x: int(x.split("/")[-1].split(".")[0])
        # )
        # gt_img_path = gt_img_path

        gt_depth_path = glob.glob(self.root + "/depths/*.npy")
        gt_depth_path = sorted(gt_depth_path)
        # gt_depth_path = sorted(
        #     gt_depth_path, key=lambda x: int(x.split("/")[-1].split(".")[0])
        # )

        # gt_nerf_path = glob.glob(self.root + "/vertex_textures/*.png")
        # # gt_nerf_path = glob.glob(self.root + "/uv_maps/*.png")
        # # gt_img_path = [path for path in gt_img_path if "[No_Face]" not in path]
        # gt_nerf_path = sorted(gt_nerf_path)
        # gt_nerf_path = sorted(
        #     gt_nerf_path, key=lambda x: int(x.split("/")[-1].split(".")[0])
        # )
        # gt_nerf_path = gt_nerf_path[:2100]
        gt_mesh_path = glob.glob(self.root + "/texture_meshes/*.png")
        # gt_img_path = [path for path in gt_img_path if "[No_Face]" not in path]
        gt_mesh_path = sorted(gt_mesh_path)
        gt_img_path = gt_img_path[:len(gt_mesh_path)]

        gt_unmesh_path = glob.glob(self.root + "/untexture_meshes/*.png")
        # gt_img_path = [path for path in gt_img_path if "[No_Face]" not in path]
        gt_unmesh_path = sorted(gt_unmesh_path)
        
        gt_normal_path = glob.glob(self.root + "/normals/*.png")
        gt_normal_path = sorted(gt_normal_path)

        # gt_feature_path = glob.glob(self.root+'/FaceMesh/*.png')
        # gt_feature_path = sorted(gt_feature_path, key = lambda x: int(x.split('/')[-1].split('.')[0]))
        # gt_feature_path = [path.replace('Origin', 'FaceMesh') for path in gt_img_path]
        # gt_feature_path = [path.replace('origin', 'facemesh_shoulder') for path in gt_img_path]
        # gt_feature_path = [path.replace("origin", "output") for path in gt_img_path]
        split = int(len(gt_img_path) * (80 / 100))
        # split = 0
        # split_t = int(len(gt_img_path) * (70 / 100))

        self.first_image_path = gt_img_path[0]

        if self.opt.isTrain and not self.opt.is_val:
            self.gt_img_path = gt_img_path[:split]
            self.gt_depth_path = gt_depth_path[:split]
            # self.gt_nerf_path = gt_nerf_path[:split]
            self.gt_mesh_path = gt_mesh_path[:split]
            self.gt_normal_path = gt_normal_path[:split]
            self.gt_unmesh_path = gt_unmesh_path[:split]
        # if self.opt.isTrain and not self.opt.is_val:
        #     self.gt_img_path = gt_img_path[:split_t]
        #     self.gt_feature_path = gt_feature_path[:split_t]
        # elif self.opt.isTrain and self.opt.is_val:
        #     self.gt_img_path = gt_img_path[split_t:split]
        #     self.gt_feature_path = gt_feature_path[split_t:split]
        else:
            self.gt_img_path = gt_img_path[split:]
            self.gt_depth_path = gt_depth_path[split:]
            # self.gt_nerf_path = gt_nerf_path[split:]
            self.gt_mesh_path = gt_mesh_path[split:]
            self.gt_normal_path = gt_normal_path[split:]
            self.gt_unmesh_path = gt_unmesh_path[split:]

        self.transform = A.Compose(
            [
                A.Resize(self.opt.loadSize, self.opt.loadSize),
                AP.transforms.ToTensor(
                    normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
                ),
            ]
        )
        # pdb.set_trace()

        # tmp = []
        # for j in range(4):
        #     output = imread(self.root+'/candidates/normalized_full_{}.png'.format(j))
        #     # output = self.transform(output)
        #     output = AP.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)})(image=output)['image']
        #     tmp.append(output)
        # self.cand = torch.cat(tmp)

        # print(len(self.gt_img_path))

    def __len__(self):
        return len(self.gt_img_path)

    def name(self):
        return "Custom3DDataset"

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

        gt_file_path_0 = self.first_image_path
        gt_image_0 = imread(gt_file_path_0)[..., :3]
        gt_image_0 = resize(gt_image_0, (self.opt.loadSize, self.opt.loadSize))

        gt_file_path = self.gt_img_path[target_ind]
        gt_image = imread(gt_file_path)[..., :3]
        gt_image = resize(gt_image, (self.opt.loadSize, self.opt.loadSize))
        # depth_file_path = self.gt_depth_path[target_ind]
        # depth_map = imread(depth_file_path)
        # depth_map = resize(depth_map, (self.opt.loadSize, self.opt.loadSize))
        # depth_map = np.array(depth_map).astype(np.float32) / 1000.
        # nerf_file_path = self.gt_nerf_path[target_ind]
        # nerf_image = imread(nerf_file_path)[..., :3]
        # nerf_image = resize(nerf_image, (self.opt.loadSize, self.opt.loadSize))
        #mesh
        # mesh_file_path = self.gt_mesh_path[target_ind]
        # mesh_image = imread(mesh_file_path)[..., :3]
        # mesh_image = resize(mesh_image, (self.opt.loadSize, self.opt.loadSize))

        unmesh_file_path = self.gt_unmesh_path[target_ind]
        unmesh_image = imread(unmesh_file_path)[..., :3]
        unmesh_image = resize(unmesh_image, (self.opt.loadSize, self.opt.loadSize))

        #normal
        # normal_file_path = self.gt_normal_path[target_ind]
        # normal_image = imread(normal_file_path)[..., :3]
        # normal_image = resize(normal_image, (self.opt.loadSize, self.opt.loadSize))
        #depth 
        # depth_file_path = self.gt_depth_path[target_ind]
        # depth_map = np.load(depth_file_path) / 10.
        # pdb.set_trace()

        # transforms
        # gt_image = self.transform(gt_image)
        gt_image_0 = AP.transforms.ToTensor(
            normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        )(image=gt_image_0)["image"]

        gt_image = AP.transforms.ToTensor(
            normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        )(image=gt_image)["image"]
        # nerf_image = AP.transforms.ToTensor(
        #     normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        # )(image=nerf_image)["image"]
        # depth_map = depth_map.reshape(1, self.opt.loadSize, self.opt.loadSize)
        # depth_map = torch.from_numpy(depth_map)
        # mesh_image = AP.transforms.ToTensor(
        #     normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        # )(image=mesh_image)["image"]

        unmesh_image = AP.transforms.ToTensor(
            normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        )(image=unmesh_image)["image"]

        # out = AP.transforms.ToTensor(
        #     normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        # )(image=out)["image"]

        # normal_image = AP.transforms.ToTensor(
        #     normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        # )(image=normal_image)["image"]
        # depth_map = torch.from_numpy(depth_map).permute(2,0,1)
        

        # extra
        # cand_image = self.cand
        tmp = []
        for i in range(-2, 3):
            if i == 0:
                continue
            # output_depth = imread(self.gt_depth_path[target_ind - i])
            # output_depth = resize(output_depth, (self.opt.loadSize, self.opt.loadSize))
            # output_depth = np.array(output_depth).astype(np.float32) / 1000.
            # output_depth = output_depth.reshape(1, self.opt.loadSize, self.opt.loadSize)
            # output_depth = torch.from_numpy(output_depth)
            # output_nerf = imread(self.gt_nerf_path[target_ind - i])[..., :3]
            # output_nerf = resize(output_nerf, (self.opt.loadSize, self.opt.loadSize))
            # output_nerf = AP.transforms.ToTensor(
            #     normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
            # )(image=output_nerf)["image"]
            output_mesh = imread(self.gt_unmesh_path[target_ind - i])[..., :3]
            output_mesh = resize(output_mesh, (self.opt.loadSize, self.opt.loadSize))
            output_mesh = AP.transforms.ToTensor(
                normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
            )(image=output_mesh)["image"]
            # tmp.append(torch.cat([output_depth, output_nerf]))
            tmp.append(torch.cat([
                                #   output_depth, 
                                #   output_nerf, 
                                  output_mesh,
                                  ]))
        cand_image = torch.cat(tmp)
        # print(feature_map.min())
        # print(feature_map.max())

        return_list = {
            # "feature_map": torch.cat([depth_map, nerf_image]),
            "feature_map": torch.cat([
                                    #   mesh_image,
                                    #   depth_map,
                                    #   normal_image,
                                    unmesh_image,
                                    gt_image_0,
                                      ]),
            "cand_image": cand_image,
            "tgt_image": gt_image,
            "weight_mask": gt_image,
            "cand_feature": gt_image,
            "mesh": unmesh_image,
            # "depth": depth_map,
            # "normal": normal_image,
            # "mask": mask,
        }

        return return_list
