""" Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Base class for our video dataset
"""

from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np

import torchvision.transforms as transforms

import scipy.io as sio
from .base_datasets import BaseVideoDataset
from .transforms import TPSWarp
from random import Random

class _50SaladsDataset(BaseVideoDataset):
    """
    Base dataset class for all video-type datasets in landmark learning
    """
    def __init__(self, args, partition):
        super(_50SaladsDataset, self).__init__(args, partition)

        self.TPSWarp = TPSWarp([self.img_size, self.img_size], 10, 10, 10,
                        rot_range=[args['rot_lb'], args['rot_ub']],
                        trans_range=[args['trans_lb'], args['trans_ub']],
                        scale_range=[args['scale_lb'], args['scale_ub']],
                        nonlinear_pert_range=[-2, 2],
                        append_offset_channels=True)

    def setup_frame_array(self, args, partition):
        """
        Implement this function to setup the cummulative array
        cummulative array should have N+1 bins, where N is the number of videos
        first bin should be 0
        last bin should be the total number of frames in the dataset
        Also use this function to setup any dataset-specific fields
        """
        # load the annotations file
        f = open(os.path.join(self.dataset_path, "annotations.txt"), "r")
        lines = f.readlines()
        Random(0).shuffle(lines)

        self.idx_to_filename = []
        self.idx_to_filename_pairs = []

        flag = None
        if partition == 'train':
            flag = '1'
        elif partition == 'validation':
            flag = '2'

        train_max = float('inf')
        if self.n_training_max:
            train_max = self.n_training_max

        if self.randomize_pairs:
            train_max *= 2

        i = 0
        for l in lines:
            name, subset = l.split()
            if subset == flag:
                if i >= train_max and partition == 'train':
                    break

                self.idx_to_filename.append(name)

                i += 1
        
        if self.randomize_pairs:
            for i in range(0, len(self.idx_to_filename)-1, 2):
                self.idx_to_filename_pairs.append((self.idx_to_filename[i], self.idx_to_filename[i+1]))

        # first bin is 0
        self.num_frames_array = [0]
        frac = 1
        if partition == 'val' or partition == 'validation':
            frac = args['val_frac']
        # truncate validation if frac is specified
        dset_size = len(self.idx_to_filename)
        if self.randomize_pairs:
            dset_size = len(self.idx_to_filename_pairs)

        self.num_frames_array.append(int(dset_size*frac))

        return self.num_frames_array

    def __len__(self):
        """
        returns length of dataset (total number of frames)
        """
        return self.num_frames_array[-1]

    def get_frame_index(self, global_idx):
        """maps global frame index to video index and local video frame index
        """
        return 0, global_idx

    def process_batch(self, vid_idx, img_idx):
        """
        implement this function
        extracts the requisite frames from the dataset
        returns a dictionary that must include entries in the required_keys variable in __getitem__
        """
        if self.randomize_pairs:
            filename_1, filename_2 = self.idx_to_filename_pairs[img_idx]
        else:
            filename_1 = self.idx_to_filename[img_idx]
            filename_2 = filename_1
        img_1 = os.path.join(self.dataset_path, filename_1 + '_0.jpg')
        img_2 = os.path.join(self.dataset_path, filename_2 + '_1.jpg')

        img_a = Image.open(img_1).convert('RGB')
        img_temporal = Image.open(img_2).convert('RGB')

        # randomly flip
        if np.random.rand() <= self.flip_probability:
            # flip both images
            img_a = transforms.functional.hflip(img_a)
            img_temporal = transforms.functional.hflip(img_temporal)

        img_temporal = self.to_tensor(self.resize(img_temporal))
        img_temporal = self.normalize(img_temporal)
        
        img_a_color_jittered, img_a_warped, img_a_warped_offsets, target=self.construct_color_warp_pair(img_a, flip = np.random.rand() <= self.invert_probability)

        return {'input_a': img_a_color_jittered, 'input_b': img_a_warped,
                'input_temporal': img_temporal, 'target': target,
                'imname': filename_1 + '_0.jpg',
                'warping_tps_params': img_a_warped_offsets, 'gt_kpts': []}
