# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:12:50 2020

@author: giles
"""
from __future__ import print_function, division
import os
import pickle
import random

import torch
import pandas as pd
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import RotationMatrix6D, pose_from_filename


class ImagePairDataset(Dataset):
    def __init__(self, dir_path):
        self.name_pair = self.dataNamePair(dir_path)
        self.length = len(self.name_pair)
        self.path = dir_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = {}

        filename_list = self.name_pair[index]
        img1, img2, poseA, poseB = self.imgPreprocess(filename_list)
        groundtruth_pose = self.groundTruthTensor(filename_list)
        input1 = self.catImgPose(img1, poseA)

        data = {'input1': input1,
                'img1': img1,
                'img2': img2,
                'poseA': poseA,
                'poseB': poseB,
                'groundtruth_pose': groundtruth_pose
                }

        return data

    def dataNamePair(self, datadir):
        im_list = os.listdir(datadir)
        name_list = []
        index_list = []
        current_list = []
        current = 0
        for i in range(len(im_list)):
            name = im_list[i]
            split = name.split("_")
            if split[0] not in name_list:
                index_list.append(current_list)
                current_list = []
                current = split[0]
                name_list.append(split[0])
            current_list.append(i)
            if i == len(im_list) - 1:
                index_list.append(current_list)
        index_list.pop(0)

        name_pair = []
        for cur_list in index_list:
            length = len(cur_list)
            for j in range(length - 1):  # index1 = cur_list[j]
                for k in range(length - 1 - j):  # index2 = cur_list[k]
                    name1 = im_list[cur_list[j]]
                    name2 = im_list[cur_list[k + j + 1]]
                    name_pair.append([name1, name2])

        return name_pair

    def imgPreprocess(self, filename_list):  # normalize

        trans = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        tensor = torch.tensor(())
        image1 = tensor.new_zeros((1, 3, 128, 128))
        image2 = tensor.new_zeros((1, 3, 128, 128))
        poseB = tensor.new_zeros((1, 12))
        poseA = tensor.new_zeros((1, 12))

        image1[0] = trans(Image.open(os.path.join(self.path, filename_list[0][0])))
        image2[0] = trans(Image.open(os.path.join(self.path, filename_list[0][1])))

        R1 = pose_from_filename(os.path.splitext(image1)[0])
        poseA[0] = torch.from_numpy(np.reshape(R1, (12, 1)))[:, 0]
        R2 = pose_from_filename(os.path.splitext(image2)[0])
        poseB[0] = torch.from_numpy(np.reshape(R2, (12, 1)))[:, 0]

        return image1, image2, poseA, poseB

    def groundTruthTensor(self, filename_list):
        gt = np.zeros((len(filename_list), 9))
        for i in range(len(filename_list)):
            gt[i, :] = RotationMatrix6D(filename_list[i][0], filename_list[i][1])

        return torch.from_numpy(gt)

    def catImgPose(self, img, pose):
        pose = pose[:, :, None, None]
        pose = pose.repeat(1, 1, 128, 128)
        input1 = torch.cat((img, pose), dim=1)

        return input1


class SRNDataset(Dataset):

    def __init__(self, split, path='./data/SRN/cars_train', pickle_file='./data/cars.pickle', imgsize=128):
        self.imgsize = imgsize
        self.path = path
        super().__init__()
        self.pickle_file = pickle.load(open(pickle_file, 'rb'))

        all_the_vid = sorted(list(self.pickle_file.keys()))

        random.seed(0)
        random.shuffle(all_the_vid)
        self.split = split
        if split == 'train':
            self.ids = all_the_vid[:int(len(all_the_vid) * 0.9)]
        else:
            self.ids = all_the_vid[int(len(all_the_vid) * 0.9):]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        item = self.ids[idx]

        intrinsics_filename = os.path.join(self.path, item, 'intrinsics', self.pickle_file[item][0][:-4] + ".txt")

        indices = random.sample(self.pickle_file[item], k=2)

        imgs = []
        poses = []
        for i in indices:
            img_filename = os.path.join(self.path, item, 'rgb', i)
            img = Image.open(img_filename)
            if self.imgsize != 128:
                img = img.resize((self.imgsize, self.imgsize))
            img = np.array(img) / 255 * 2 - 1

            img = img.transpose(2, 0, 1)[:3].astype(np.float32)
            imgs.append(img)

            pose_filename = os.path.join(self.path, item, 'pose', i[:-4] + ".txt")
            with open(pose_filename) as file:
                # file =open(pose_filename).read()
                pose = np.array(file.read().strip().split()).astype(float).reshape((4, 4))
                # pose_filename.close()
                pose = pose[:3, :].reshape((12, ))
                poses.append(pose)

        imgs = np.stack(imgs, 0)
        poses = np.stack(poses, 0)

        return imgs.astype(np.float32), poses.astype(np.float32)
