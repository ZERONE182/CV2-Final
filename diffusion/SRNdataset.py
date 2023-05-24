import os
import pickle
import random
import numpy as np
import torch
import torchvision.transforms.functional
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class SRNDataset(Dataset):

    def __init__(self, split, path='./data/SRN/cars_train', pickle_file='./data/cars.pickle', imgsize=128,
                 use_hue_loss=False):
        self.imgsize = imgsize
        self.path = path
        super().__init__()
        self.pickle_file = pickle.load(open(pickle_file, 'rb'))

        all_the_vid = sorted(list(self.pickle_file.keys()))

        random.seed(0)
        random.shuffle(all_the_vid)
        self.use_hue_loss = use_hue_loss
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
        K = np.array(open(intrinsics_filename).read().strip().split()).astype(float).reshape((3, 3))

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
            pose = np.array(open(pose_filename).read().strip().split()).astype(float).reshape((4, 4))
            poses.append(pose)

        imgs = np.stack(imgs, 0)
        hue_delta = 0
        if self.split == 'train' and self.use_hue_loss:
            hue_delta = random.random() - 0.5
            adjust_img = torchvision.transforms.functional.adjust_hue(torch.Tensor(imgs[1]), hue_delta)
            imgs[1] = adjust_img.numpy()
        poses = np.stack(poses, 0)
        R = poses[:, :3, :3]
        T = poses[:, :3, 3]

        return imgs, R, T, K, hue_delta

# if __name__ == "__main__":
#
#     from torch.utils.data import DataLoader
#
#     d = SRNDataset('train')
#     dd = d[0]
#
#     for ddd in dd:
#         print(ddd.shape)
