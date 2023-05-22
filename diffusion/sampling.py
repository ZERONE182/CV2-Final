from xunet import XUNet

import torch
import numpy as np

from tqdm import tqdm
import os
import glob
from PIL import Image
import random
import utils

import argparse


def dataPreparation(args):
    data_imgs = []
    data_Rs = []
    data_Ts = []
    img_filename = None
    for img_filename in sorted(glob.glob(args.target + "/rgb/*.png")):
        img = Image.open(img_filename)
        img = img.resize((args.image_size, args.image_size))
        img = np.array(img) / 255 * 2 - 1

        img = img.transpose(2, 0, 1)[:3].astype(np.float32)
        data_imgs.append(img)

        pose_filename = os.path.join(args.target, 'pose', os.path.basename(img_filename)[:-4] + ".txt")
        pose = np.array(open(pose_filename).read().strip().split()).astype(float).reshape((4, 4))

        data_Rs.append(pose[:3, :3])
        data_Ts.append(pose[:3, 3])
    data_K = np.array(open(os.path.join(args.target, 'intrinsics',
                                        os.path.basename(img_filename)[:-4]
                                        + ".txt")).read().strip().split()).astype(float).reshape((3, 3))
    data_K = torch.tensor(data_K)
    return data_imgs, data_Rs, data_Ts, data_K


def main(args):
    data_imgs, data_Rs, data_Ts, data_K = dataPreparation(args)

    model = XUNet(H=args.image_size, W=args.image_size, ch=128)
    model = torch.nn.DataParallel(model)
    model.to(utils.dev())

    ckpt = torch.load(args.model, map_location=torch.device(utils.dev()))
    model.load_state_dict(ckpt['model'])

    w = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    b = w.shape[0]
    record = [[data_imgs[0][None].repeat(b, axis=0),
               data_Rs[0],
               data_Ts[0]]]

    data_K = data_K.repeat(b, 1, 1)

    result_dir = os.path.join(args.result_dir, os.path.basename(args.target))

    os.makedirs(os.path.join(result_dir, '0'), exist_ok=True)
    Image.fromarray(((data_imgs[0].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)).save(
        os.path.join(result_dir, '0', 'gt.png'))

    with torch.no_grad():
        step = 1
        for gt, R, T in tqdm(zip(data_imgs[1:], data_Rs[1:], data_Ts[1:]), total=len(data_imgs[1:]), desc='view loop',
                             position=0):

            R = torch.tensor(R)
            T = torch.tensor(T)

            img = sample(model, record=record, target_R=R, target_T=T, K=data_K, w=w)

            record.append([img, R.cpu().numpy(), T.cpu().numpy()])
            os.makedirs(os.path.join(result_dir, f'{step}'), exist_ok=True)
            Image.fromarray(((gt.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)).save(
                os.path.join(result_dir, f'{step}', 'gt.png'))
            for i in w:
                Image.fromarray(((img[i].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)).save(
                    os.path.join(result_dir, f'{step}', f'{i}.png'))
            step += 1


@torch.no_grad()
def sample(model, record, target_R, target_T, K, w, timesteps=256):
    b = w.shape[0]
    img = torch.randn_like(torch.tensor(record[0][0]))

    logsnrs = utils.logsnr_schedule_cosine(torch.linspace(1., 0., timesteps + 1)[:-1])
    logsnr_nexts = utils.logsnr_schedule_cosine(torch.linspace(1., 0., timesteps + 1)[1:])

    for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts), total=len(logsnrs), desc='diffusion loop', position=1,
                                    leave=False):  # [1, ..., 0] = size is 257
        condition_img, condition_R, condition_T = random.choice(record)
        condition_img = torch.tensor(condition_img)
        condition_R = torch.tensor(condition_R)
        condition_T = torch.tensor(condition_T)

        R = torch.stack([condition_R, target_R], 0)[None].repeat(b, 1, 1, 1)
        T = torch.stack([condition_T, target_T], 0)[None].repeat(b, 1, 1)
        img = utils.p_sample(model,
                             z=img,
                             x=condition_img,
                             R=R,
                             T=T,
                             K=K,
                             logsnr=logsnr, logsnr_next=logsnr_next,
                             w=w)

    return img.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="./trained_model.pt")
    parser.add_argument('--target', type=str, default="./data/SRN/cars_train/a4d535e1b1d3c153ff23af07d9064736")
    parser.add_argument('--result_dir', type=str, default='sample_result')
    parser.add_argument('--image_size', type=int, default=128)
    opts = parser.parse_args()
    main(opts)
