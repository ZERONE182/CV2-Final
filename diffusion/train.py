from xunet import XUNet

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from tqdm import tqdm
from einops import rearrange
import time

from SRNdataset import dataset, MultiEpochsDataLoader
from tensorboardX import SummaryWriter
import os
import utils
import argparse


def main(args):
    d = dataset('train', path=args.data_path, picklefile=args.pickle_path, imgsize=args.image_size)
    d_val = dataset('val', path=args.data_path, picklefile=args.pickle_path, imgsize=args.image_size)

    loader = MultiEpochsDataLoader(d, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True,
                                   num_workers=args.num_workers)
    loader_val = DataLoader(d_val, batch_size=args.batch_size,
                            shuffle=True, drop_last=True,
                            num_workers=args.num_workers)

    model = XUNet(H=args.image_size, W=args.image_size, ch=128)
    model = torch.nn.DataParallel(model)
    model.to(utils.dev())

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    if args.transfer == "":
        now = './results/shapenet_SRN_car/' + str(int(time.time()))
        writer = SummaryWriter(now)
        step = 0
    else:
        print('transferring from: ', args.transfer)

        ckpt = torch.load(os.path.join(args.transfer, 'latest.pt'))

        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])

        now = args.transfer
        writer = SummaryWriter(now)
        step = ckpt['step']
    train(model, optimizer, loader, loader_val, writer, now, step, args)


def warmup(optimizer, step, last_step, last_lr):
    if step < last_step:
        optimizer.param_groups[0]['lr'] = step / last_step * last_lr

    else:
        optimizer.param_groups[0]['lr'] = last_lr


def train(model, optimizer, loader, loader_val, writer, now, step, args):
    a = 1
    for e in range(args.num_epochs):
        print(f'starting epoch {e}')

        for img, R, T, K in tqdm(loader):
            warmup(optimizer, step, args.warmup_step / args.batch_size, args.lr)

            B = img.shape[0]

            optimizer.zero_grad()

            logsnr = utils.logsnr_schedule_cosine(torch.rand((B,)))

            loss = utils.p_losses(model, img=img, R=R, T=T, K=K, logsnr=logsnr, loss_type="l2",
                                  cond_prob=0.1)
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step=step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step=step)

            if step % args.verbose_interval == 0:
                print("Loss:", loss.item())

            if step % args.validation_interval == 900:
                validation(model, loader_val, writer, step, args.batch_size)

            if step == int(args.warmup_step / args.batch_size):
                torch.save({'optim': optimizer.state_dict(), 'model': model.state_dict(), 'step': step},
                           now + f"/after_warmup.pt")

            step += 1

        if e % args.save_interval == 0:
            torch.save({'optim': optimizer.state_dict(), 'model': model.state_dict(), 'step': step, 'epoch': e},
                       now + f"/latest.pt")


def validation(model, loader_val, writer, step, batch_size=8):
    model.eval()
    with torch.no_grad():
        ori_img, R, T, K = next(iter(loader_val))
        # w = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(16)
        w = torch.tensor([3.0] * batch_size)
        img = utils.sample(model, img=ori_img, R=R, T=T, K=K, w=w)

        img = rearrange(((img[-1].clip(-1, 1) + 1) * 127.5).astype(np.uint8),
                        "(b a) c h w -> a c h (b w)",
                        a=8, b=16)

        gt = rearrange(((ori_img[:, 1] + 1) * 127.5).detach().cpu().numpy().astype(np.uint8),
                       "(b a) c h w -> a c h (b w)", a=8, b=16)
        cd = rearrange(((ori_img[:, 0] + 1) * 127.5).detach().cpu().numpy().astype(np.uint8),
                       "(b a) c h w -> a c h (b w)", a=8, b=16)

        fi = np.concatenate([cd, gt, img], axis=2)
        for i, ww in enumerate(range(8)):
            writer.add_image(f"train/{ww}", fi[i], step)

    print('image sampled!')
    writer.flush()
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/SRN/cars_train")
    parser.add_argument('--pickle_path', type=str, default="../data/cars.pickle")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--transfer', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--warmup_step', type=int, default=10000000)
    parser.add_argument('--verbose_interval', type=int, default=500)
    parser.add_argument('--validation_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--save_path', type=str, default="./results")
    opts = parser.parse_args()
    main(opts)
