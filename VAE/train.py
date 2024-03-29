from CVAE import ConditionalVAE

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
import argparse
import cv2


def main(args):
    d = dataset('train', path=args.data_path, picklefile=args.pickle_path, imgsize=args.image_size)
    d_val = dataset('val', path=args.data_path, picklefile=args.pickle_path, imgsize=args.image_size)

    loader = MultiEpochsDataLoader(d, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True,
                                   num_workers=args.num_workers)
    loader_val = DataLoader(d_val, batch_size=args.batch_size,
                            shuffle=True, drop_last=True,
                            num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalVAE(H=args.image_size, W=args.image_size, z_dim=128, n_resolution=3)
    model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    if args.transfer == "":
        if args.exp_name is None:
            now = './results/shapenet_SRN_car/' + str(int(time.time()))
        else:
            now = './results/shapenet_SRN_car/' + args.exp_name
        # now = './results/shapenet_SRN_car/' + "no_nonlinear"
        writer = SummaryWriter(now)
        step = 0
    else:
        print('transferring from: ', args.transfer)

        ckpt = torch.load(os.path.join(args.transfer, 'latest.pt'))

        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])

        if not args.val:
            now = args.transfer
        else:
            now = args.transfer + "val"
        writer = SummaryWriter(now)
        step = ckpt['step']
    if args.val:
        for i in range(120):
            validation(model, loader, writer, step+i, args.batch_size, save_path=now)
    else:
        train(model, optimizer, loader, loader_val, writer, now, step, args)


def warmup(optimizer, step, last_step, last_lr):
    if step < last_step:
        optimizer.param_groups[0]['lr'] = step / last_step * last_lr

    else:
        optimizer.param_groups[0]['lr'] = last_lr


def train(model, optimizer, loader, loader_val, writer, now, step, args):
    a = 1
    freezed = True ## No freezing
    for e in range(args.num_epochs):
        print(f'starting epoch {e}')

        for img, R, T, K in tqdm(loader):
            if not freezed and step > args.freeze_step:
                print('freezing encoder')
                model.module.freeze_encoder()
                freezed = True
            warmup(optimizer, step, args.warmup_step / args.batch_size, args.lr)

            B = img.shape[0]

            optimizer.zero_grad()

            batch = {'x':img[:,0], 'z':img[:,1], 'R': R, 't': T, 'K': K,}

            kld_loss, img_loss = model(batch, None)
            loss = kld_loss + img_loss
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/kld_loss", kld_loss.item(), global_step=step)
            writer.add_scalar("train/img_loss", img_loss.item(), global_step=step)
            writer.add_scalar("train/loss", loss.item(), global_step=step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step=step)

            # if step % args.verbose_interval == 0:
            #     print(f"loss: {loss.item()}, kld loss: {kld_loss.item()}, img loss: {img_loss.item()}")

            if step % args.validation_interval == 0:
                validation(model, loader_val, writer, step, args.batch_size)

            if step == int(args.warmup_step / args.batch_size):
                torch.save({'optim': optimizer.state_dict(), 'model': model.state_dict(), 'step': step},
                           now + f"/after_warmup.pt")

            step += 1

        print(f"loss: {loss.item()}, kld loss: {kld_loss.item()}, img loss: {img_loss.item()}")
        if e % args.save_interval == 0:
            torch.save({'optim': optimizer.state_dict(), 'model': model.state_dict(), 'step': step, 'epoch': e},
                       now + f"/latest.pt")


def validation(model, loader_val, writer, step, batch_size=8, device='cuda', save_path=None):
    # TODO: Add image writer
    model.eval()
    with torch.no_grad():
        ori_img, R, T, K = next(iter(loader_val))
        
        batch = {'x':ori_img[:,0].to(device), 'z':ori_img[:,1].to(device), 'R': R.to(device), 't': T.to(device), 'K': K.to(device),}
        input_img = ori_img[:, 0].detach().cpu().numpy()
        input_img = ((input_img.clip(-1, 1)+1)*127.5).astype(np.uint8)
        gt_img = ori_img[:, 1].detach().cpu().numpy()
        gt_img = ((gt_img.clip(-1, 1)+1)*127.5).astype(np.uint8)
        pred_img, recon_img = model.module.eval_img(batch, None)
        pred_img = pred_img.detach().cpu().numpy()
        recon_img = recon_img.detach().cpu().numpy()

        writer.add_scalar("val/recon_min", recon_img.min(), global_step=step)
        writer.add_scalar("val/recon_max", recon_img.max(), global_step=step)
        writer.add_scalar("val/gen_min", pred_img.min(), global_step=step)
        writer.add_scalar("val/gen_max", pred_img.max(), global_step=step)

        pred_img = ((pred_img.clip(-1, 1)+1)*127.5).astype(np.uint8)
        recon_img = ((recon_img.clip(-1, 1)+1)*127.5).astype(np.uint8)

        writer.add_images(f"train/input", input_img, step)
        writer.add_images(f"train/gt", gt_img, step)
        writer.add_images(f"train/pred",pred_img, step)
        writer.add_images(f"train/recon", recon_img, step)

    # save image locally
    if save_path is not None:
        save_input = np.transpose(np.copy(input_img[0]), (1, 2, 0))
        save_recon = np.transpose(np.copy(recon_img[0]), (1, 2, 0))
        save_gt = np.transpose(np.copy(gt_img[0]), (1, 2, 0))
        save_pred = np.transpose(np.copy(pred_img[0]), (1, 2, 0))

        save_input = cv2.cvtColor(save_input, cv2.COLOR_RGB2BGR)
        save_recon = cv2.cvtColor(save_recon, cv2.COLOR_RGB2BGR)
        save_gt = cv2.cvtColor(save_gt, cv2.COLOR_RGB2BGR)
        save_pred = cv2.cvtColor(save_pred, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(save_path, str(step) + "input.png"), save_input)
        cv2.imwrite(os.path.join(save_path, str(step) + "recon.png"), save_recon)
        cv2.imwrite(os.path.join(save_path, str(step) + "gt.png"), save_gt)
        cv2.imwrite(os.path.join(save_path, str(step) + "pred.png"), save_pred)

    # print('image sampled!')
    writer.flush()
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/SRN/cars_train")
    parser.add_argument('--pickle_path', type=str, default="../data/cars.pickle")
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--transfer', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--val', action='store_true', default=False)
    
    parser.add_argument('--warmup_step', type=int, default=0)
    parser.add_argument('--freeze_step', type=float, default=5000)

    parser.add_argument('--verbose_interval', type=int, default=500)
    parser.add_argument('--validation_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_path', type=str, default="./results")
    opts = parser.parse_args()
    main(opts)
