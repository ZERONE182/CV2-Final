import torch
import torch.nn as nn
import torch.nn.functional as F
import visu3d
import numpy as np

### TODO: For Prof. Zeng: you can always reconstruct the whole model as you like. All the code below is not tested, error may occur.

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, x):
        B, D = x.shape
        H = W = int(np.sqrt(D / 256))
        return x.view(B, 256, H, W)

class PoseCondition(nn.Module):
    def __init__(self) -> None:
        # TODO: Decide to imitate the method done in XUnet. 
        # Suppose we have pose data R, t, K. (I don't know whether it is paired -> (R1, R2, t1, t1, K1, K2) or just (R, t, K))
        # By using the camera model in visu3d, we can get the ray info for all the pixels in the image. 
        # The ray origin is 3-dim vector and direction is also 3-dim vector. 
        # If we just concate the ray info for all the pixel in the image, we can get a tensor in shape (H, W, 6), 6 can be seen as channel?
        # NeRF PE is applied on ray origin and ray direction. 
        # CNN is also used to change the spatial size to be the same as the downsampled image during VAE processing. 
        super().__init__()


class EncoderBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, input_h:int, input_w:int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 5, 2, 2)
        self.norm = nn.LayerNorm([out_channel, input_h // 2, input_w // 2])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, input_h:int, input_w:int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, 5, 2, 2, output_padding=1)
        self.layer_norm = nn.LayerNorm([out_channel, input_h * 2, input_w * 2])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        return x

class ConditionalVAE(nn.Module):
    def __init__(self, H: int = 128, W: int = 128, z_dim: int = 128, n_resolution: int = 3) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.z_dim = z_dim
        self.n_resolution = n_resolution

        # TODO: The in channel for all the blocks below are wrong, as the pose info needs to be injected.
        self.ec1 = EncoderBlock(3, 32, H, W)
        self.ec2 = EncoderBlock(32, 64, H // 2, W // 2)
        self.ec3 = EncoderBlock(64, 128, H // 4, W // 4)
        self.ec4 = EncoderBlock(128, 256, H // 8, W // 8)

        self.fc1 = nn.Linear(256 * (H // 16) * (W // 16), z_dim) # for mu
        self.fc2 = nn.Linear(256 * (H // 16) * (W // 16), z_dim) # for logvar
        self.fc3 = nn.Linear(z_dim, 256 * (H // 16) * (W // 16)) # for decoder

        self.dc1 = DecoderBlock(256, 128, H // 16, W // 16)
        self.dc2 = DecoderBlock(128, 64, H // 8, W // 8)
        self.dc3 = DecoderBlock(64, 32, H // 4, W // 4)
        self.dc4 = DecoderBlock(32, 3, H // 2, W // 2)

    def bottle_neck(self, x):
        assert len(x.shape) == 2
        mu = self.fc1(x)
        logvar = self.fc2(x)
        std = logvar.mul(0.5).exp_()
        epsilon = torch.randn(mu.shape)
        z_sampled = mu + std * epsilon
        z_sampled = self.fc3(z_sampled)
        return z_sampled, mu, logvar

