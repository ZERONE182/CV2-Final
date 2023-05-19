import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import visu3d as v3d
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

def posenc_nerf(x, min_deg=0, max_deg=15):
    """Concatenate x and its positional encodings, following NeRF."""
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)]).float().to(x)

    xb = rearrange(
        (x[..., None, :] * scales[:, None]), "b f h w c d -> b f h w (c d)")
    emb = torch.sin(torch.concat([xb, xb + torch.pi / 2.], dim=-1))

    return torch.concat([x, emb], dim=-1)
# class PoseCondition(nn.Module):
#     def __init__(self) -> None:
#         # TODO: Decide to imitate the method done in XUnet. 
#         # By using the camera model in visu3d, we can get the ray info for all the pixels in the image. 
#         # The ray origin is 3-dim vector and direction is also 3-dim vector. 
#         # If we just concate the ray info for all the pixel in the image, we can get a tensor in shape (H, W, 6), 6 can be seen as channel?
#         # NeRF PE is applied on ray origin and ray direction. 
#         # CNN is also used to change the spatial size to be the same as the downsampled image during VAE processing. 
#         super().__init__()

class PoseConditionProcessor(torch.nn.Module):

    def __init__(self, emb_ch, H, W,
                 num_resolutions,
                 use_pos_emb=False,
                 use_ref_pose_emb=False):

        super().__init__()

        self.emb_ch = emb_ch
        self.num_resolutions = num_resolutions
        self.use_pos_emb = use_pos_emb
        self.use_ref_pose_emb = use_ref_pose_emb

        D = 144
        # D is related to the max_deg and the min_deg of posenc_nerf together with x.shape[-1]
        # So if all the values about are fixed, then no need to change D
        if use_pos_emb:
            self.pos_emb = torch.nn.Parameter(torch.zeros(D, H, W), requires_grad=True)
            torch.nn.init.normal_(self.pos_emb, std=(1 / np.sqrt(D)))

        # if use_ref_pose_emb:
        #     self.first_emb = torch.nn.Parameter(torch.zeros(1, 1, D, 1, 1), requires_grad=True)
        #     torch.nn.init.normal_(self.first_emb, std=(1 / np.sqrt(D)))

        #     self.other_emb = torch.nn.Parameter(torch.zeros(1, 1, D, 1, 1), requires_grad=True)
        #     torch.nn.init.normal_(self.other_emb, std=(1 / np.sqrt(D)))

        convs = []
        for i_level in range(self.num_resolutions):
            convs.append(torch.nn.Conv2d(in_channels=D,
                                         out_channels=self.emb_ch,
                                         kernel_size=3,
                                         stride=2 ** (i_level+1), padding=1))

        self.convs = torch.nn.ModuleList(convs)

    def forward(self, batch, cond_mask):

        B, C, H, W = batch['x'].shape

        world_from_cam = v3d.Transform(R=batch['R'].cpu().numpy(), t=batch['t'].cpu().numpy())
        cam_spec = v3d.PinholeCamera(resolution=(H, W), K=batch['K'].unsqueeze(1).cpu().numpy())
        rays = v3d.Camera(
            spec=cam_spec, world_from_cam=world_from_cam).rays()

        pose_emb_pos = posenc_nerf(torch.tensor(rays.pos).float().to(batch['x']), min_deg=0, max_deg=15)
        pose_emb_dir = posenc_nerf(torch.tensor(rays.dir).float().to(batch['x']), min_deg=0, max_deg=8)

        pose_emb = torch.concat([pose_emb_pos, pose_emb_dir], dim=-1)  # [batch, h, w, 144]

        if cond_mask is not None:
            assert cond_mask.shape == (B,), (cond_mask.shape, B)
            cond_mask = cond_mask[:, None, None, None, None]
            pose_emb = torch.where(cond_mask, pose_emb, torch.zeros_like(pose_emb))  # [B, F, H, W, 144]

        pose_emb = rearrange(pose_emb, "b f h w c -> b f c h w")
        # pose_emb = torch.tensor(pose_emb).float().to(device)

        # now [B, 1, C=144, H, W]

        if self.use_pos_emb:
            pose_emb += self.pos_emb[None, None]
        if self.use_ref_pose_emb:
            pose_emb = torch.concat([self.first_emb, self.other_emb], dim=1) + pose_emb
            # now [B, 2, C=144, H, W]

        pose_embs = []
        for i_level in range(self.num_resolutions):
            B, F = pose_emb.shape[:2]
            pose_embs.append(
                rearrange(self.convs[i_level](
                    rearrange(pose_emb, 'b f c h w -> (b f) c h w')
                ),
                    '(b f) c h w -> b f c h w', b=B, f=F
                )
            )

        return pose_embs

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
    def __init__(self, H: int = 128, W: int = 128, z_dim: int = 128, n_resolution: int = 3, emb_ch : int = 128) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.z_dim = z_dim
        self.n_resolution = n_resolution
        self.emb_ch = emb_ch
        self.beta = 1/z_dim

        self.condition_processor = PoseConditionProcessor(emb_ch, H, W, n_resolution)
        # TODO: Now hardcode for layers, change to list
        self.ec1 = EncoderBlock(3, 32, H, W)
        self.ec2 = EncoderBlock(32 + emb_ch, 64, H // 2, W // 2)
        self.ec3 = EncoderBlock(64 + emb_ch, 128, H // 4, W // 4)
        self.ec4 = EncoderBlock(128 + emb_ch, 256, H // 8, W // 8)

        self.flatten = Flatten()
        self.fc1 = nn.Linear(256 * (H // 16) * (W // 16), 2*z_dim) # for mu, logvar
        self.fc2 = nn.Linear(z_dim, 256 * (H // 16) * (W // 16)) # for decoder
        self.unflatten = UnFlatten()

        self.dc1 = DecoderBlock(256, 128, H // 16, W // 16)
        self.dc2 = DecoderBlock(128 + emb_ch, 64, H // 8, W // 8)
        self.dc3 = DecoderBlock(64 + emb_ch, 32, H // 4, W // 4)
        self.dc4 = DecoderBlock(32 + emb_ch, 3, H // 2, W // 2)

    def bottle_neck(self, x):
        assert len(x.shape) == 2
        mu = self.fc1(x)
        logvar = self.fc2(x)
        std = logvar.mul(0.5).exp_()
        epsilon = torch.randn(mu.shape)
        z_sampled = mu + std * epsilon
        z_sampled = self.fc3(z_sampled)
        return z_sampled, mu, logvar

    def encode(self, x, pose_embeds):
        out1 = self.ec1(x)
        input2 = torch.concat([out1, pose_embeds[0][:,0,:]], dim=1)
        out2 = self.ec2(input2)
        input3 = torch.concat([out2, pose_embeds[1][:,0,:]], dim=1)
        out3 = self.ec3(input3)
        input4 = torch.concat([out3, pose_embeds[2][:,0,:]], dim=1)
        out4 = self.ec4(input4)
        z_out = self.fc1(self.flatten(out4))
        return z_out[:,:self.z_dim], z_out[:,self.z_dim:]

    def reparaterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, pose_embeds):
        input1 = self.fc2(z)
        out1 = self.dc1(self.unflatten(input1))
        input2 = torch.concat([out1, pose_embeds[2][:,1,:]], dim=1)
        out2 = self.dc2(input2)
        input3 = torch.concat([out2, pose_embeds[1][:,1,:]], dim=1)
        out3 = self.dc3(input3)
        input4 = torch.concat([out3, pose_embeds[0][:,1,:]], dim=1)
        out4 = self.dc4(input4)
        return out4


    def forward(self, batch, cond_mask=None):
        pose_embeds = self.condition_processor(batch, cond_mask)
        # print([pose_embeds[i].shape for i in range(3)])
        x = batch['x']
        z_mu, z_logvar = self.encode(x, pose_embeds)
        z = self.reparaterize(z_mu, z_logvar)
        img_recon = self.decode(z, pose_embeds)
        return self.loss(z_mu, z_logvar, img_recon, x)
    
    def loss(self, z_mu, z_logvar, img_recon, img_gt):
        kld = torch.mean(
        -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0
        )
        img_loss = ((img_gt - img_recon)**2).mean()
        return self.beta * kld , img_loss
    
    def eval_img(self, batch, cond_mask=None):
        pose_embeds = self.condition_processor(batch, cond_mask)
        x = batch['x']
        z_mu, z_logvar = self.encode(x, pose_embeds)
        img_recon = self.decode(z_mu, pose_embeds)
        return img_recon

