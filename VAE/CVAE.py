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
    def __init__(self, channel:int = 256) -> None:
        super().__init__()
        self.channel = channel

    def forward(self, x):
        B, D = x.shape
        H = W = int(np.sqrt(D / self.channel))
        return x.view(B, self.channel, H, W)

def posenc_nerf(x, min_deg=0, max_deg=15):
    """Concatenate x and its positional encodings, following NeRF."""
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)]).float().to(x)

    xb = rearrange(
        (x[..., None, :] * scales[:, None]), "b f h w c d -> b f h w (c d)")
    emb = torch.sin(torch.concat([xb, xb + torch.pi / 2.], dim=-1))

    return torch.concat([x, emb], dim=-1)

def rt_to_quaternion(R:torch.tensor, t:torch.tensor):
    """Converts rotation matrix and translation vector to quaternion."""
    # R: [B, 3, 3]
    # t: [B, 3]
    # q: [B, 4]
    trace = torch.trace(R)
    trace = torch.clamp(trace, min=-1, max=3)
    if trace > 0:
        qw = 0.5 * torch.sqrt(1 + trace)
        qx = R[2, 1] - R[1, 2] / (4 * qw)
        qy = R[0, 2] - R[2, 0] / (4 * qw)
        qz = R[1, 0] - R[0, 1] / (4 * qw)
    else:
        max_diag = torch.argmax(torch.diag(R))
        if max_diag == 0:
            qx = 0.5 * torch.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            qy = R[0, 1] + R[1, 0] / (4 * qx)
            qz = R[0, 2] + R[2, 0] / (4 * qx)
            qw = R[2, 1] - R[1, 2] / (4 * qx)
        elif max_diag == 1:
            qy = 0.5 * torch.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            qx = R[0, 1] + R[1, 0] / (4 * qy)
            qz = R[1, 2] + R[2, 1] / (4 * qy)
            qw = R[0, 2] - R[2, 0] / (4 * qy)
        elif max_diag == 2:
            qz = 0.5 * torch.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            qx = R[0, 2] + R[2, 0] / (4 * qz)
            qy = R[1, 2] + R[2, 1] / (4 * qz)
            qw = R[1, 0] - R[0, 1] / (4 * qz)
        else:
            qw = 0
            qx = 0
            qy = 0
            qz = 0
    q = torch.tensor([qw, qx, qy, qz])
    return q

class PoseConditionProcessor(torch.nn.Module):

    def __init__(self, emb_ch, H, W,
                 num_resolutions,
                 use_pos_emb=True,
                 use_ref_pose_emb=True):

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

        if use_ref_pose_emb:
            self.first_emb = torch.nn.Parameter(torch.zeros(1, 1, D, 1, 1), requires_grad=True)
            torch.nn.init.normal_(self.first_emb, std=(1 / np.sqrt(D)))

            self.other_emb = torch.nn.Parameter(torch.zeros(1, 1, D, 1, 1), requires_grad=True)
            torch.nn.init.normal_(self.other_emb, std=(1 / np.sqrt(D)))

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
        self.conv1 = nn.Conv2d(in_channel, out_channel, 5, 1, 2)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 5, 2, 2)
        self.norm1 = nn.LayerNorm([out_channel, input_h, input_w])
        self.norm2 = nn.LayerNorm([out_channel, input_h // 2, input_w // 2])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, input_h:int, input_w:int, end:bool=False) -> None:
        super().__init__()
        # self.end = end
        self.conv1 = nn.Conv2d(in_channel, out_channel, 5, 1, 2)
        self.conv2 = nn.ConvTranspose2d(out_channel, out_channel, 5, 2, 2, output_padding=1)
        self.norm1 = nn.LayerNorm([out_channel, input_h, input_w])
        self.norm2 = nn.LayerNorm([out_channel, input_h * 2, input_w * 2])
        self.used_relu = nn.ReLU()
        if not end:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.used_relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.relu:
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
        # self.ec2 = EncoderBlock(32 + emb_ch, 64, H // 2, W // 2)
        # self.ec3 = EncoderBlock(64 + emb_ch, 128, H // 4, W // 4)
        # self.ec4 = EncoderBlock(128 + emb_ch, 256, H // 8, W // 8)
        self.ec2 = EncoderBlock(32, 64, H // 2, W // 2)
        self.ec3 = EncoderBlock(64, 128, H // 4, W // 4)
        self.ec4 = EncoderBlock(128, 256, H // 8, W // 8)

        self.flatten = Flatten()
        self.fc1 = nn.Linear(256 * (H // 16) * (W // 16), 2*z_dim) # for mu, logvar
        self.fc2 = nn.Linear(z_dim, 256 * (H // 16) * (W // 16)) # for decoder
        self.unflatten = UnFlatten()

        self.dc1 = DecoderBlock(256, 128, H // 16, W // 16)
        self.dc2 = DecoderBlock(128 + emb_ch, 64, H // 8, W // 8)
        self.dc3 = DecoderBlock(64 + emb_ch, 32, H // 4, W // 4)
        self.dc4 = DecoderBlock(32 + emb_ch, 3, H // 2, W // 2, True)

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
        # input2 = torch.concat([out1, pose_embeds[0][:,0]], dim=1)
        out2 = self.ec2(out1)
        # input3 = torch.concat([out2, pose_embeds[1][:,0]], dim=1)
        out3 = self.ec3(out2)
        # input4 = torch.concat([out3, pose_embeds[2][:,0]], dim=1)
        out4 = self.ec4(out3)
        z_out = self.fc1(self.flatten(out4))
        return z_out[:,:self.z_dim], z_out[:,self.z_dim:]

    def reparaterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, pose_embeds):
        input1 = self.fc2(z)
        out1 = self.dc1(self.unflatten(input1))

        # generate new image
        input2_z = torch.concat([out1, pose_embeds[2][:,1]], dim=1)
        out2_z = self.dc2(input2_z)
        input3_z = torch.concat([out2_z, pose_embeds[1][:,1]], dim=1)
        out3_z = self.dc3(input3_z)
        input4_z = torch.concat([out3_z, pose_embeds[0][:,1]], dim=1)
        out4_z = self.dc4(input4_z)

        # reconstruct input image
        input2_x = torch.concat([out1, pose_embeds[2][:,0]], dim=1)
        out2_x = self.dc2(input2_x)
        input3_x = torch.concat([out2_x, pose_embeds[1][:,0]], dim=1)
        out3_x = self.dc3(input3_x)
        input4_x = torch.concat([out3_x, pose_embeds[0][:,0]], dim=1)
        out4_x = self.dc4(input4_x)

        return out4_z, out4_x


    def forward(self, batch, cond_mask=None):
        pose_embeds = self.condition_processor(batch, cond_mask)
        # print([pose_embeds[i].shape for i in range(3)])
        x = batch['x']
        gt = batch['z']
        z_mu, z_logvar = self.encode(x, pose_embeds)
        z = self.reparaterize(z_mu, z_logvar)
        img_gen, img_recon = self.decode(z, pose_embeds)
        return self.loss(z_mu, z_logvar, img_gen, gt, img_recon, x)
    
    def loss(self, z_mu, z_logvar, img_gen, img_gt, img_recon, img_input):
        kld = torch.mean(
        -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0
        )
        # img_loss = ((img_gen - img_gt)**2).sum(dim=(1,2,3)).mean()
        # img_loss += ((img_recon - img_input)**2).sum(dim=(1,2,3)).mean()
        img_loss = F.mse_loss(img_gen, img_gt) + F.mse_loss(img_recon, img_input)
        return self.beta * kld , img_loss
    
    def eval_img(self, batch, cond_mask=None):
        pose_embeds = self.condition_processor(batch, cond_mask)
        x = batch['x']
        z_mu, z_logvar = self.encode(x, pose_embeds)
        pred_img, recon_img = self.decode(z_mu, pose_embeds)
        return pred_img, recon_img

class PoseMapping(nn.Module):
    '''Map the pose(quaternion) to two vectors'''
    def __init__(self, embed:int = 64) -> None:
        super().__init__()
        self.fc_x = nn.Linear(4, embed)
        self.fc_y = nn.Linear(4, embed)

    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 4
        return self.fc_x(x), self.fc_y(x)
    
class CDVAEEncBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class ConditionalDeformableVAE(nn.Module):
    def __init__(self, pose_embed:int=64) -> None:
        super().__init__()
        self.pose_embed = pose_embed

        self.pose_mapping_enc = PoseMapping(pose_embed)
        self.pose_mapping_dec = PoseMapping(pose_embed)
