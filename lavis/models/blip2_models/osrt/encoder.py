import numpy as np
import torch
import torch.nn as nn
from osrt.layers import RayEncoder, Transformer, SlotAttention


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class ImprovedSRTEncoder(nn.Module):
    """
    Scene Representation Transformer Encoder with the improvements from Appendix A.4 in the OSRT paper.
    """
    def __init__(self, num_conv_blocks=3, num_att_blocks=5, pos_start_octave=0):
        super().__init__()
        self.ray_encoder = RayEncoder(pos_octaves=15, pos_start_octave=pos_start_octave,
                                      ray_octaves=15)

        conv_blocks = [SRTConvBlock(idim=183, hdim=96)]
        cur_hdim = 192
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.per_patch_linear = nn.Conv2d(cur_hdim, 768, kernel_size=1)

        self.transformer = Transformer(768, depth=num_att_blocks, heads=12, dim_head=64,
                                       mlp_dim=1536, selfatt=True)

    def forward(self, images, camera_pos, rays):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        batch_size, num_images = images.shape[:2]

        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)

        ray_enc = self.ray_encoder(camera_pos, rays)
        x = torch.cat((x, ray_enc), 1)
        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)
        x = x.flatten(2, 3).permute(0, 2, 1)

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image, channels_per_patch)

        x = self.transformer(x)

        return x


class OSRTEncoder(nn.Module):
    def __init__(self, pos_start_octave=0, num_slots=6, slot_dim=1536, slot_iters=1,
                 randomize_initial_slots=False):
        super().__init__()
        self.srt_encoder = ImprovedSRTEncoder(num_conv_blocks=3, num_att_blocks=5,
                                             pos_start_octave=pos_start_octave)

        self.slot_attention = SlotAttention(num_slots, slot_dim=slot_dim, iters=slot_iters,
                                            randomize_initial_slots=randomize_initial_slots)

    def forward(self, images, camera_pos, rays):
        set_latents = self.srt_encoder(images, camera_pos, rays)
        slot_latents = self.slot_attention(set_latents)
        return slot_latents

