# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the CC BY-NC 4.0 License (https://creativecommons.org/licenses/by-nc/4.0/)
# Modified by Jiatian Zhang, 2025.
# --------------------------------------------------------
# Reference:
# https://github.com/microsoft/SimMIM/blob/main/models/simmim.py
# https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import logging
from typing import Sequence, Union

import torch
import torch.nn as nn
import math
from monai.networks.layers import Conv

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.nets import ViT
from models.swin_3d import SwinTransformer3D

from mmengine.runner import load_checkpoint
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Block



__all__ = ["SimMIMSwin"]


class SimMIMSwin(nn.Module):


    def __init__(
        self,
        pretrained: Union[None, str],
        patch_size: Union[Sequence[int], int],
        in_chans: int = 1,
        embed_dim: int = 96,
        depths: Sequence[int] = [2, 2, 6, 2],
        num_heads: Sequence[int] = [3, 6, 12, 24],
        window_size: Sequence[int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Union[None, bool] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer=nn.LayerNorm,
        patch_norm: bool = False,
        frozen_stages: int = -1,
        use_checkpoint: bool = False,
        masking_ratio: float = 0.75,
        revise_keys=[("model.", "")],
        # loss
        loss_pixel_w: float = 1.0,
        **kwargs,

    ) -> None:
        super().__init__()


        # --------------------------------------------------------------------------
        # MIM encoder
        self.pretrained = pretrained
        self.patch_size = patch_size
        assert (
                masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        self.encoder = SwinTransformer3D(
            pretrained=pretrained,
            pretrained2d=False,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint,
        )

        # patch embedding
        num_features = self.encoder.num_features
        num_layers = self.encoder.num_layers
        final_resolution = self.encoder.final_resolution
        self.num_features = num_features
        self.num_layers = num_layers
        self.final_resolution = self.encoder.final_resolution

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder
        conv_trans = Conv[Conv.CONVTRANS, 3]
        self.conv3d_transpose = conv_trans(
            num_features,
            16,
            kernel_size=(
                self.patch_size[0],
                2 ** (num_layers - 1),
                2 ** (num_layers - 1),
            ),
            stride=(self.patch_size[0], 2 ** (num_layers - 1), 2 ** (num_layers - 1),),
        )
        self.conv3d_transpose_1 = conv_trans(
            in_channels=16,
            out_channels=in_chans,
            kernel_size=(1, self.patch_size[1], self.patch_size[2]),
            stride=(1, self.patch_size[1], self.patch_size[2]),
        )  # B C D H W
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # loss
        self.loss_pixel_w = loss_pixel_w
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        #trunc_normal_(self.mask_token, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)
        # --------------------------------------------------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, D, H, W)
        x: (N, L, patch_size**3 *1)
        """
        p = 4
        assert imgs.shape[2] == imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0

        h = w = d = imgs.shape[2] // p  # 24
        x = imgs.reshape(shape=(imgs.shape[0], 1, d, p, h, p, w, p))
        x = torch.einsum('ncdphqwj->ndhwpqjc', x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p ** 3 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *1)
        imgs: (N, 1, D, H, W)
        """
        p = 4
        h = w = d = round((x.shape[1]) ** (1 / 3))

        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, 1))
        x = torch.einsum('ndhwpqjc->ncdphqwj', x)
        imgs = x.reshape(shape=(x.shape[0], 1, d * p, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        seed = 42
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # [N, len_keep]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # [N, len_keep, D]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # B, C, D, H, W
        # embed patches
        x = self.encoder.patch_embed(x)  # [B, 24**3, embed_dim]

        # masking: length -> length * masking_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)  # [B,keep,96],[B,24**3],[B,24**3]
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

        # cat x and mask_tokens
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add absolute pos embed
        if self.encoder.ape:
            x = x + self.encoder.absolute_pos_embed
        x = self.encoder.pos_drop(x)

        # encoder
        for layer in self.encoder.layers:
            x = layer(x)
        x = self.encoder.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x):
        # predictor projection (pixel level)
        x = x.transpose(1, 2).view(
            -1, self.num_features, *self.final_resolution
        )
        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)  # (B,1,96,96,96)

        return x

    def forward_loss(self, imgs, pred_pixel, mask):
        """
               imgs: [B, 1, D, H, W]
               pred_pixel: [B, 1, D, H, W]
               mask: [B, L], 0 is keep, 1 is remove
        """
        # pixel loss
        target = self.patchify(imgs)
        pred = self.patchify(pred_pixel)
        loss_pixel = (pred - target) ** 2
        loss_pixel = loss_pixel.mean(dim=-1)  # [B, L], mean loss per patch

        loss_pixel = (loss_pixel * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = loss_pixel

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        imgs = imgs.permute(0, 1, 4, 2, 3)  # (B, 1, d, h, w)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred_pixel = self.forward_decoder(latent)  # (B,1,96,96,96)
        loss = self.forward_loss(imgs, pred_pixel, mask)

        return loss, pred_pixel, mask

if __name__ == "__main__":

    model = SimMIMSwin(
        pretrained=None,
        patch_size=[4,4,4],
        in_chans=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(7,7,7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        frozen_stages=-1,
        use_checkpoint=False,
        masking_ratio=0.75,
        loss_pixel_w=1.0

    )

    x = torch.randn(2, 1, 96, 96, 96)
    latent, mask, ids_restore = model.forward_encoder(x, mask_ratio=0.75)
    pred = model.forward_decoder(latent)
    loss = model.forward_loss(x, pred, mask)

    print(loss)
