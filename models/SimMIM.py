"""Jiatian Zhang
zjt20174025@mail.ustc.edu.cn
Reference:
https://github.com/microsoft/SimMIM/blob/main/models/simmim.py
https://github.com/facebookresearch/mae/blob/main/models_mae.py
https://github.com/Project-MONAI/MONAI/blob/b61db797e2f3bceca5abbaed7b39bb505989104d/monai/networks/nets/vit.py
"""

import logging
from typing import Sequence, Union

import torch
import torch.nn as nn
import math

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.nets import ViT

import torch.fft as fft
from mmengine.runner import load_checkpoint
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Block
from utils.focal_frequency_loss import FocalFrequencyLoss as FFL



__all__ = ["SimMIMViT"]


class SimMIMViT(nn.Module):


    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        norm_pix_loss = True,
        pretrained = None,
        revise_keys=[("model.", "")],
        # loss
        loss_pixel_w: float = 1.0,
        **kwargs,

    ) -> None:
        super().__init__()


        # --------------------------------------------------------------------------
        # MIM encoder
        self.spatial_dims = spatial_dims

        self.encoder = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        patch_embedding = self.encoder.patch_embedding
        self.patch_embeddings = patch_embedding.patch_embeddings
        n_patches = patch_embedding.n_patches  # 216
        patch_dim = patch_embedding.patch_dim  # 16*16*16*1

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))

        self.norm = nn.LayerNorm(hidden_size)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Pixel Decoder
        self.decoder_pred = nn.Linear(hidden_size, patch_dim, bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # loss
        self.loss_pixel_w = loss_pixel_w
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # others
        self.norm_pix_loss = norm_pix_loss

        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-2.0, b=2.0)
        trunc_normal_(self.mask_token, mean=0.0, std=0.02, a=-2.0, b=2.0)

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
        p = 16
        assert imgs.shape[2] == imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0

        h = w = d = imgs.shape[2] // p  # 6
        x = imgs.reshape(shape=(imgs.shape[0], 1, d, p, h, p, w, p))
        x = torch.einsum('ncdphqwj->ndhwpqjc', x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p ** 3 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *1)
        imgs: (N, 1, D, H, W)
        """
        p = 16
        h = w = d = round((x.shape[1]) ** (1 / 3))  # 6

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
        # embed patches
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]
        #x = self.norm(x)

        # masking: length -> length * masking_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)  #[B,keep,768],[B,216],[B,216]
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

        # cat x and mask_tokens
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # append cls token
        #cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        #x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed
        x = x + self.position_embeddings  # [B, num_patches + 1, hidden_size]
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.norm(x)

        # remove cls token
        #x = x[:, 1:, :]

        return x, mask, ids_restore

    def forward_decoder(self, x):

        # predictor projection (pixel level)
        x = self.decoder_pred(x)  # [B, num_patches, patch_dim]
        return x

    def forward_loss(self, imgs, pred_pixel, mask):
        """
               imgs: [B, 1, D, H, W]
               pred_pixel: [B, L, patch_size**3 *1]
               mask: [B, L], 0 is keep, 1 is remove
        """
        # pixel loss
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss_pixel = (pred_pixel - target) ** 2  # [B, 216, 4096]
        loss_pixel = loss_pixel.mean(dim=-1)  # [B, L], mean loss per patch

        loss_pixel = (loss_pixel * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = loss_pixel

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        imgs = imgs.permute(0, 1, 4, 2, 3)  # (B, 1, d, h, w)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred_pixel = self.forward_decoder(latent)  # (B,216,16*16*16*1)
        loss = self.forward_loss(imgs, pred_pixel, mask)

        return loss, pred_pixel, mask

if __name__ == "__main__":
    model = SimMIMViT(
        in_channels=1,
        img_size=(96,96,96),
        patch_size=(16,16,16),
    )

    x = torch.randn(2, 1, 96, 96, 96)
    y = model(x)

    print(y[1].shape)