import torch
import torch.nn as nn
import torch.fft as fft


class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.
    Modified by Jiatian Zhang, 2025, for BDFM

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_size=[4, 4, 4], ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_size = patch_size

        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x, mask):
        # crop image patches
        B, C, D, H, W = x.shape
        patch_size = self.patch_size
        patch_d, patch_h, patch_w = patch_size

        assert H % patch_h == 0 and W % patch_w == 0 and D % patch_d == 0, (
            'Patch size should be divisible by the dimensions H, W, D')

        patch_list = []
        num_patches_d = D // patch_d
        num_patches_h = H // patch_h
        num_patches_w = W // patch_w

        for i in range(num_patches_d):
            for j in range(num_patches_h):
                for k in range(num_patches_w):
                    patch_x = x[:, :, i * patch_d:(i + 1) * patch_d, j * patch_h:(j + 1) * patch_h,
                              k * patch_w:(k + 1) * patch_w]
                    patch_mask = mask[:, :, i * patch_d:(i + 1) * patch_d, j * patch_h:(j + 1) * patch_h,
                                 k * patch_w:(k + 1) * patch_w]
                    if patch_mask.sum() > 0:
                        patch_list.append(patch_x)

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 3D DFT (real-to-complex, orthonormalization)
        freq = fft.fftn(y, dim=(2, 3, 4), norm='ortho')
        freq = torch.stack([freq.real, freq.imag], -1)

        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values.max(-1).values[:, :, :, None, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, mask, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (B, C, D, H, W). Predicted tensor.
            target (torch.Tensor): of shape (B, C, D, H, W). Target tensor.
            mask (torch.Tensor): of shape (B, C, D, H, W). Mask tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred, mask)
        target_freq = self.tensor2freq(target, mask)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight