import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import logging

class PiecewiseLinearVTLNWarp(nn.Module):
    def __init__(self, num_bins=80):
        super().__init__()
        # TODO: check if we can check self.epoch
        self.num_bins = num_bins
        self.alpha1_raw = nn.Parameter(torch.tensor(1.0))

    def _warp_frequencies(self, f, alpha1, B):
        alpha1 = alpha1.view(B, 1)  # [B, 1]
        f = f.view(1, -1)  # [1, F]
        warped_f = f**(alpha1)
        return warped_f.view(B, 1, -1)  # [B, 1, F]

    def forward(self, x, utt_2_warp=None):
        # Decoding:
        if len(x.shape) == 2:
            T, D = x.shape
            B = 1
            if D == 80:
                fbank = x
                aux = None
            else:
                fbank, aux = x[:, :self.num_bins], x[:, self.num_bins:]
        else:
            B, T, D = x.shape # Batch Size, Amount of Frames, Dimension of fbanks
            fbank, aux = x[:, :, :self.num_bins], x[:, :, self.num_bins:]
        if D == 80:
            fbank = x
            aux = None

        alpha1 = self.alpha1_raw
        # Utt2warp given in decoding
        if utt_2_warp is not None:
            logging.info(f"warp layer with {utt_2_warp}") # check in exp/train.../decode/test.../logdir/ if utt2warp is passed here
            assert B == 1, "utt_2_warp is only supported for batch size of 1"
            alpha1 = utt_2_warp.squeeze()  # ensure it's a scalar tensor
        
        f = torch.linspace(0, 1, self.num_bins, device=x.device, dtype = x.dtype)
        warped_f = self._warp_frequencies(f, alpha1, B)
        time_grid = torch.linspace(-1, 1, T, device=x.device, dtype = x.dtype).view(1, T, 1).expand(B, T, self.num_bins)
        grid = torch.stack([warped_f.expand(-1, T, -1) * 2 - 1, time_grid], dim=-1)
        grid = grid.to(fbank.dtype)  # Match dtype to input
        fbank = fbank.to(grid.dtype)
        fbank = fbank.unsqueeze(1) # Add channel for grid sample function
        warped = F.grid_sample(fbank, grid, mode="bilinear", align_corners=True, padding_mode="zeros").squeeze(1)
        if aux == None:
            return warped
        return torch.cat([warped, aux], dim=-1)