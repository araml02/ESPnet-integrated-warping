from typing import Tuple

import librosa
import torch
import numpy as np
import torchaudio
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.layers.differ_mel_banks import DifferentiableMelBanks   # assume you saved the class here
import logging


class LogMel(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base
        # if trainable_warp:
        #     self.alpha_train = nn.Parameter(torch.tensor(0.0))  # Learnable
        # else:
        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        # Note(kamo): The mel matrix of librosa is different from kaldi.
        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

        self.melbank = DifferentiableMelBanks(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            alpha_init=1.0,  # Will be dynamically overridden
        )

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
        warp_alpha: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # logging.info(self.melmat.shape)
        if warp_alpha is not None and warp_alpha.item() != 0.0:
            alpha = warp_alpha
            logging.info(f"MelMat warping: alpha={alpha.item():.3f}")
            # melmat = self.warp_melmat_np(self.melmat, alpha.item())
            # melmat = self.warp_melmat_differentiable(self.melmat, alpha)
            melmat = self.make_slaney_vtln_mel_filterbank(vtln_warp = alpha)
        # elif self.alpha_train.item() != 0.0 and self.alpha_train.item() != 1.0:
        elif self.alpha_train.item() != 0.0:
            # if warp_alpha is None:
            #     warp_alpha = 0.75 + 0.5 * 0.5 * (torch.tanh(self.alpha_train) + 1)
            alpha = self.alpha_train
            # logging.info(f"MelMat TRAIN warping: alpha={alpha.item():.3f}")
            # melmat = self.warp_melmat_np(self.melmat, alpha.item())
            # melmat = self.warp_melmat_differentiable(self.melmat, alpha)
            melmat = self.make_slaney_vtln_mel_filterbank(vtln_warp = alpha)
        else:
            melmat = self.melmat  # precomputed buffer
            logging.info("No Melmat Warping")

        mel_feat = torch.matmul(feat, melmat)  # (B, T, D2)
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        # mel_feat = torch.matmul(feat, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / torch.log(self.log_base)

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
            )
        return logmel_feat, ilens

    # def warp_melmat_np(self, melmat: torch.Tensor, alpha: float) -> torch.Tensor:
    #     melmat_np = melmat.cpu().numpy()
    #     n_fft_bins, n_mels = melmat_np.shape
    #     orig_x = np.linspace(0, 1, n_fft_bins)
    #     warped_x = np.clip(orig_x / alpha, 0, 1)

    #     warped = np.stack([
    #         np.interp(orig_x, warped_x, melmat_np[:, i])
    #         for i in range(n_mels)
    #     ], axis=1)

    #     return torch.from_numpy(warped).float()

    def warp_melmat_np(self, melmat: torch.Tensor, alpha: float) -> torch.Tensor:
        melmat_np = melmat.cpu().numpy()
        n_fft_bins, n_mels = melmat_np.shape
        orig_x = np.linspace(0, 1, n_fft_bins)
        warped_x = np.clip(orig_x ** (1 / alpha), 0, 1)

        warped = np.stack([
            np.interp(orig_x, warped_x, melmat_np[:, i])
            for i in range(n_mels)
        ], axis=1)

        return torch.from_numpy(warped).float()

    def warp_melmat_differentiable(self, melmat: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        n_fft_bins, n_mels = melmat.shape  # 257, 80
        device = melmat.device

        # Input shape for grid_sample must be (N, C, H, W)
        # We'll treat melmat.T as a single-channel image of shape (1, 1, 80, 257)
        melmat = melmat.T.unsqueeze(0).unsqueeze(0)  # (1, 1, 80, 257)

        # Generate 1D warped x positions
        orig_x = torch.linspace(0, 1, n_fft_bins, device=device)
        # warped_x = torch.clamp(orig_x ** (1 / alpha), 0, 1)
        warped_x = torch.clamp(orig_x / alpha, 0, 1)
        grid_x = warped_x * 2 - 1  # normalize to [-1, 1]

        # Create 2D sampling grid for all mel bands at once (height = 80)
        grid_x = grid_x.unsqueeze(0).repeat(n_mels, 1)  # (80, 257)
        grid_y = torch.linspace(-1, 1, n_mels, device=device).unsqueeze(1).repeat(1, n_fft_bins)  # (80, 257)
        grid = torch.stack((grid_x, grid_y), dim=-1)  # (80, 257, 2)
        grid = grid.unsqueeze(0)  # (1, 80, 257, 2)

        # Warp using grid_sample
        warped = F.grid_sample(melmat, grid, mode='bilinear', align_corners=True)  # (1, 1, 80, 257)

        # Reshape back to (257, 80)
        warped = warped.squeeze(0).squeeze(0).T  # (257, 80)

        return warped

    def make_slaney_vtln_mel_filterbank(
        self,
        sr=16000,
        n_fft=512,
        n_mels=80,
        fmin=20,
        fmax=None,
        vtln_low=100,
        vtln_high=7000,
        vtln_warp=1.0  # Can now be a float or a torch tensor
    ):
        if fmax is None:
            fmax = sr / 2

        def hz_to_mel(f):
            # Accepts numpy or torch
            return 2595 * (torch.log10(f / 700 + 1) if isinstance(f, torch.Tensor) else np.log10(1 + f / 700))

        def mel_to_hz(m):
            # Accepts numpy or torch
            return 700 * ((10 ** (m / 2595) - 1) if not isinstance(m, torch.Tensor) else (torch.pow(10, m / 2595) - 1))

        def vtln_warp_freq(freq, low_cutoff, high_cutoff, low_freq, high_freq, warp_factor):
            # freq: float or 1D torch tensor
            # warp_factor: float or torch tensor
            freq = torch.as_tensor(freq, dtype=torch.float64)
            warp_factor = torch.as_tensor(warp_factor, dtype=torch.float64)
            scale = 1.0 / warp_factor
            Fl = scale * low_cutoff
            Fh = scale * high_cutoff
            scale_left = (Fl - low_freq) / (low_cutoff - low_freq)
            scale_right = (high_freq - Fh) / (high_freq - high_cutoff)
            out = torch.empty_like(freq)
            out[freq < low_cutoff] = low_freq + scale_left * (freq[freq < low_cutoff] - low_freq)
            mid = (freq >= low_cutoff) & (freq <= high_cutoff)
            out[mid] = scale * freq[mid]
            # out[mid] = freq[mid] ** scale
            out[freq > high_cutoff] = high_freq + scale_right * (freq[freq > high_cutoff] - high_freq)
            return out

        # Compute mel-scale bin edges, convert to Hz, then warp
        mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2, dtype=np.float64)
        hz_edges = mel_to_hz(mel_edges)
        hz_edges_torch = torch.from_numpy(hz_edges).to(dtype=torch.float64)

        # vtln_warp can be a float or a tensor (e.g., shape [B])
        warped_hz_edges = vtln_warp_freq(
            hz_edges_torch, vtln_low, vtln_high, fmin, fmax, vtln_warp
        )

        bin_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        bin_freqs_torch = torch.from_numpy(bin_freqs).to(dtype=torch.float64)

        filterbank = torch.zeros((n_mels, len(bin_freqs_torch)), dtype=torch.float64)

        for i in range(n_mels):
            f_left = warped_hz_edges[i]
            f_center = warped_hz_edges[i + 1]
            f_right = warped_hz_edges[i + 2]

            height = 2.0 / (f_right - f_left + 1e-10)
            # valid_bins = bin_freqs_torch[(bin_freqs_torch >= f_left) & (bin_freqs_torch <= f_right)]
            # if len(valid_bins) >= 2:
            #     height = 2.0 / (valid_bins[-1] - valid_bins[0] + 1e-10)
            #     height = 2.0 / (f_right - f_left + 1e-10)
            # else:
            #     height = 0.0  # no energy passed
            if f_center > 6000:
                height = 0.005

            left_mask = (bin_freqs_torch >= f_left) & (bin_freqs_torch <= f_center)
            filterbank[i, left_mask] = (
                height * (bin_freqs_torch[left_mask] - f_left) / (f_center - f_left + 1e-10)
            )

            right_mask = (bin_freqs_torch > f_center) & (bin_freqs_torch <= f_right)
            filterbank[i, right_mask] = (
                height * (f_right - bin_freqs_torch[right_mask]) / (f_right - f_center + 1e-10)
            )
        return filterbank.T.float()  # [n_mels, n_fft_bins]



    def make_slaney_vtln_mel_filterbank_np(
        self,
        sr=16000,
        n_fft=512,
        n_mels=80,
        fmin=20,
        fmax=None,
        vtln_low=100,
        vtln_high=7500,
        vtln_warp=1.0  # 1.0 = no warp, <1 = upward formant shift
    ):
        if fmax is None:
            fmax = sr / 2

        def hz_to_mel(f):
            return 2595 * np.log10(1 + f / 700)

        def mel_to_hz(m):
            return 700 * (10**(m / 2595) - 1)

        def vtln_warp_freq(freq, low_cutoff, high_cutoff, low_freq, high_freq, warp_factor):
            if freq < low_freq or freq > high_freq:
                return freq
            scale = 1.0 / warp_factor
            Fl = scale * low_cutoff
            Fh = scale * high_cutoff
            scale_left = (Fl - low_freq) / (low_cutoff - low_freq)
            scale_right = (high_freq - Fh) / (high_freq - high_cutoff)
            if freq < low_cutoff:
                return low_freq + scale_left * (freq - low_freq)
            elif freq <= high_cutoff:
                # return scale * freq
                return freq ** (warp_factor)
            else:
                return high_freq + scale_right * (freq - high_freq)

        # Compute mel-scale bin edges, convert to Hz, then warp
        mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2, dtype=np.float64)

        hz_edges = mel_to_hz(mel_edges)
        warped_hz_edges = np.array([
            vtln_warp_freq(f, vtln_low, vtln_high, fmin, fmax, vtln_warp)
            for f in hz_edges
        ])

        bin_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        filterbank = np.zeros((n_mels, len(bin_freqs)))

        for i in range(n_mels):
            f_left = warped_hz_edges[i]
            f_center = warped_hz_edges[i + 1]
            f_right = warped_hz_edges[i + 2]

            # Slaney-style height = 2 / (f_right - f_left)
            height = 2.0 / (f_right - f_left + 1e-10)

            # Mask left side
            left_mask = (bin_freqs >= f_left) & (bin_freqs <= f_center)
            filterbank[i, left_mask] = (
                height * (bin_freqs[left_mask] - f_left) / (f_center - f_left + 1e-10)
            )

            # Mask right side
            right_mask = (bin_freqs > f_center) & (bin_freqs <= f_right)
            filterbank[i, right_mask] = (
                height * (f_right - bin_freqs[right_mask]) / (f_right - f_center + 1e-10)
            )
        return torch.from_numpy(filterbank.T).float()  # [n_mels, n_fft_bins]


