# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from layers.Conv_Blocks import DilatedConvEncoder
# # import numpy as np


# # class Model(nn.Module):
# #     def __init__(self, configs, hidden_dims=128, output_dims=320, kernel_size=3):
# #         super(Model, self).__init__()
# #         self.task_name = configs.task_name
# #         self.seq_len = configs.seq_len
# #         self.pred_len = configs.pred_len
# #         self.output_attention = configs.output_attention

# #         self.encoder = DilatedConvEncoder(
# #             configs.enc_in,
# #             [hidden_dims] * configs.e_layers + [output_dims],  # a list here
# #             kernel_size=kernel_size,
# #         )

# #         # Decoder
# #         if (
# #             self.task_name == "long_term_forecast"
# #             or self.task_name == "short_term_forecast"
# #         ):
# #             raise NotImplementedError
# #         if self.task_name == "imputation":
# #             raise NotImplementedError
# #         if self.task_name == "anomaly_detection":
# #             raise NotImplementedError
# #         if self.task_name == "classification":
# #             self.act = F.gelu
# #             self.dropout = nn.Dropout(configs.dropout)
# #             self.projection = nn.Linear(output_dims, configs.num_class)

# #     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
# #         raise NotImplementedError

# #     def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
# #         raise NotImplementedError

# #     def anomaly_detection(self, x_enc):
# #         raise NotImplementedError

# #     def classification(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
# #         # conv encoder
# #         output = self.dropout(
# #             self.encoder(x_enc.transpose(1, 2))
# #         )  # (batch_size, output_dims, timestamps)
# #         output = output.transpose(1, 2)  # (batch_size, timestamps, hidden_dims)

# #         output = F.max_pool1d(
# #             output.transpose(1, 2), kernel_size=output.size(1)
# #         ).transpose(1, 2)  # (batch_size, 1, output_dims)
# #         output = output.squeeze(1)  # (batch_size, output_dims)
# #         output = self.projection(output)  # (batch_size, num_classes)
# #         return output

# #     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
# #         if (
# #             self.task_name == "long_term_forecast"
# #             or self.task_name == "short_term_forecast"
# #         ):
# #             raise NotImplementedError
# #         if self.task_name == "imputation":
# #             raise NotImplementedError
# #         if self.task_name == "anomaly_detection":
# #             raise NotImplementedError
# #         if self.task_name == "classification":
# #             dec_out = self.classification(x_enc, x_mark_enc)
# #             return dec_out  # [B, N]
# #         return None

    
    

    

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import os
# import warnings

# try:
#     import pywt  # type: ignore[import-not-found]
# except Exception:
#     pywt = None

# try:
#     from scipy.signal import savgol_filter  # type: ignore[import-not-found]
# except Exception:
#     savgol_filter = None


# class BidirectionalCausalConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation_forward=1, dilation_backward=1, groups=1):
#         super().__init__()
#         # Compute the padding size required for causality
#         self.padding_forward = (kernel_size - 1) * dilation_forward
#         self.padding_backward = (kernel_size - 1) * dilation_backward
        
#         self.conv_forward = nn.Conv1d(
#             in_channels, out_channels, kernel_size, padding=0, dilation=dilation_forward, groups=groups
#         )
#         self.conv_backward = nn.Conv1d(
#             in_channels, out_channels, kernel_size, padding=0, dilation=dilation_backward, groups=groups
#         )

#     def forward(self, x):
#         # Only left-side padding is required for causality
#         x_forward = F.pad(x, (self.padding_forward, 0))  # Padding for forward direction
#         x_backward = F.pad(x.flip(-1), (self.padding_backward, 0))  # Flip input for backward direction

#         # Perform convolution in both directions
#         out_forward = self.conv_forward(x_forward)
#         out_backward = self.conv_backward(x_backward)

#         # Combine forward and backward outputs (concatenation or summation)
#         out = out_forward + out_backward.flip(-1)  # Flip the backward output to match the original sequence length
#         return out


# class BidirectionalDilatedConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation_forward, dilation_backward, final=False):
#         super().__init__()
#         self.conv1 = BidirectionalCausalConv(
#             in_channels, out_channels, kernel_size, dilation_forward=dilation_forward, dilation_backward=dilation_backward
#         )
#         self.conv2 = BidirectionalCausalConv(
#             out_channels, out_channels, kernel_size, dilation_forward=dilation_forward, dilation_backward=dilation_backward
#         )
#         self.projector = (
#             nn.Conv1d(in_channels, out_channels, 1)
#             if in_channels != out_channels or final
#             else None
#         )

#     def forward(self, x):
#         residual = x if self.projector is None else self.projector(x)
#         x = F.gelu(x)
#         x = self.conv1(x)
#         x = F.gelu(x)
#         x = self.conv2(x)
#         return x + residual


# class BidirectionalDilatedConvEncoder(nn.Module):
#     def __init__(self, in_channels, channels, kernel_size):
#         super().__init__()
#         self.net = nn.Sequential(
#             *[
#                 BidirectionalDilatedConvBlock(
#                     channels[i - 1] if i > 0 else in_channels,
#                     channels[i],
#                     kernel_size=kernel_size,
#                     # dilation_forward=2**(i),  # Forward dilation
#                     dilation_forward=2**(len(channels)-i-1), 
#                     dilation_backward=2**(len(channels)-i-1),  # Backward dilation (can be adjusted as needed)
#                     # dilation_forward=2**i, 
#                     # dilation_backward=2**i,  # Backward dilation (can be adjusted as needed)
#                     final=(i == len(channels) - 1),
#                 )
#                 for i in range(len(channels))
#             ]
#         )

#     def forward(self, x):
#         return self.net(x)


# class Model(nn.Module):
#     def __init__(self, configs, hidden_dims=128, output_dims=320, kernel_size=3):
#         super(Model, self).__init__()
#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#         self.t = configs.t
#         self.n = configs.n
#         # self.a = configs.a
#         # self.b = configs.b

#         if configs.learnab:
#             self.a = nn.Parameter(torch.tensor(configs.a))  # �������� 0.0
#             self.b = nn.Parameter(torch.tensor(configs.b))  # �������� 0.0
#             # Print initial learnable parameters (useful for per-seed verification).
#             # print(f"[HMBiTCN] init learnab=True a={self.a.item():.6f}, b={self.b.item():.6f}")
#         else:
#             self.a = configs.a
#             self.b = configs.b

#         # -------------------------
#         # SNR evaluation (optional)
#         # -------------------------
#         self.snr_enabled = bool(getattr(configs, "log_snr", False))
#         self.snr_log_interval = max(1, int(getattr(configs, "snr_log_interval", 1)))
#         self.snr_changed_set = "front" if self.t > 0 else "back"
#         self.sampling_rate = float(getattr(configs, "sampling_rate", 256))
#         # Default: generic EEG-like bands; override via configs if needed.
#         self.snr_signal_band = (
#             float(getattr(configs, "snr_signal_band_low", 0.5)),
#             float(getattr(configs, "snr_signal_band_high", 40)),
#         )
#         self.snr_noise_band = (
#             float(getattr(configs, "snr_noise_band_low", 70)),
#             float(getattr(configs, "snr_noise_band_high", 100)),
#         )
#         self.denoise_method = str(getattr(configs, "denoise_method", "fusion")).lower()
#         self.denoise_line_freq = float(getattr(configs, "denoise_line_freq", 50.0))
#         self.denoise_notch_bw = float(getattr(configs, "denoise_notch_bw", 1.0))
#         self.denoise_notch_harmonics = max(1, int(getattr(configs, "denoise_notch_harmonics", 1)))
#         self.denoise_bp_low = float(getattr(configs, "denoise_bp_low", 1.0))
#         self.denoise_bp_high = float(getattr(configs, "denoise_bp_high", 40.0))
#         self.denoise_median_kernel = max(3, int(getattr(configs, "denoise_median_kernel", 5)))
#         if self.denoise_median_kernel % 2 == 0:
#             self.denoise_median_kernel += 1
#         self.denoise_savgol_window = max(5, int(getattr(configs, "denoise_savgol_window", 11)))
#         if self.denoise_savgol_window % 2 == 0:
#             self.denoise_savgol_window += 1
#         self.denoise_savgol_polyorder = max(1, int(getattr(configs, "denoise_savgol_polyorder", 2)))
#         self.denoise_wavelet = str(getattr(configs, "denoise_wavelet", "db4"))
#         self.denoise_wavelet_level = max(1, int(getattr(configs, "denoise_wavelet_level", 3)))
#         self._denoise_warned = False
#         self._snr_reset()
#         self.reset_snr_seed_stats()
#         # Optional per-iter SNR logging to txt.
#         # The file path will be configured by the training loop (Exp_Classification).
#         self.snr_iter_txt_path = None
#         self._snr_iter_txt_fp = None
#         self._snr_iter_txt_header_written = False

#         self.encoder = BidirectionalDilatedConvEncoder(
#             configs.enc_in,
#             [hidden_dims] * configs.e_layers + [output_dims],  # a list here
#             kernel_size=kernel_size,
#         )

#         # Decoder
#         if (
#             self.task_name == "long_term_forecast"
#             or self.task_name == "short_term_forecast"
#         ):
#             raise NotImplementedError
#         if self.task_name == "imputation":
#             raise NotImplementedError
#         if self.task_name == "anomaly_detection":
#             raise NotImplementedError
#         if self.task_name == "classification":
#             self.act = F.gelu
#             self.dropout = nn.Dropout(configs.dropout)
#             self.projection = nn.Linear(output_dims, configs.num_class)
        
#         # 可学习的通道权重矩阵
#         # self.channel_weights = nn.Parameter(torch.eye(16) + torch.randn(16, 16) * 0.01)

#     def classification(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
#         output = self.dropout(self.encoder(x_enc.transpose(1, 2)))  # (batch_size, output_dims, timestamps)
#         output = output.transpose(1, 2)  # (batch_size, timestamps, output_dims)
#         output = F.max_pool1d(output.transpose(1, 2), kernel_size=output.size(1)).transpose(1, 2)
#         output = output.squeeze(1)  # (batch_size, output_dims)
#         output = self.projection(output)  # (batch_size, num_classes)
#         return output

#     # -------------------------
#     # Helper: changed channels
#     # -------------------------
#     def _changed_channels(self, x):
#         # x: [B, T, C]
#         n = self.n
#         if self.t > 0:
#             return x[:, :, :n]
#         return x[:, :, -n:]

#     def _set_changed_channels(self, x_full, changed):
#         if self.t > 0:
#             x_full[:, :, : self.n] = changed
#         else:
#             x_full[:, :, -self.n :] = changed
#         return x_full

#     def _warn_once(self, msg):
#         if not self._denoise_warned:
#             warnings.warn(msg)
#             self._denoise_warned = True

#     def _apply_bandpass_notch(self, x):
#         # x: [B, T, C]
#         t_len = x.size(1)
#         if t_len < 3:
#             return x
#         x_fft = torch.fft.rfft(x, dim=1)
#         freqs = torch.fft.rfftfreq(t_len, d=1.0 / self.sampling_rate).to(x.device)
#         mask = (freqs >= self.denoise_bp_low) & (freqs <= self.denoise_bp_high)
#         for h in range(1, self.denoise_notch_harmonics + 1):
#             f0 = self.denoise_line_freq * h
#             notch = (freqs >= (f0 - self.denoise_notch_bw)) & (freqs <= (f0 + self.denoise_notch_bw))
#             mask = mask & (~notch)
#         x_fft = x_fft * mask.view(1, -1, 1).to(x_fft.dtype)
#         return torch.fft.irfft(x_fft, n=t_len, dim=1)

#     def _apply_median(self, x):
#         # x: [B, T, C]
#         if self.denoise_median_kernel <= 1:
#             return x
#         pad = self.denoise_median_kernel // 2
#         x_bct = x.permute(0, 2, 1)
#         x_pad = F.pad(x_bct, (pad, pad), mode="reflect")
#         x_unf = x_pad.unfold(dimension=2, size=self.denoise_median_kernel, step=1)
#         x_med = x_unf.median(dim=-1).values
#         return x_med.permute(0, 2, 1)

#     def _apply_savgol(self, x):
#         # x: [B, T, C]
#         if savgol_filter is None:
#             self._warn_once("scipy is not available, savgol falls back to moving average.")
#             k = self.denoise_savgol_window
#             x_bct = x.permute(0, 2, 1)
#             x_sm = F.avg_pool1d(
#                 F.pad(x_bct, (k // 2, k // 2), mode="reflect"),
#                 kernel_size=k,
#                 stride=1,
#             )
#             return x_sm.permute(0, 2, 1)

#         x_np = x.detach().cpu().numpy()
#         b, t, c = x_np.shape
#         win = min(self.denoise_savgol_window, t if t % 2 == 1 else t - 1)
#         win = max(5, win)
#         if win % 2 == 0:
#             win -= 1
#         poly = min(self.denoise_savgol_polyorder, win - 1)
#         out = x_np.copy()
#         for bi in range(b):
#             for ci in range(c):
#                 out[bi, :, ci] = savgol_filter(x_np[bi, :, ci], win, poly, mode="interp")
#         return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)

#     def _apply_wavelet(self, x):
#         # x: [B, T, C]
#         if pywt is None:
#             self._warn_once("pywt is not available, wavelet denoise falls back to identity.")
#             return x
#         x_np = x.detach().cpu().numpy()
#         b, t, c = x_np.shape
#         out = x_np.copy()
#         wave = self.denoise_wavelet
#         for bi in range(b):
#             for ci in range(c):
#                 sig = x_np[bi, :, ci]
#                 coeffs = pywt.wavedec(sig, wave, level=self.denoise_wavelet_level, mode="symmetric")
#                 detail = coeffs[-1]
#                 sigma = float((abs(detail)).mean() / 0.6745) if detail.size > 0 else 0.0
#                 uth = sigma * math.sqrt(2.0 * math.log(max(t, 2)))
#                 coeffs_d = [coeffs[0]]
#                 for d in coeffs[1:]:
#                     coeffs_d.append(pywt.threshold(d, value=uth, mode="soft"))
#                 rec = pywt.waverec(coeffs_d, wave, mode="symmetric")
#                 out[bi, :, ci] = rec[:t]
#         return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)

#     def _apply_denoise(self, x):
#         method = self.denoise_method
#         if method in {"none", "off", ""}:
#             return x
#         if method == "bandpass_notch":
#             return self._apply_bandpass_notch(x)
#         if method == "wavelet_db4":
#             return self._apply_wavelet(x)
#         if method == "median":
#             return self._apply_median(x)
#         if method == "savgol":
#             return self._apply_savgol(x)
#         self._warn_once(f"Unknown denoise_method={method}, fallback to identity.")
#         return x

#     def set_snr_iter_txt_path(self, txt_path: str):
#         """Configure per-iter SNR txt logging file (called from training loop)."""
#         self.snr_iter_txt_path = txt_path
#         # Reset file handle so the new path takes effect immediately.
#         self._snr_iter_txt_fp = None
#         self._snr_iter_txt_header_written = False

#     def _snr_open_iter_txt_if_needed(self):
#         if not self.snr_enabled:
#             return
#         if not self.snr_iter_txt_path:
#             return
#         if self._snr_iter_txt_fp is not None:
#             return
#         # Ensure parent directory exists (best-effort).
#         os.makedirs(os.path.dirname(self.snr_iter_txt_path), exist_ok=True)
#         self._snr_iter_txt_fp = open(self.snr_iter_txt_path, "w", encoding="utf-8")
#         self._snr_iter_txt_header_written = False

#     def _snr_write_iter_txt(
#         self,
#         *,
#         iter_idx: int,
#         trial_delta_db: float,
#         trial_imp_count: int,
#         trial_imp_den: int,
#         trial_imp_ratio: float,
#         trial_before_ch_list: list,
#         trial_after_ch_list: list,
#         trial_delta_ch_list: list,
#         band_delta_db: float,
#         band_imp_count: int,
#         band_imp_den: int,
#         band_imp_ratio: float,
#         band_before_ch_list: list,
#         band_after_ch_list: list,
#         band_delta_ch_list: list,
#     ):
#         if not self.snr_enabled:
#             return
#         if not self.snr_iter_txt_path:
#             return
#         self._snr_open_iter_txt_if_needed()
#         if self._snr_iter_txt_fp is None:
#             return
#         fp = self._snr_iter_txt_fp
#         if not self._snr_iter_txt_header_written:
#             fp.write(
#                 "iter\ttrial_delta_db\ttrial_improved_ch_count\ttrial_improved_ch_den\ttrial_improved_ratio\t"
#                 "trial_before_ch_dB_list\ttrial_after_ch_dB_list\ttrial_delta_ch_dB_list\t"
#                 "band_delta_db\tband_improved_ch_count\tband_improved_ch_den\tband_improved_ratio\t"
#                 "band_before_ch_dB_list\tband_after_ch_dB_list\tband_delta_ch_dB_list\n"
#             )
#             fp.flush()
#             self._snr_iter_txt_header_written = True
#         fp.write(
#             f"{iter_idx}\t{trial_delta_db:.6f}\t{trial_imp_count}\t{trial_imp_den}\t{trial_imp_ratio:.6f}\t"
#             f"{','.join(trial_before_ch_list)}\t{','.join(trial_after_ch_list)}\t{','.join(trial_delta_ch_list)}\t"
#             f"{band_delta_db:.6f}\t{band_imp_count}\t{band_imp_den}\t{band_imp_ratio:.6f}\t"
#             f"{','.join(band_before_ch_list)}\t{','.join(band_after_ch_list)}\t{','.join(band_delta_ch_list)}\n"
#         )
#         fp.flush()

#     # -------------------------
#     # Trial-averaging SNR
#     # -------------------------
#     def _trial_avg_snr_db(self, x):
#         """
#         x: [B, T, C] (changed channels)
#         signal = mean over trials (batch)
#         noise  = trial residual (x - mean)
#         SNR(dB) = 10*log10(Var(signal)/Var(noise))
#         """
#         if x.size(0) < 2:
#             # Avoid degenerate noise=0 case; return NaN so it won't affect epoch stats.
#             return torch.tensor(float("nan"), device=x.device)
#         signal = x.mean(dim=0, keepdim=True)
#         noise = x - signal
#         signal_power = signal.var(unbiased=False).clamp_min(1e-12)
#         noise_power = noise.var(unbiased=False).clamp_min(1e-12)
#         return 10.0 * torch.log10(signal_power / noise_power)

#     # -------------------------
#     # Band-power ratio (proxy SNR)
#     # -------------------------
#     def _band_power_ratio_db(self, x):
#         """
#         x: [B, T, C] (changed channels)
#         SNR(dB) = 10*log10( P(signal_band) / P(noise_band) )
#         where P is averaged power spectral density over B and C.
#         """
#         t_len = x.size(1)
#         if t_len < 2:
#             return torch.tensor(float("nan"), device=x.device)

#         x_fft = torch.fft.rfft(x, dim=1)
#         power = x_fft.abs().pow(2).mean(dim=(0, 2))  # [F]
#         freqs = torch.fft.rfftfreq(t_len, d=1.0 / self.sampling_rate).to(x.device)

#         s_low, s_high = self.snr_signal_band
#         n_low, n_high = self.snr_noise_band
#         signal_mask = (freqs >= s_low) & (freqs < s_high)
#         noise_mask = (freqs >= n_low) & (freqs < n_high)
#         if signal_mask.sum() == 0 or noise_mask.sum() == 0:
#             return torch.tensor(float("nan"), device=x.device)

#         signal_power = power[signal_mask].mean().clamp_min(1e-12)
#         noise_power = power[noise_mask].mean().clamp_min(1e-12)
#         return 10.0 * torch.log10(signal_power / noise_power)

#     # -------------------------
#     # Epoch-level accumulators
#     # -------------------------
#     def _snr_reset(self):
#         self._snr_count = 0
#         self._snr_iter_count = 0
#         self._trial_before_sum = 0.0
#         self._trial_after_sum = 0.0
#         self._trial_delta_sum = 0.0
#         self._trial_delta_sumsq = 0.0
#         self._trial_improved_count = 0

#         self._band_before_sum = 0.0
#         self._band_after_sum = 0.0
#         self._band_delta_sum = 0.0
#         self._band_delta_sumsq = 0.0
#         self._band_improved_count = 0

#         # Per-channel accumulators for changed channels (length n).
#         # trial delta: trial_snr_after_per_channel - trial_snr_before_per_channel
#         dev = next(self.parameters()).device
#         n = self.n
#         self._trial_delta_ch_sum = torch.zeros(n, device=dev)
#         self._trial_delta_ch_sumsq = torch.zeros(n, device=dev)
#         self._trial_delta_ch_count = torch.zeros(n, device=dev)
#         self._trial_delta_ch_pos = torch.zeros(n, device=dev)

#         self._band_delta_ch_sum = torch.zeros(n, device=dev)
#         self._band_delta_ch_sumsq = torch.zeros(n, device=dev)
#         self._band_delta_ch_count = torch.zeros(n, device=dev)
#         self._band_delta_ch_pos = torch.zeros(n, device=dev)

#         self._last_total_ch = None

#     def reset_snr_epoch_stats(self):
#         if not self.snr_enabled:
#             return
#         self._snr_reset()

#     # -------------------------
#     # Seed-level accumulators (across all epochs/iters of one seed)
#     # -------------------------
#     def reset_snr_seed_stats(self):
#         if not self.snr_enabled:
#             return
#         dev = next(self.parameters()).device
#         n = self.n
#         self._seed_delta_trial_ch_sum = torch.zeros(n, device=dev)
#         self._seed_delta_trial_ch_count = torch.zeros(n, device=dev)
#         self._seed_delta_trial_ch_pos_count = torch.zeros(n, device=dev)
#         self._seed_last_total_ch = None

#     @torch.no_grad()
#     def _update_snr_seed_stats_from_precomputed(self, delta_trial_ch, total_ch):
#         """
#         delta_trial_ch: [n] dB, for changed channels only.
#         """
#         if not self.snr_enabled:
#             return
#         self._seed_last_total_ch = total_ch
#         finite = torch.isfinite(delta_trial_ch)
#         if finite.any():
#             m = finite.to(delta_trial_ch.dtype)
#             self._seed_delta_trial_ch_sum += delta_trial_ch * m
#             self._seed_delta_trial_ch_count += m
#             self._seed_delta_trial_ch_pos_count += (
#                 (delta_trial_ch > 0).to(delta_trial_ch.dtype) * m
#             )

#     def get_snr_seed_stats(self):
#         """
#         Return number of changed channels whose mean trial SNR delta > 0.
#         """
#         if not self.snr_enabled or not hasattr(self, "_seed_delta_trial_ch_sum"):
#             return None

#         n = self.n
#         count_safe = self._seed_delta_trial_ch_count.clamp_min(1.0)
#         mean_delta_ch = self._seed_delta_trial_ch_sum / count_safe  # [n]

#         improved_mask = (self._seed_delta_trial_ch_count > 0) & (mean_delta_ch > 0)
#         improved_count = int(improved_mask.sum().detach().cpu().item())
#         improved_ratio = improved_count / max(n, 1)

#         total_ch = self._seed_last_total_ch if self._seed_last_total_ch is not None else self.n
#         # local->global mapping
#         if self.t > 0:
#             map_idx = lambda local_idx: int(local_idx)
#         else:
#             map_idx = lambda local_idx: int(total_ch - self.n + local_idx)

#         improved_local_idx = torch.nonzero(improved_mask, as_tuple=False).squeeze(1).detach().cpu().tolist()
#         improved_global_idx = [map_idx(li) for li in improved_local_idx]

#         # include mean values for the improved channels (optional small list)
#         improved_mean_vals = [float(mean_delta_ch[li].detach().cpu().item()) for li in improved_local_idx]

#         return {
#             "improved_count": improved_count,
#             "improved_ratio": improved_ratio,
#             "improved_global_idx": improved_global_idx,
#             "improved_mean_delta_ch": improved_mean_vals,
#             "n_changed_channels": n,
#         }

#     @torch.no_grad()
#     def _update_snr_epoch_stats(self, x_enc, x_enc_new):
#         if not self.snr_enabled:
#             return
#         x_b = self._changed_channels(x_enc)
#         x_a = self._changed_channels(x_enc_new)

#         # Store total channel count for mapping local->global indices.
#         self._last_total_ch = x_enc.size(2)

#         trial_before = self._trial_avg_snr_db(x_b)
#         trial_after = self._trial_avg_snr_db(x_a)
#         band_before = self._band_power_ratio_db(x_b)
#         band_after = self._band_power_ratio_db(x_a)

#         if torch.isfinite(trial_before) and torch.isfinite(trial_after):
#             delta_trial = trial_after - trial_before
#             self._trial_before_sum += float(trial_before.detach().cpu().item())
#             self._trial_after_sum += float(trial_after.detach().cpu().item())
#             self._trial_delta_sum += float(delta_trial.detach().cpu().item())
#             self._trial_delta_sumsq += float(delta_trial.detach().cpu().item() ** 2)
#             self._trial_improved_count += int(delta_trial.item() > 0)

#         if torch.isfinite(band_before) and torch.isfinite(band_after):
#             delta_band = band_after - band_before
#             self._band_before_sum += float(band_before.detach().cpu().item())
#             self._band_after_sum += float(band_after.detach().cpu().item())
#             self._band_delta_sum += float(delta_band.detach().cpu().item())
#             self._band_delta_sumsq += float(delta_band.detach().cpu().item() ** 2)
#             self._band_improved_count += int(delta_band.item() > 0)

#         # Per-channel trial-avg SNR and band-ratio deltas (for changed channels only)
#         # trial per channel
#         trial_before_ch = self._trial_avg_snr_db_per_channel(x_b)
#         trial_after_ch = self._trial_avg_snr_db_per_channel(x_a)
#         delta_trial_ch = trial_after_ch - trial_before_ch  # [n]
#         finite = torch.isfinite(delta_trial_ch)
#         if finite.any():
#             m = finite.to(delta_trial_ch.dtype)
#             self._trial_delta_ch_sum += (delta_trial_ch * m)
#             self._trial_delta_ch_sumsq += (delta_trial_ch * m).pow(2)
#             self._trial_delta_ch_count += m
#             self._trial_delta_ch_pos += ((delta_trial_ch > 0).to(delta_trial_ch.dtype) * m)

#         # band-ratio per channel
#         band_before_ch = self._band_power_ratio_db_per_channel(x_b)
#         band_after_ch = self._band_power_ratio_db_per_channel(x_a)
#         delta_band_ch = band_after_ch - band_before_ch
#         finite_b = torch.isfinite(delta_band_ch)
#         if finite_b.any():
#             m = finite_b.to(delta_band_ch.dtype)
#             self._band_delta_ch_sum += (delta_band_ch * m)
#             self._band_delta_ch_sumsq += (delta_band_ch * m).pow(2)
#             self._band_delta_ch_count += m
#             self._band_delta_ch_pos += ((delta_band_ch > 0).to(delta_band_ch.dtype) * m)

#         # Count as one update if either metric is valid.
#         if (torch.isfinite(trial_before) and torch.isfinite(trial_after)) or (
#             torch.isfinite(band_before) and torch.isfinite(band_after)
#         ):
#             self._snr_count += 1

#     @torch.no_grad()
#     def _update_snr_epoch_stats_from_precomputed(
#         self,
#         trial_before,
#         trial_after,
#         band_before,
#         band_after,
#         delta_trial_ch,
#         delta_band_ch,
#         total_ch,
#     ):
#         """
#         Update epoch accumulators using metrics computed in the current forward.
#         delta_trial_ch / delta_band_ch are in dB and correspond to changed channels only.
#         """
#         if not self.snr_enabled:
#             return

#         self._last_total_ch = total_ch

#         # Scalar accumulators
#         if torch.isfinite(trial_before) and torch.isfinite(trial_after):
#             delta_trial = trial_after - trial_before
#             self._trial_before_sum += float(trial_before.detach().cpu().item())
#             self._trial_after_sum += float(trial_after.detach().cpu().item())
#             self._trial_delta_sum += float(delta_trial.detach().cpu().item())
#             self._trial_delta_sumsq += float(delta_trial.detach().cpu().item() ** 2)
#             self._trial_improved_count += int(delta_trial.item() > 0)

#         if torch.isfinite(band_before) and torch.isfinite(band_after):
#             delta_band = band_after - band_before
#             self._band_before_sum += float(band_before.detach().cpu().item())
#             self._band_after_sum += float(band_after.detach().cpu().item())
#             self._band_delta_sum += float(delta_band.detach().cpu().item())
#             self._band_delta_sumsq += float(delta_band.detach().cpu().item() ** 2)
#             self._band_improved_count += int(delta_band.item() > 0)

#         # Per-channel accumulators
#         finite_t = torch.isfinite(delta_trial_ch)
#         if finite_t.any():
#             m = finite_t.to(delta_trial_ch.dtype)
#             self._trial_delta_ch_sum += delta_trial_ch * m
#             self._trial_delta_ch_sumsq += (delta_trial_ch * m).pow(2)
#             self._trial_delta_ch_count += m
#             self._trial_delta_ch_pos += ((delta_trial_ch > 0).to(delta_trial_ch.dtype) * m)

#         finite_b = torch.isfinite(delta_band_ch)
#         if finite_b.any():
#             m = finite_b.to(delta_band_ch.dtype)
#             self._band_delta_ch_sum += delta_band_ch * m
#             self._band_delta_ch_sumsq += (delta_band_ch * m).pow(2)
#             self._band_delta_ch_count += m
#             self._band_delta_ch_pos += ((delta_band_ch > 0).to(delta_band_ch.dtype) * m)

#         # Count update if either scalar pair is valid.
#         if (torch.isfinite(trial_before) and torch.isfinite(trial_after)) or (
#             torch.isfinite(band_before) and torch.isfinite(band_after)
#         ):
#             self._snr_count += 1

#     def get_snr_epoch_stats(self):
#         if not self.snr_enabled or self._snr_count == 0:
#             return None

#         def _mean_std(sum_, sumsq_):
#             mean = sum_ / max(self._snr_count, 1)
#             var = sumsq_ / max(self._snr_count, 1) - mean * mean
#             var = max(var, 0.0)
#             std = math.sqrt(var)
#             return mean, std

#         trial_before_mean, _ = _mean_std(self._trial_before_sum, 0.0)
#         trial_after_mean, _ = _mean_std(self._trial_after_sum, 0.0)
#         trial_delta_mean, trial_delta_std = _mean_std(self._trial_delta_sum, self._trial_delta_sumsq)
#         trial_improved_ratio = self._trial_improved_count / max(self._snr_count, 1)

#         band_before_mean, _ = _mean_std(self._band_before_sum, 0.0)
#         band_after_mean, _ = _mean_std(self._band_after_sum, 0.0)
#         band_delta_mean, band_delta_std = _mean_std(self._band_delta_sum, self._band_delta_sumsq)
#         band_improved_ratio = self._band_improved_count / max(self._snr_count, 1)

#         # Per-channel stats
#         def _per_channel(mean_sum, sumsq, count, pos_count):
#             count_safe = count.clamp_min(1.0)
#             mean = mean_sum / count_safe
#             var = sumsq / count_safe - mean.pow(2)
#             var = torch.clamp(var, min=0.0)
#             std = torch.sqrt(var)
#             improved_ratio = pos_count / count_safe
#             return mean, std, improved_ratio

#         trial_mean_ch, trial_std_ch, trial_imp_ratio_ch = _per_channel(
#             self._trial_delta_ch_sum,
#             self._trial_delta_ch_sumsq,
#             self._trial_delta_ch_count,
#             self._trial_delta_ch_pos,
#         )
#         band_mean_ch, band_std_ch, band_imp_ratio_ch = _per_channel(
#             self._band_delta_ch_sum,
#             self._band_delta_ch_sumsq,
#             self._band_delta_ch_count,
#             self._band_delta_ch_pos,
#         )

#         # Map local changed-channel indices to global indices for interpretability.
#         total_ch = self._last_total_ch if self._last_total_ch is not None else self.n
#         if self.t > 0:
#             map_idx = lambda local_idx: int(local_idx)
#         else:
#             map_idx = lambda local_idx: int(total_ch - self.n + local_idx)

#         # top/bottom 3 by trial mean delta
#         k = min(3, self.n)
#         trial_sorted = torch.argsort(trial_mean_ch)
#         band_sorted = torch.argsort(band_mean_ch)

#         trial_bottom = trial_sorted[:k].detach().cpu().tolist()
#         trial_top = trial_sorted[-k:].detach().cpu().tolist()
#         band_bottom = band_sorted[:k].detach().cpu().tolist()
#         band_top = band_sorted[-k:].detach().cpu().tolist()

#         def _fmt(idx_list, mean_ch):
#             out = []
#             for li in idx_list:
#                 gi = map_idx(li)
#                 out.append(f"ch{gi}:{mean_ch[li].item():.3f}dB")
#             return out

#         trial_top_str = _fmt(trial_top, trial_mean_ch)
#         trial_bottom_str = _fmt(trial_bottom, trial_mean_ch)
#         band_top_str = _fmt(band_top, band_mean_ch)
#         band_bottom_str = _fmt(band_bottom, band_mean_ch)

#         # Full per-channel mean deltas (changed channels only).
#         trial_delta_all_str = []
#         band_delta_all_str = []
#         for li in range(self.n):
#             gi = map_idx(li)
#             tv = trial_mean_ch[li].item()
#             bv = band_mean_ch[li].item()
#             trial_delta_all_str.append(f"ch{gi}:{tv:.3f}dB" if torch.isfinite(trial_mean_ch[li]) else f"ch{gi}:nan")
#             band_delta_all_str.append(f"ch{gi}:{bv:.3f}dB" if torch.isfinite(band_mean_ch[li]) else f"ch{gi}:nan")

#         # Positive/negative/median summary on changed channels (robust evidence for "local gains").
#         finite_trial = torch.isfinite(trial_mean_ch)
#         trial_vals = trial_mean_ch[finite_trial]
#         trial_median = torch.median(trial_vals) if trial_vals.numel() > 0 else torch.tensor(float("nan"), device=trial_mean_ch.device)
#         trial_pos = trial_vals[trial_vals > 0]
#         trial_neg = trial_vals[trial_vals < 0]
#         trial_pos_mean = trial_pos.mean() if trial_pos.numel() > 0 else torch.tensor(float("nan"), device=trial_mean_ch.device)
#         trial_neg_mean = trial_neg.mean() if trial_neg.numel() > 0 else torch.tensor(float("nan"), device=trial_mean_ch.device)

#         finite_band = torch.isfinite(band_mean_ch)
#         band_vals = band_mean_ch[finite_band]
#         band_median = torch.median(band_vals) if band_vals.numel() > 0 else torch.tensor(float("nan"), device=band_mean_ch.device)
#         band_pos = band_vals[band_vals > 0]
#         band_neg = band_vals[band_vals < 0]
#         band_pos_mean = band_pos.mean() if band_pos.numel() > 0 else torch.tensor(float("nan"), device=band_mean_ch.device)
#         band_neg_mean = band_neg.mean() if band_neg.numel() > 0 else torch.tensor(float("nan"), device=band_mean_ch.device)

#         return {
#             "trial_before_mean": trial_before_mean,
#             "trial_after_mean": trial_after_mean,
#             "trial_delta_mean": trial_delta_mean,
#             "trial_delta_std": trial_delta_std,
#             "trial_improved_ratio": trial_improved_ratio,
#             "trial_mean_ch": trial_mean_ch.detach().cpu().tolist(),
#             "trial_improved_ratio_ch": trial_imp_ratio_ch.detach().cpu().tolist(),
#             "trial_top3": trial_top_str,
#             "trial_bottom3": trial_bottom_str,
#             "trial_delta_all": trial_delta_all_str,
#             "trial_pos_mean_ch": float(trial_pos_mean.detach().cpu().item()),
#             "trial_neg_mean_ch": float(trial_neg_mean.detach().cpu().item()),
#             "trial_median_ch": float(trial_median.detach().cpu().item()),
#             "band_before_mean": band_before_mean,
#             "band_after_mean": band_after_mean,
#             "band_delta_mean": band_delta_mean,
#             "band_delta_std": band_delta_std,
#             "band_improved_ratio": band_improved_ratio,
#             "band_mean_ch": band_mean_ch.detach().cpu().tolist(),
#             "band_improved_ratio_ch": band_imp_ratio_ch.detach().cpu().tolist(),
#             "band_top3": band_top_str,
#             "band_bottom3": band_bottom_str,
#             "band_delta_all": band_delta_all_str,
#             "band_pos_mean_ch": float(band_pos_mean.detach().cpu().item()),
#             "band_neg_mean_ch": float(band_neg_mean.detach().cpu().item()),
#             "band_median_ch": float(band_median.detach().cpu().item()),
#             "n_updates": self._snr_count,
#         }

#     # Per-channel helpers
#     def _trial_avg_snr_db_per_channel(self, x):
#         """
#         x: [B, T, C] (changed channels only, C=n)
#         Returns SNR(dB) per channel: [C]
#         """
#         if x.size(0) < 2:
#             return torch.full((x.size(2),), float("nan"), device=x.device)
#         signal = x.mean(dim=0, keepdim=False)  # [T, C]
#         noise = x - signal.unsqueeze(0)  # [B, T, C]
#         # Power across time for the mean waveform, and residual variance across trials+time.
#         signal_power = signal.var(dim=0, unbiased=False).clamp_min(1e-12)  # [C]
#         noise_power = noise.var(dim=(0, 1), unbiased=False).clamp_min(1e-12)  # [C]
#         return 10.0 * torch.log10(signal_power / noise_power)

#     def _band_power_ratio_db_per_channel(self, x):
#         """
#         x: [B, T, C] (changed channels only)
#         Returns band-ratio SNR(dB) per channel: [C]
#         """
#         t_len = x.size(1)
#         if t_len < 2:
#             return torch.full((x.size(2),), float("nan"), device=x.device)
#         x_fft = torch.fft.rfft(x, dim=1)  # [B,F,C]
#         power = x_fft.abs().pow(2).mean(dim=0)  # [F,C]
#         freqs = torch.fft.rfftfreq(t_len, d=1.0 / self.sampling_rate).to(x.device)
#         s_low, s_high = self.snr_signal_band
#         n_low, n_high = self.snr_noise_band
#         signal_mask = (freqs >= s_low) & (freqs < s_high)
#         noise_mask = (freqs >= n_low) & (freqs < n_high)
#         if signal_mask.sum() == 0 or noise_mask.sum() == 0:
#             return torch.full((x.size(2),), float("nan"), device=x.device)
#         signal_power = power[signal_mask, :].mean(dim=0).clamp_min(1e-12)
#         noise_power = power[noise_mask, :].mean(dim=0).clamp_min(1e-12)
#         return 10.0 * torch.log10(signal_power / noise_power)

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        
#         t = self.t
        
#         n = self.n
#         x_enc_new = x_enc.clone()
#         if self.denoise_method == "fusion":
#             front_8_half = x_enc[:, :, :n]
#             back_8_half = x_enc[:, :, -n:]
#             added_half = front_8_half * self.a + back_8_half * self.b
#             if t > 0:
#                 x_enc_new[:, :, :n] = added_half
#             else:
#                 x_enc_new[:, :, -n:] = added_half
#         else:
#             changed = self._changed_channels(x_enc)
#             changed_denoised = self._apply_denoise(changed)
#             x_enc_new = self._set_changed_channels(x_enc_new, changed_denoised)


        

        

#         if (
#             self.task_name == "long_term_forecast"
#             or self.task_name == "short_term_forecast"
#         ):
#             raise NotImplementedError
#         if self.task_name == "imputation":
#             raise NotImplementedError
#         if self.task_name == "anomaly_detection":
#             raise NotImplementedError
#         if self.task_name == "classification":
#             if self.snr_enabled:
#                 self._snr_iter_count += 1
#                 # Only print every snr_log_interval iterations (set to 1 to print all).
#                 should_print = (self._snr_iter_count % self.snr_log_interval == 0)

#                 x_b = self._changed_channels(x_enc)
#                 x_a = self._changed_channels(x_enc_new)

#                 trial_before = self._trial_avg_snr_db(x_b)
#                 trial_after = self._trial_avg_snr_db(x_a)
#                 band_before = self._band_power_ratio_db(x_b)
#                 band_after = self._band_power_ratio_db(x_a)

#                 trial_before_ch = self._trial_avg_snr_db_per_channel(x_b)
#                 trial_after_ch = self._trial_avg_snr_db_per_channel(x_a)
#                 delta_trial_ch = trial_after_ch - trial_before_ch  # [n]

#                 band_before_ch = self._band_power_ratio_db_per_channel(x_b)
#                 band_after_ch = self._band_power_ratio_db_per_channel(x_a)
#                 delta_band_ch = band_after_ch - band_before_ch  # [n]

#                 # Update epoch accumulators
#                 self._update_snr_epoch_stats_from_precomputed(
#                     trial_before=trial_before,
#                     trial_after=trial_after,
#                     band_before=band_before,
#                     band_after=band_after,
#                     delta_trial_ch=delta_trial_ch,
#                     delta_band_ch=delta_band_ch,
#                     total_ch=x_enc.size(2),
#                 )

#                 # Update seed-level channel improvement stats.
#                 self._update_snr_seed_stats_from_precomputed(
#                     delta_trial_ch=delta_trial_ch,
#                     total_ch=x_enc.size(2),
#                 )

#                 # Per-iter channel improvement stats (used for both txt + terminal).
#                 delta_trial = trial_after - trial_before
#                 delta_band = band_after - band_before

#                 vt = torch.isfinite(delta_trial_ch)
#                 vb = torch.isfinite(delta_band_ch)
#                 trial_imp_count = int((delta_trial_ch[vt] > 0).sum().detach().cpu().item()) if vt.any() else 0
#                 trial_imp_den = int(vt.sum().detach().cpu().item()) if vt.any() else 0
#                 trial_imp_ratio = float(((delta_trial_ch[vt] > 0).float().mean().item())) if vt.any() else float("nan")

#                 band_imp_count = int((delta_band_ch[vb] > 0).sum().detach().cpu().item()) if vb.any() else 0
#                 band_imp_den = int(vb.sum().detach().cpu().item()) if vb.any() else 0
#                 band_imp_ratio = float(((delta_band_ch[vb] > 0).float().mean().item())) if vb.any() else float("nan")

#                 # Write per-iter SNR statistics to a dedicated txt file.
#                 if self.snr_enabled and self.snr_iter_txt_path:
#                     # Store per-ch SNRs for the changed channels only.
#                     # We serialize as comma-separated strings to keep one iter per line.
#                     tb_list = []
#                     ta_list = []
#                     td_list = []
#                     bb_list = []
#                     ba_list = []
#                     bd_list = []
#                     for li in range(self.n):
#                         tv = float(trial_before_ch[li].detach().cpu().item())
#                         av = float(trial_after_ch[li].detach().cpu().item())
#                         dv = float(delta_trial_ch[li].detach().cpu().item())
#                         if math.isfinite(tv):
#                             tb_list.append(f"{tv:.3f}")
#                         else:
#                             tb_list.append("nan")
#                         if math.isfinite(av):
#                             ta_list.append(f"{av:.3f}")
#                         else:
#                             ta_list.append("nan")
#                         if math.isfinite(dv):
#                             td_list.append(f"{dv:.3f}")
#                         else:
#                             td_list.append("nan")

#                         bv = float(band_before_ch[li].detach().cpu().item())
#                         avb = float(band_after_ch[li].detach().cpu().item())
#                         dbv = float(delta_band_ch[li].detach().cpu().item())
#                         if math.isfinite(bv):
#                             bb_list.append(f"{bv:.3f}")
#                         else:
#                             bb_list.append("nan")
#                         if math.isfinite(avb):
#                             ba_list.append(f"{avb:.3f}")
#                         else:
#                             ba_list.append("nan")
#                         if math.isfinite(dbv):
#                             bd_list.append(f"{dbv:.3f}")
#                         else:
#                             bd_list.append("nan")

#                     self._snr_write_iter_txt(
#                         iter_idx=int(self._snr_iter_count),
#                         trial_delta_db=float(delta_trial.detach().cpu().item()),
#                         trial_imp_count=trial_imp_count,
#                         trial_imp_den=trial_imp_den,
#                         trial_imp_ratio=trial_imp_ratio,
#                         trial_before_ch_list=tb_list,
#                         trial_after_ch_list=ta_list,
#                         trial_delta_ch_list=td_list,
#                         band_delta_db=float(delta_band.detach().cpu().item()),
#                         band_imp_count=band_imp_count,
#                         band_imp_den=band_imp_den,
#                         band_imp_ratio=band_imp_ratio,
#                         band_before_ch_list=bb_list,
#                         band_after_ch_list=ba_list,
#                         band_delta_ch_list=bd_list,
#                     )

#                 if should_print:

#                     k = min(3, self.n)

#                     # Robust top/bottom: argsort only on finite channels.
#                     trial_bottom = []
#                     trial_top = []
#                     band_bottom = []
#                     band_top = []
#                     if vt.any():
#                         trial_idxs = torch.nonzero(vt, as_tuple=False).squeeze(1)  # [M]
#                         trial_vals = delta_trial_ch[trial_idxs]  # [M]
#                         order = torch.argsort(trial_vals)
#                         k_eff = min(k, trial_idxs.numel())
#                         trial_bottom = trial_idxs[order[:k_eff]].detach().cpu().tolist()
#                         trial_top = trial_idxs[order[-k_eff:]].detach().cpu().tolist()

#                     if vb.any():
#                         band_idxs = torch.nonzero(vb, as_tuple=False).squeeze(1)  # [M]
#                         band_vals = delta_band_ch[band_idxs]  # [M]
#                         order = torch.argsort(band_vals)
#                         k_eff = min(k, band_idxs.numel())
#                         band_bottom = band_idxs[order[:k_eff]].detach().cpu().tolist()
#                         band_top = band_idxs[order[-k_eff:]].detach().cpu().tolist()

#                     total_ch = x_enc.size(2)
#                     if self.t > 0:
#                         map_idx = lambda li: int(li)
#                     else:
#                         map_idx = lambda li: int(total_ch - self.n + li)

#                     def _fmt_local(idx_list, delta_ch):
#                         out = []
#                         for li in idx_list:
#                             gi = map_idx(li)
#                             out.append(f"ch{gi}:{delta_ch[li].item():.3f}dB")
#                         return out

#                     trial_top_str = _fmt_local(trial_top, delta_trial_ch)
#                     trial_bottom_str = _fmt_local(trial_bottom, delta_trial_ch)
#                     band_top_str = _fmt_local(band_top, delta_band_ch)
#                     band_bottom_str = _fmt_local(band_bottom, delta_band_ch)

#                     print(
#                         f"[SNR-iter] iter={self._snr_iter_count} "
#                         f"trial Δ={delta_trial.item():.6f} "
#                         f"(before={trial_before.item():.6f}, after={trial_after.item():.6f}, improved_ratio={trial_imp_ratio:.3f}) | "
#                         f"band Δ={delta_band.item():.6f} (improved_ratio={band_imp_ratio:.3f})"
#                     )
#                     print(
#                         f"[SNR-iter-ch-count] trial_improved={trial_imp_count}/{trial_imp_den} "
#                         f"(ratio={trial_imp_ratio:.3f}) | band_improved={band_imp_count}/{band_imp_den} "
#                         f"(ratio={band_imp_ratio:.3f})"
#                     )
#                     print(
#                         f"[SNR-iter-ch] trial_top3={','.join(trial_top_str)} trial_bottom3={','.join(trial_bottom_str)} | "
#                         f"band_top3={','.join(band_top_str)} band_bottom3={','.join(band_bottom_str)}"
#                     )

#                     # Print full per-ch deltas (changed channels only).
#                     # Avoid extremely long logs when n is large.
#                     if self.n <= 16:
#                         trial_all = []
#                         band_all = []
#                         total_ch = x_enc.size(2)
#                         if self.t > 0:
#                             map_idx_all = lambda li: int(li)
#                         else:
#                             map_idx_all = lambda li: int(total_ch - self.n + li)
#                         for li in range(self.n):
#                             gi = map_idx_all(li)
#                             vt = delta_trial_ch[li].item()
#                             vb = delta_band_ch[li].item()
#                             if math.isfinite(vt):
#                                 trial_all.append(f"ch{gi}:{vt:.3f}dB")
#                             else:
#                                 trial_all.append(f"ch{gi}:nan")
#                             if math.isfinite(vb):
#                                 band_all.append(f"ch{gi}:{vb:.3f}dB")
#                             else:
#                                 band_all.append(f"ch{gi}:nan")
#                         print(
#                             f"[SNR-iter-ch-list] trial_delta_all=[{', '.join(trial_all)}]"
#                         )
#                         print(
#                             f"[SNR-iter-ch-list] band_delta_all=[{', '.join(band_all)}]"
#                         )
#             dec_out = self.classification(x_enc_new, x_mark_enc)
#             # dec_out = self.classification(x_enc, x_mark_enc)
#             return dec_out  # [B, N]
#         return None





import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cif_utils import apply_cif, init_cif


class BidirectionalCausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_forward=1, dilation_backward=1, groups=1):
        super().__init__()
        # Compute the padding size required for causality
        self.padding_forward = (kernel_size - 1) * dilation_forward
        self.padding_backward = (kernel_size - 1) * dilation_backward
        
        self.conv_forward = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=0, dilation=dilation_forward, groups=groups
        )
        self.conv_backward = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=0, dilation=dilation_backward, groups=groups
        )

    def forward(self, x):
        # Only left-side padding is required for causality
        x_forward = F.pad(x, (self.padding_forward, 0))  # Padding for forward direction
        x_backward = F.pad(x.flip(-1), (self.padding_backward, 0))  # Flip input for backward direction

        # Perform convolution in both directions
        out_forward = self.conv_forward(x_forward)
        out_backward = self.conv_backward(x_backward)

        # Combine forward and backward outputs (concatenation or summation)
        out = out_forward + out_backward.flip(-1)  # Flip the backward output to match the original sequence length
        return out


class BidirectionalDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_forward, dilation_backward, final=False):
        super().__init__()
        self.conv1 = BidirectionalCausalConv(
            in_channels, out_channels, kernel_size, dilation_forward=dilation_forward, dilation_backward=dilation_backward
        )
        self.conv2 = BidirectionalCausalConv(
            out_channels, out_channels, kernel_size, dilation_forward=dilation_forward, dilation_backward=dilation_backward
        )
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class BidirectionalDilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            *[
                BidirectionalDilatedConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    # dilation_forward=2**(i),  # Forward dilation
                    dilation_forward=2**(len(channels)-i-1), 
                    dilation_backward=2**(len(channels)-i-1),  # Backward dilation (can be adjusted as needed)
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, configs, hidden_dims=128, output_dims=320, kernel_size=3):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        init_cif(self, configs)
        self.use_cif = True
        self.encoder = BidirectionalDilatedConvEncoder(
            configs.enc_in,
            [hidden_dims] * configs.e_layers + [output_dims],  # a list here
            kernel_size=kernel_size,
        )

        # Decoder
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            raise NotImplementedError
        if self.task_name == "imputation":
            raise NotImplementedError
        if self.task_name == "anomaly_detection":
            raise NotImplementedError
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(output_dims, configs.num_class)

    def classification(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        output = self.dropout(self.encoder(x_enc.transpose(1, 2)))  # (batch_size, output_dims, timestamps)
        output = output.transpose(1, 2)  # (batch_size, timestamps, output_dims)
        output = F.max_pool1d(output.transpose(1, 2), kernel_size=output.size(1)).transpose(1, 2)
        output = output.squeeze(1)  # (batch_size, output_dims)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            raise NotImplementedError
        if self.task_name == "imputation":
            raise NotImplementedError
        if self.task_name == "anomaly_detection":
            raise NotImplementedError
        if self.task_name == "classification":
            dec_out = self.classification(apply_cif(self, x_enc), x_mark_enc)
            return dec_out  # [B, N]
        return None