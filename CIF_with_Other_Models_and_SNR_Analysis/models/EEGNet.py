import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Conv_Blocks import TemporalSpatialConv
from models.cif_utils import apply_cif, init_cif
import numpy as np


class Model(nn.Module):

    def __init__(self, configs, f1=32, d=2, kernel_size=128):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.t = configs.t
        self.n = configs.n
        init_cif(self, configs)

        # TemporalSpatialConv expects input as [B, C, T]
        # and uses f1/d/kernel_size/dropout_rate from EEGNet-style hyperparameters.
        self.encoder = TemporalSpatialConv(
            f1=f1,
            d=d,
            channels=configs.enc_in,
            kernel_size=kernel_size,
            dropout_rate=configs.dropout,
        )

        # Decoder
        if self.task_name == "classification":
            # TemporalSpatialConv ends with `.squeeze()`, which can remove batch dim
            # when batch size is 1. We will handle that in `classification()` too.
            # Here we avoid LazyLinear to be compatible with SWA deepcopy.
            with torch.no_grad():
                dummy = torch.zeros(2, configs.enc_in, configs.seq_len)
                enc_out = self.encoder(dummy)  # shape: [B, ...] or squeezed
                if enc_out.dim() == 1:
                    enc_out = enc_out.unsqueeze(0)
                elif enc_out.dim() == 2:
                    enc_out = enc_out.unsqueeze(0)
                feat_dim = enc_out.reshape(enc_out.shape[0], -1).shape[1]
            self.projection = nn.Linear(feat_dim, configs.num_class)

    def classification(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        # conv encoder
        # Input from dataloader is [B, T, C] -> transpose to [B, C, T] for EEGNet.
        output = self.encoder(x_enc.transpose(1, 2))

        # TemporalSpatialConv uses `.squeeze()`, so batch dimension may disappear when B==1.
        if output.dim() == 1:
            output = output.unsqueeze(0)
        elif output.dim() == 2:
            output = output.unsqueeze(0)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            dec_out = self.classification(apply_cif(self, x_enc), x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError("Task name not recognized or not implemented within the EEGNet Model")