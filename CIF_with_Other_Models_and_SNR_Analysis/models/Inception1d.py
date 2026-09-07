import torch.nn as nn

from layers.ecg_backbones import Inception1dBackbone
from models.cif_utils import apply_cif, init_cif


class Model(nn.Module):
    def __init__(self, configs, depth=6, nb_filters=32, bottleneck_size=32, use_residual=True):
        super().__init__()
        self.task_name = configs.task_name
        init_cif(self, configs)
        kernel_size = min(max(getattr(configs, "ecg_kernel_size", 40), 15), configs.seq_len)
        self.backbone = Inception1dBackbone(
            num_classes=configs.num_class,
            input_channels=configs.enc_in,
            kernel_size=kernel_size,
            depth=depth,
            bottleneck_size=bottleneck_size,
            nb_filters=nb_filters,
            use_residual=use_residual,
            ps_head=configs.dropout,
        )

    def classification(self, x_enc, x_mark_enc):
        return self.backbone(x_enc.transpose(1, 2))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            return self.classification(apply_cif(self, x_enc), x_mark_enc)
        raise ValueError("Task name not recognized for Inception1d")
