import torch.nn as nn

from layers.ecg_backbones import xresnet1d18
from models.cif_utils import apply_cif, init_cif


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        init_cif(self, configs)
        self.backbone = xresnet1d18(
            num_classes=configs.num_class,
            input_channels=configs.enc_in,
            ps_head=configs.dropout,
        )

    def classification(self, x_enc, x_mark_enc):
        return self.backbone(x_enc.transpose(1, 2))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            return self.classification(apply_cif(self, x_enc), x_mark_enc)
        raise ValueError("Task name not recognized for XResNet1d")
