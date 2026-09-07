import torch.nn as nn

from layers.ecg_backbones import resnet1d18, resnet1d_wang
from models.cif_utils import apply_cif, init_cif


class Model(nn.Module):
    def __init__(self, configs, variant="wang"):
        super().__init__()
        self.task_name = configs.task_name
        init_cif(self, configs)
        builder = resnet1d_wang if variant == "wang" else resnet1d18
        self.backbone = builder(
            num_classes=configs.num_class,
            input_channels=configs.enc_in,
            ps_head=configs.dropout,
        )

    def classification(self, x_enc, x_mark_enc):
        return self.backbone(x_enc.transpose(1, 2))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            return self.classification(apply_cif(self, x_enc), x_mark_enc)
        raise ValueError("Task name not recognized for ResNet1d")
