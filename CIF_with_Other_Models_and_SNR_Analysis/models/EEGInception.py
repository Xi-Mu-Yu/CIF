import torch.nn as nn

from layers.Conv_Blocks import InceptionBlock, SpatialBlock
from models.cif_utils import apply_cif, init_cif


class Model(nn.Module):
    def __init__(
        self,
        configs,
        n_blocks=3,
        channels=(96, 192, 384),
        kernel_sizes=(8, 16, 32),
        depth_multiplier=2,
        bottleneck_channels=32,
    ):
        super().__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in
        init_cif(self, configs)

        blocks = []
        in_ch = self.enc_in
        for out_ch in channels:
            blocks.append(
                InceptionBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=nn.ReLU(inplace=True),
                    dropout=configs.dropout,
                )
            )
            blocks.append(
                SpatialBlock(
                    in_channels=out_ch,
                    depth_multiplier=depth_multiplier,
                    activation=nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.feature_extractor = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        if self.task_name == "classification":
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(configs.dropout),
                nn.Linear(in_ch, configs.num_class),
            )

    def classification(self, x_enc, x_mark_enc):
        x = x_enc.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        return self.classifier(x)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            return self.classification(apply_cif(self, x_enc), x_mark_enc)
        raise ValueError("Task name not recognized for EEGInception")
