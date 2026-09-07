import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.ADformer_EncDec import Encoder, EncoderLayer
from layers.Embed import TokenChannelEmbedding
from layers.SelfAttention_Family import ADformerLayer
from models.cif_utils import apply_cif, init_cif


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        init_cif(self, configs)

        if configs.no_temporal_block and configs.no_spatial_block:
            raise ValueError("At least one of the two blocks should be True")
        patch_len_list = [] if configs.no_temporal_block else list(map(int, configs.patch_len_list.split(",")))
        up_dim_list = [] if configs.no_spatial_block else list(map(int, configs.up_dim_list.split(",")))
        stride_list = patch_len_list
        patch_num_list = [
            int((configs.seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")

        self.enc_embedding = TokenChannelEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            patch_len_list,
            up_dim_list,
            stride_list,
            configs.dropout,
            augmentations,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ADformerLayer(
                        len(patch_len_list),
                        len(up_dim_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        if self.task_name == "classification":
            self.classifier = nn.Linear(
                configs.d_model * len(patch_num_list) + configs.d_model * len(up_dim_list),
                configs.num_class,
            )

    def classification(self, x_enc, x_mark_enc):
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        enc_out_t, enc_out_c, _, _ = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            enc_out = enc_out_t + enc_out_c
        enc_out = torch.cat([x[:, -1, :].unsqueeze(1) for x in enc_out], dim=1)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        return self.classifier(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            return self.classification(apply_cif(self, x_enc), x_mark_enc)
        raise ValueError("Task name not recognized for ADformer")
