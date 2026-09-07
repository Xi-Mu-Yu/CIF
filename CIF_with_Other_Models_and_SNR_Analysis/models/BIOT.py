import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import BIOTEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from models.cif_utils import apply_cif, init_cif


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        init_cif(self, configs)
        patch_len = configs.patch_len
        stride = configs.patch_len

        self.enc_embedding = BIOTEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            patch_len,
            stride,
            ["mask", "channel"],
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
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
            patch_num = int((configs.seq_len - configs.patch_len) / stride + 2)
            self.projection = nn.Linear(
                configs.d_model * patch_num * configs.enc_in,
                configs.num_class,
            )

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            return self.classification(apply_cif(self, x_enc), x_mark_enc)
        raise ValueError("Task name not recognized for BIOT")
