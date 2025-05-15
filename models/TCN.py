
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.t = configs.t
        self.n = configs.n


        if configs.learnab:
            self.a = nn.Parameter(torch.tensor(configs.a))
            self.b = nn.Parameter(torch.tensor(configs.b))
        else:
            self.a = configs.a
            self.b = configs.b
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
        
        t = self.t
        n = self.n
        front_8_half = x_enc[:, :, :n]
        back_8_half = x_enc[:, :, -n:]
        x_enc_new = x_enc.clone()
        added_half = front_8_half * self.a + back_8_half * self.b

        if t >0:
            x_enc_new[:, :, :n] = added_half
        else:
            x_enc_new[:, :, -n:] = added_half
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
            dec_out = self.classification(x_enc_new, x_mark_enc)
            return dec_out  # [B, N]
        return None