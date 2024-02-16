import torch
import torch.nn as nn


class ConvRNN(nn.Module):
    """ConvRNN module. ConvLSTM cell followed by a feed forward convolution layer.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): separable conv receptive field
        stride (int): separable conv stride.
        padding (int): padding.
        separable (boolean): if True, uses depthwise separable convolution for the forward convolutional layer.
        separable_hidden (boolean): if True, uses depthwise separable convolution for the hidden convolutional layer.
        cell (string): RNN cell type, currently gru and lstm only are supported.
        **kwargs: additional parameters for the feed forward convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 cell='lstm'):
        assert cell.lower() in ('lstm', 'gru'), f"Only 'gru' or 'lstm' cells are supported, got {cell}"
        super(ConvRNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_x2h = nn.Sequential(
            nn.Conv2d(in_channels, 4 * out_channels if cell == "lstm" else in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(4 * out_channels if cell == "lstm" else in_channels),
            nn.ReLU(),
        )

        if cell.lower() == "lstm":
            self.timepool = ConvLSTMCell(out_channels, 3, conv_func=nn.Conv2d)
        else:
            self.timepool = ConvGRUCell(in_channels, out_channels, kernel_size=3, conv_func=nn.Conv2d)

    def forward(self, x):
        y = self.conv_x2h(x)
        h = self.timepool(y)
        return h

    def reset(self, mask=torch.zeros((1,), dtype=torch.float32)):
        """Resets memory of the network."""
        self.timepool.reset(mask)

    @torch.jit.export
    def reset_all(self):
        self.timepool.reset_all()

    def detach(self):
        self.timepool.detach()


class ConvLSTMCell(nn.Module):
    """ConvLSTMCell module, applies sequential part of LSTM.

    LSTM with matrix multiplication replaced by convolution
    See Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
    (Shi et al.)

    Args:
        hidden_dim (int): number of output_channels of hidden state.
        kernel_size (int): internal convolution receptive field.
        conv_func (fun): functional that you can replace if you want to interact with your 2D state differently.
        hard (bool): applies hard gates.
    """

    def __init__(self, hidden_dim, kernel_size, conv_func=nn.Conv2d):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv_h2h = conv_func(in_channels=self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=True)
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh

        self.prev_h = torch.zeros((1, self.hidden_dim, 0, 0), dtype=torch.float32)
        self.prev_c = torch.zeros((1, self.hidden_dim, 0, 0), dtype=torch.float32)

    @torch.jit.export
    def get_dims_NCHW(self):
        return self.prev_h.size()

    def forward(self, x):
        assert x.dim() == 4

        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_h.size()
        input_N, input_C, input_H, input_W = x.size()
        assert input_C == 4 * hidden_C
        assert hidden_C == self.hidden_dim

        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = x.device
            self.prev_h = torch.zeros((input_N, self.hidden_dim, input_H, input_W), dtype=torch.float32).to(device)
            self.prev_c = torch.zeros((input_N, self.hidden_dim, input_H, input_W), dtype=torch.float32).to(device)

        # self.prev_h.detach_()
        # self.prev_c.detach_()

        tmp = self.conv_h2h(self.prev_h) + x

        cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
        i = self.sigmoid(cc_i)
        f = self.sigmoid(cc_f)
        o = self.sigmoid(cc_o)
        g = self.tanh(cc_g)

        c = f * self.prev_c + i * g
        h = o * self.tanh(c)
        result = h

        self.prev_h = h
        self.prev_c = c

        return result


    @torch.jit.export
    def reset(self, mask):
        """Sets the memory (or hidden state to zero), normally at the beginning of a new sequence.

        `reset()` needs to be called at the beginning of a new sequence. The mask is here to indicate which elements
        of the batch are indeed new sequences. """
        if self.prev_h.numel() == 0:
            return
        batch_size, _, _, _ = self.prev_h.size()
        if batch_size == len(mask):
            assert batch_size == mask.numel()
            mask = mask.reshape(-1, 1, 1, 1)
            assert mask.shape == torch.Size([len(self.prev_h), 1, 1, 1])
            self.prev_h.detach_()
            self.prev_c.detach_()
            self.prev_h = self.prev_h*mask.to(device=self.prev_h.device)
            self.prev_c = self.prev_c*mask.to(device=self.prev_c.device)

    @torch.jit.export
    def reset_all(self):
        """Resets memory for all sequences in one batch."""
        self.reset(torch.zeros((len(self.prev_h), 1, 1, 1), dtype=torch.float32, device=self.prev_h.device))

    def detach(self):
        self.prev_h.detach_()
        self.prev_c.detach_()


class ConvGRUCell(nn.Module):
    """
    ConvGRUCell module, applies sequential part of the Gated Recurrent Unit.

    GRU with matrix multiplication replaced by convolution
    See Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural networks on sequence modeling.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output_channels of hidden state.
        kernel_size (int): internal convolution receptive field.
        padding (int): padding parameter for the convolution
        conv_func (fun): functional that you can replace if you want to interact with your 2D state differently.
        hard (bool): applies hard gates.

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, conv_func=nn.Conv2d,
                 stride=1, dilation=1):
        super(ConvGRUCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.in_channels + self.out_channels, out_channels=2 * self.out_channels,
                                 kernel_size=kernel_size, padding=1)
        self.conv_f = conv_func(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.prev_h = torch.zeros((1, self.out_channels, 0, 0), dtype=torch.float32)

        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh

        # self.init_sig = False


    def forward(self, xt):
        """
        xt size: (B,C,H,W)
        return size: (B, C',H,W)
        """
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_h.size()
        input_N, input_C, input_H, input_W = xt.size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = xt.device
            self.prev_h = torch.zeros((input_N, self.out_channels, input_H, input_W), dtype=torch.float32,
                                      device=device)

        # self.prev_h.detach_()

        z, r = self.conv_rz(torch.cat((self.prev_h, xt), dim=1)).split(self.out_channels, 1)
        update_gate = self.sigmoid(z)
        reset_gate = self.sigmoid(r)

        f = self.conv_f(torch.cat((self.prev_h * reset_gate, xt), dim=1))
        input_gate = self.tanh(f)

        self.prev_h = (1 - update_gate) * self.prev_h + update_gate * input_gate

        result = self.prev_h
        return result


    @torch.jit.export
    def reset(self, mask):
        """Sets the memory (or hidden state to zero), normally at the beginning of a new sequence.

        `reset()` needs to be called at the beginning of a new sequence. The mask is here to indicate which elements
        of the batch are indeed new sequences. """
        batch_size, _, _, _ = self.prev_h.size()
        if batch_size == len(mask) and self.prev_h.device == mask.device:   # if already init
            assert mask.shape == torch.Size([len(self.prev_h), 1, 1, 1])
            self.prev_h.detach_()
            self.prev_h = self.prev_h*mask.to(device=self.prev_h.device)

        # if not self.init_sig:
        #     self.prev_h = self.prev_h * mask.to(device=self.prev_h.device)
        #     self.init_sig = True
        # else:
        #     assert batch_size == len(mask) and self.prev_h.device == mask.device, 'batch_size: {}, len(mask): {}'.format(batch_size, len(mask))
        #     assert mask.shape == torch.Size([len(self.prev_h), 1, 1, 1])
        #     self.prev_h.detach_()
        #     self.prev_h = self.prev_h*mask.to(device=self.prev_h.device)


    def detach(self):
        self.prev_h.detach_()



