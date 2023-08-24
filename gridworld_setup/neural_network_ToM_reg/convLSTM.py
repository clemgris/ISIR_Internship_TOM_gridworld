import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, input, hidden_state):
        h, c = hidden_state
        combined = torch.cat((input, h), dim=1)
        gates = self.conv(combined)
        ingate, remembergate, cellgate, outgate = torch.split(
            gates, self.hidden_channels, dim=1
        )

        ingate = torch.sigmoid(ingate)
        remembergate = torch.sigmoid(remembergate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c_new = remembergate * c + ingate * cellgate
        h_new = outgate * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels[i - 1]
            cell_list.append(
                ConvLSTMCell(cur_input_channels, hidden_channels[i], kernel_size)
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_states=None):
        if hidden_states is None:
            hidden_states = [
                (
                    torch.zeros(input.size(0), ch, input.size(3), input.size(4)).to(
                        input.device
                    ),
                    torch.zeros(input.size(0), ch, input.size(3), input.size(4)).to(
                        input.device
                    ),
                )
                for ch in self.hidden_channels
            ]

        seq_length = input.shape[1]
        cur_layer_input = input  # [batch, num_steps, num_channels, H, W]
        for layer_idx in range(self.num_layers):
            h, c = hidden_states[layer_idx]
            output_inner = []
            for t in range(seq_length):
                h, c = self.cell_list[layer_idx](
                    input=cur_layer_input[:, t, :, :, :], hidden_state=(h, c)
                )
                output_inner.append(h)
            cur_layer_input = torch.stack(output_inner, dim=1)
            hidden_states[layer_idx] = (h, c)

        return cur_layer_input, hidden_states
