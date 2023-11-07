from turtle import forward
import torch
import torch.nn as nn

from utils import quantize
from activation import activation

class QuantizedGRU(nn.Module):
    def __init__(self, hidden_size=12, device='cpu', weights=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.gru_cell = QuantizedGRUCell(hidden_size=hidden_size, weights=weights, device=self.device)
        self.fcnn = QuantizedFCNN(hidden_size=hidden_size, weights=weights)

    def forward(self, input):
        output = torch.zeros([input.shape[0], input.shape[1], 1]).to(self.device)
        hidden_states = torch.zeros([input.shape[1], self.hidden_size]).to(self.device)

        for i in range(input.shape[0]):
            hidden_states = self.gru_cell(input[i], hidden_states)
            output[i] = self.fcnn(hidden_states)

        return output


class QuantizedGRUCell(nn.Module):
    def __init__(self, hidden_size=12, weights=None, device='cpu') -> None:
        super().__init__()
        
        self.hidden_size = hidden_size

        self.act = activation(device=device)

        if weights is None:
            self.weight_ih_l0 = nn.Parameter(torch.randn([self.hidden_size * 3, 1]))
            self.weight_hh_l0 = nn.Parameter(torch.randn([self.hidden_size * 3, self.hidden_size]))
            self.bias_ih_l0 = nn.Parameter(torch.randn([self.hidden_size * 3, 1]))
            self.bias_hh_l0 = nn.Parameter(torch.randn([self.hidden_size * 3, 1]))
        else:
            self.weight_ih_l0 = nn.Parameter(QuantizeTensor.apply(weights['weight_ih_l0']))
            self.weight_hh_l0 = nn.Parameter(QuantizeTensor.apply(weights['weight_hh_l0']))
            self.bias_ih_l0 = nn.Parameter(QuantizeTensor.apply(weights['bias_ih_l0']))
            self.bias_hh_l0 = nn.Parameter(QuantizeTensor.apply(weights['bias_hh_l0']))

        # self.quantize = QuantizeTensor()

    def forward(self, input, hidden_state):
        ii = QuantizeTensor.apply(torch.matmul(input, self.weight_ih_l0.T))
        ii = QuantizeTensor.apply(ii + self.bias_ih_l0)

        hh = QuantizeTensor.apply(torch.matmul(hidden_state, self.weight_hh_l0.T))
        hh = QuantizeTensor.apply(hh + self.bias_hh_l0)

        rt = QuantizeTensor.apply(self.act.sigmoid_approx(ii[:, 0:self.hidden_size] + hh[:, 0:self.hidden_size]))
        zt = QuantizeTensor.apply(self.act.sigmoid_approx(ii[:, self.hidden_size:2*self.hidden_size] + hh[:, self.hidden_size:2*self.hidden_size]))

        nt = QuantizeTensor.apply(self.act.tanh_approx(ii[:, 2*self.hidden_size:3*self.hidden_size] + rt * hh[:, 2*self.hidden_size:3*self.hidden_size]))

        hidden_state = QuantizeTensor.apply((1 - zt) * nt + zt * hidden_state)

        return hidden_state.squeeze(-1)



class QuantizedFCNN(nn.Module):
    def __init__(self, hidden_size=12, weights=None) -> None:
        super().__init__()
        if weights is not None:
            self.linear1 = QuantizedLinear(
                input_size=hidden_size,
                output_size=hidden_size//2,
                weights=QuantizeTensor.apply(weights['linear1.weight']),
                bias=QuantizeTensor.apply(weights['linear1.bias'])
            )

            self.linear2 = QuantizedLinear(
                input_size=hidden_size//2,
                output_size=1,
                weights=QuantizeTensor.apply(weights['linear2.weight']),
                bias=QuantizeTensor.apply(weights['linear2.bias'])
            )

    def forward(self, input):
        output = QuantizeTensor.apply(self.linear1(input))
        output = QuantizeTensor.apply(self.linear2(torch.relu(output)))
        return output



class QuantizedLinear(nn.Module):
    def __init__(self, input_size, output_size, weights, bias, requires_grad=True) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.weights = nn.Parameter(weights, requires_grad=requires_grad)
        self.bias = nn.Parameter(bias, requires_grad=requires_grad)

        # self.quantize = QuantizeTensor()

    def forward(self, input):
        output = QuantizeTensor.apply(torch.matmul(input, self.weights.T))
        output = QuantizeTensor.apply(output + self.bias)

        return output


class QuantizeTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return quantize(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


if __name__ == '__main__':
    print(quantize(torch.tensor(-3.2389)))
