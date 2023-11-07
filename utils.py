import torch


def loss_func(pred, label, weights):
  
    # print(pred.shape, label.shape)
    label = label.repeat(1024, 1).T
    # print(label.shape)
    error = (pred - label) ** 2 / (label ** 2)
    error = error * weights
    error = torch.mean(error)
    return error

def quantize(input, W=16, I=4, signed=True):
    fractional_bit = W - I
    output = input * (2 ** fractional_bit)
    output = torch.round(output)

    if signed is True:
        output = torch.clip(output, min=-2**(W-1), max=2**(W-1))

    output = output / (2 ** fractional_bit)

    return output


def read_state_dict(state_dict):
    """ Read the state dict and save the weights of RNN in the dictionary.

    Args:
        state_dict: state dict of the trained RNN model

    Returns:
        rnn_state_dict: dictionary containing the weights of the RNN only
    """
    rnn_state_dict = {}
    fcnn_state_dict = {}
    for param_tensor in state_dict:
        if 'rnn' in param_tensor:
            rnn_state_dict[param_tensor.split('.')[-1]] = state_dict[param_tensor]
        elif 'linear' in param_tensor:
            fcnn_state_dict[param_tensor] = state_dict[param_tensor]
    return rnn_state_dict, fcnn_state_dict

if __name__ == '__main__':
    print( quantize(torch.tensor(-1024.5), W=12, I=11))