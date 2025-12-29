import torch


SIGMOID_MAX = 9.21024
LOGIT_MAX = 0.9999

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -9.21, 9.21)
    return torch.sigmoid(tensor)

def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX) # todo LOGIT_MAX: 0.9999 避免有0或1，会产生inf/nan clamp到[ε, 1-ε]
    return torch.log(tensor / (1 - tensor))
