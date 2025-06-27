import torch
import torch.nn as nn
from torch.distributions import Normal

# Flow Layer
class FlowLayer(nn.Module):
    def __init__(self, dim):
        super(FlowLayer, self).__init__()
        self.scale = nn.Parameter(torch.randn(1, dim))
        self.shift = nn.Parameter(torch.randn(1, dim))

    def forward(self, x):
        z = x * torch.exp(self.scale) + self.shift
        log_det_jacobian = torch.sum(self.scale, dim=1)
        return z, log_det_jacobian

# Normalizing Flow Model
class NormalizingFlow(nn.Module):
    def __init__(self, input_size, flow_layers=4):
        super(NormalizingFlow, self).__init__()
        self.input_size = input_size
        self.flow_layers = nn.ModuleList([FlowLayer(input_size) for _ in range(flow_layers)])
        self.register_buffer("prior_loc", torch.zeros(input_size))
        self.register_buffer("prior_scale", torch.ones(input_size))

    def forward(self, x):
        # x is expected to be of shape [Batch, input_size]
        log_det_jacobian_sum = 0
        for flow in self.flow_layers:
            x, log_det_jacobian = flow(x)
            log_det_jacobian_sum += log_det_jacobian
        return x, log_det_jacobian_sum

    def inverse(self, z):
        # The inverse transformation
        for flow in reversed(self.flow_layers):
            z = (z - flow.shift) * torch.exp(-flow.scale)
        return z

    def log_prob(self, x):
        z, log_det_jacobian_sum = self.forward(x)
        prior = Normal(self.prior_loc, self.prior_scale)
        log_prob_prior = prior.log_prob(z).sum(dim=1)
        return log_prob_prior + log_det_jacobian_sum 