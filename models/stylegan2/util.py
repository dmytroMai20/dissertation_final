import torch
from torch import nn
import torch.nn.functional as F
import math

class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        # Blurring kernel
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        # Convert the kernel to a PyTorch tensor
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        # Normalize the kernel
        kernel /= kernel.sum()
        # Save kernel as a fixed parameter (no gradient updates)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        # Padding layer
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        # Get shape of the input feature map
        b, c, h, w = x.shape
        # Reshape for smoothening
        x = x.view(-1, 1, h, w)

        # Add padding
        x = self.pad(x)

        # Smoothen (blur) with the kernel
        x = F.conv2d(x, self.kernel)

        # Reshape and return
        return x.view(b, c, h, w)

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = (1 / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.ones(out_features)*bias)
    
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, bias=self.bias)
    
class GradientPenalty(nn.Module):
    def forward(self, x: torch.Tensor, d: torch.Tensor):
        # Get batch size
        batch_size = x.shape[0]


        gradients, *_ = torch.autograd.grad(outputs=d,
                                            inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)


        gradients = gradients.reshape(batch_size, -1)

        norm = gradients.norm(2, dim=-1)

        return torch.mean(norm ** 2)
    
class PathLengthPenalty(nn.Module):
    def __init__(self, beta: float):
        super().__init__()


        self.beta = beta

        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        # Get the device
        device = x.device
        # Get number of pixels
        image_size = x.shape[2] * x.shape[3]

        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / math.sqrt(image_size)


        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)


        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        # Regularize after first step
        if self.steps > 0:

            a = self.exp_sum_a / (1 - self.beta ** self.steps)

            loss = torch.mean((norm - a) ** 2)
        else:
            # Return a dummy loss if we can't calculate $a$
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        # Update exponential sum
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)

        self.steps.add_(1.)

        # Return the penalty
        return loss