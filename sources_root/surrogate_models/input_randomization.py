import random
import typing
from abc import ABC, abstractmethod
import torchvision.transforms as T
import numpy as np
import torch
from torchvision.models import ResNet50_Weights


# from ..utils import *
# from ..gradient.mifgsm import MIFGSM


class FakeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad, loss):
        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def backward(ctx, *grad_output):
        grad, = ctx.saved_tensors
        grad_input = grad * grad_output[0] # grad_output is Delta loss, normally is one.
        return grad_input, None, None


class InputRandomizationKernel(torch.nn.Module, ABC):
    @abstractmethod
    def get_feedback(self, x, model, loss_fn, return_loss=False, adaptation=False) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
        pass

    def forward(self, x, model, loss_fn, adaptation=False):
        grad, loss = self.get_feedback(x, model, loss_fn, return_loss=True, adaptation=adaptation)
        return FakeFunc.apply(x, grad, loss.clone().detach())


    # Helping function for transformation classes
    @staticmethod
    def get_grad(loss, x, **kwargs):
        return torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]


class Identity(InputRandomizationKernel):
    # Fake Implementation
    def get_feedback(self, x, model, loss_fn, return_loss=False, adaptation=False) -> typing.Union[torch.Tensor,typing.Tuple[torch.Tensor, torch.Tensor]]:
        return None

    def forward(self, x, model, loss_fn, adaptation=False):
        return loss_fn(model(x))



def get_transformation(type, **kwargs) -> InputRandomizationKernel:
    if type == "decowa":
        return DeCowA(**kwargs)
    elif type == "bsr":
        return BSR(**kwargs)
    elif type == "none":
        return Identity()

# The code below is a modified version of the original implementation in repository https://github.com/Trustworthy-AI-Group/TransferAttack
# The TransferAttack repository is under the following MIT License

# MIT License
#
# Copyright (c) 2023 Trustworthy-AI-Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class DeCowA(InputRandomizationKernel):
    """
    DeCowA(Wapring Attack)
    'Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping (AAAI 2024)'(https://arxiv.org/abs/2402.03951)

    Arguments:
        mesh_width: the number of the control points
        mesh_height: the number of the control points = 3 * 3 = 9
        noise_scale: random noise strength
        num_warping: the number of warping transformation samples
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/decowa/resnet18 --attack decowa --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/decowa/resnet18 --eval
    """

    def __init__(self, mesh_width=3, mesh_height=3,
                 rho=0.01,
                 num_warping=20, noise_scale=2, device=None, **kwargs):
        super().__init__()
        self.num_warping = num_warping
        self.noise_scale = noise_scale
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height
        self.rho = rho
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def vwt(self, x, noise_map):
        n, c, w, h = x.size()
        X = grid_points_2d(self.mesh_width, self.mesh_height, self.device)
        Y = noisy_grid(self.mesh_width, self.mesh_height, noise_map, self.device)
        tpsb = TPS(size=(h, w), device=self.device)
        warped_grid_b = tpsb(X[None, ...].float(), Y[None, ...].float())
        warped_grid_b = warped_grid_b.repeat(x.shape[0], 1, 1, 1)
        vwt_x = torch.grid_sampler_2d(x, warped_grid_b.float(), 0, 0, False)
        return vwt_x

    def update_noise_map(self, x, model, loss):
        x.requires_grad = False
        noise_map = (torch.rand([self.mesh_height - 2, self.mesh_width - 2, 2]) - 0.5) * self.noise_scale
        for _ in range(1):
            noise_map.requires_grad = True
            vwt_x = self.vwt(x, noise_map)
            loss = loss(model(vwt_x))
            grad = self.get_grad(loss, noise_map)
            noise_map = noise_map.detach() - self.rho * grad
        return noise_map.detach()

    def get_feedback(self, x, model, loss_fn, return_loss=False, adaptation = False) -> typing.Union[torch.Tensor,typing.Tuple[torch.Tensor, torch.Tensor]]:
        x = x.clone().detach().to(self.device)
        grads = 0
        losses = 0
        # loss += loss.detach()
        for _ in range(self.num_warping):
            # Obtain the data after warping
            adv = x.clone().detach()
            noise_map_hat = self.update_noise_map(adv, model, loss_fn)
            x.requires_grad = True

            vwt_x = self.vwt(x, noise_map_hat)

            # Calculate the loss
            loss = loss_fn(model(vwt_x))

            # Calculate the gradients on x_idct
            grad = self.get_grad(loss, x)
            grads += grad
            losses += loss

        grads /= self.num_warping
        losses /= self.num_warping
        if return_loss:
            return grads, losses
        return grads

class AttackTest(DeCowA):
    def __init__(self, epsilon = 16/255, alpha = 1.6/255, epoch=10, decay=1., **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.random_start = False
        self.norm = 'linfty'


    def mifgsm(self, data, model, loss_fn, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        delta.requires_grad_()

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            xx = data + delta

            # Obtain the output
            logits = model(xx)

            # Calculate the loss
            loss = loss_fn(logits)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return  (delta.detach() + data).clamp(min = 0., max = 1.)

    def decowa(self, data, model, loss_fn, **kwargs):
        data = data.clone().detach().to(self.device)


        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        delta.requires_grad_()

        momentum = 0
        for _ in range(self.epoch):
            grads = 0
            for _ in range(self.num_warping):
                # Obtain the data after warping
                adv = (data + delta).clone().detach()
                # loss_fn = self.get_loss_fn(adv, label)
                noise_map_hat = self.update_noise_map(adv, model, loss_fn)
                vwt_x = self.vwt(data + delta, noise_map_hat)

                # Obtain the output
                logits = model(vwt_x)

                # Calculate the loss
                loss = loss_fn(logits)

                # Calculate the gradients on x_idct
                grad = self.get_grad(loss, delta)
                grads += grad

            grads /= self.num_warping

            # Calculate the momentum
            momentum = self.get_momentum(grads, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return (delta.detach() + data).clamp(min = 0., max = 1.)

    def get_momentum(self, grad, momentum, **kwargs):
        """
        The momentum calculation
        """
        return momentum * self.decay + grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True))

    def init_delta(self, data, **kwargs):
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0, 1).to(self.device)
                delta *= r / n * self.epsilon
            delta = torch.clamp(delta, 0.0 - data, 1.0 - data)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0,
                                                                                 maxnorm=self.epsilon).view_as(delta)
        delta = torch.clamp(delta, 0.0 - data, 1.0 - data)
        return delta.detach().requires_grad_(True)


def K_matrix(X, Y):
    eps = 1e-9

    D2 = torch.pow(X[:, :, None, :] - Y[:, None, :, :], 2).sum(-1)
    K = D2 * torch.log(D2 + eps)
    return K


def P_matrix(X):
    n, k = X.shape[:2]
    device = X.device

    P = torch.ones(n, k, 3, device=device)
    P[:, :, 1:] = X
    return P


class TPS_coeffs(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        n, k = X.shape[:2]  # n = 77, k =2
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)  # [1, 80, 80]
        K = K_matrix(X, X)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # Q = torch.solve(Z, L)[0]
        Q = torch.linalg.solve(L, Z)
        return Q[:, :k], Q[:, k:]


class TPS(torch.nn.Module):
    def __init__(self, size: tuple = (256, 256), device=None):
        super().__init__()
        h, w = size
        self.size = size
        self.device = device
        self.tps = TPS_coeffs()
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        self.grid = grid.view(-1, h * w, 2)

    def forward(self, X, Y):
        """Override abstract function."""
        h, w = self.size
        W, A = self.tps(X, Y)
        U = K_matrix(self.grid, X)
        P = P_matrix(self.grid)
        grid = P @ A + U @ W
        return grid.view(-1, h, w, 2)


def grid_points_2d(width, height, device):
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, height, device=device),
         torch.linspace(-1.0, 1.0, width, device=device)])
    return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)


def noisy_grid(width, height, noise_map, device):
    """
    Make uniform grid points, and add noise except for edge points.
    """
    grid = grid_points_2d(width, height, device)
    mod = torch.zeros([height, width, 2], device=device)
    mod[1:height - 1, 1:width - 1, :] = noise_map
    return grid + mod.reshape(-1, 2)


class BSR(InputRandomizationKernel):
    """
    BSR Attack
    'Boosting Adversarial Transferability by Block Shuffle and Rotation'(https://https://arxiv.org/abs/2308.10299)

    Arguments:
        num_scale (int): the number of shuffled copies in each iteration.
        num_block (int): the number of block in the image.

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10, num_block=3

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/bsr/resnet18 --attack bsr --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/bsr/resnet18 --eval
    """

    def __init__(self, num_scale=20, num_block=3, **kwargs):
        super().__init__()
        self.num_scale = num_scale
        self.num_block = num_block

    def get_length(self, length):
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        rotation_transform = T.RandomRotation(degrees=(-24, 24), interpolation=T.InterpolationMode.BILINEAR)
        return rotation_transform(x)

    def shuffle(self, x):
        dims = [2, 3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat(
            [torch.cat(self.shuffle_single_dim(self.image_rotation(x_strip), dim=dims[1]), dim=dims[1]) for x_strip in
             x_strips], dim=dims[0])

    def transform(self, x, **kwargs):
        """
        Scale the input for BSR
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)])


    def get_feedback(self, x, model, loss_fn, return_loss=False, adaptation=False) -> typing.Union[torch.Tensor,typing.Tuple[torch.Tensor, torch.Tensor]]:
        x = x.clone().detach()
        x.requires_grad = True
        loss = loss_fn(model(self.transform(x)))
        grad = self.get_grad(loss, x)
        return grad, loss if return_loss else grad


if __name__ == '__main__':
    import torchvision
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).cuda()
    transform = get_transformation("bsr")

    img = torch.rand(1, 3, 224, 224).cuda()
    img.requires_grad = True

    import torch.nn.functional as F
    loss_fn = lambda logits: -F.log_softmax(logits, dim=1)[..., 0].sum()


    loss = transform(img, model, loss_fn)

    grad = transform.get_grad(loss, img)

    print(grad)
