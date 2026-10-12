from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision.models as models


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class Scale(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self._scalar = scalar

    def forward(self, x):
        return torch.mul(x, self._scalar)


class RandRoll(nn.Module):
    def __init__(self, seed=42, active=True):
        super().__init__()
        self._seed = seed
        self._gen = torch.Generator()
        self._gen.manual_seed(self._seed)
        self._active = active

    def train(self, mode):
        self._active = mode

    def forward(self, x):
        if self._active:
            a = torch.randint(high=x.shape[-1], size=(1,), generator=self._gen).item()
            return torch.roll(x, (a, a), dims=(2, 3))
        return x


class Normalize(nn.Module):
    def __init__(self, norm="fro"):
        super().__init__()
        match norm:
            case "max":
                def f(x):
                    s = x.shape
                    return torch.linalg.norm(
                        x.view(s[0], s[1], s[2] * s[3]),
                        ord=float("inf"),
                        dim=-1,
                    )
            case _:
                f = lambda x: torch.linalg.norm(x, ord=norm, dim=(-2, -1))
        self._norm = f

    def forward(self, x):
        s = x.shape
        n = self._norm(x)
        n = n[:, -1]
        n = n.view(s[0], 1, 1, 1)
        n = n.repeat(1, s[1], s[2], s[3])
        return x / n.clamp_min(1e-12)


def circularPadConv2d(in_chan, out_chan, kern_sz=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
        nn.CircularPad2d(padding),
        nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=kern_sz,
            stride=stride,
            bias=bias,
        ),
    )


class PaddedBasicBlock(models.resnet.BasicBlock):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super().__init__(in_chan, out_chan, *args, **kwargs)
        self.conv1 = circularPadConv2d(
            in_chan,
            out_chan,
            kern_sz=3,
            stride=self.stride,
            padding=1,
        )
        self.conv2 = circularPadConv2d(
            out_chan,
            out_chan,
            kern_sz=3,
            stride=1,
            padding=1,
        )


def my_resnet18(*, padding=True, weights=None, progress=True, **kwargs):
    weights = models.resnet.ResNet18_Weights.verify(weights)
    block = PaddedBasicBlock if padding else models.resnet.BasicBlock
    model = models.resnet._resnet(block, [2, 2, 2, 2], weights, progress, **kwargs)
    if padding:
        model.conv1 = circularPadConv2d(1, 64, 7, 2, 3)
    else:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.maxpool = nn.Sequential(
        nn.CircularPad2d(1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
    )
    model.fc = nn.Identity()
    return model


class MyNet(nn.Module):
    def __init__(
        self,
        latent_dim,
        matrix_size,
        padding,
        decoder_channels=512,
        normalize_input=None,
        rescale_input=None,
        augment_input=False,
    ):
        super().__init__()

        self.matrix_size = matrix_size
        self.padding = padding
        self.latent_dim = latent_dim

        preproc_steps = []
        match normalize_input:
            case "fro":
                preproc_steps.append(Normalize("fro"))
                preproc_steps.append(Scale(matrix_size))
            case "max":
                preproc_steps.append(Normalize("max"))
            case _:
                pass
        if rescale_input:
            preproc_steps.append(Scale(rescale_input))
        if augment_input:
            preproc_steps.append(RandRoll())
        self.preproc = nn.Sequential(*preproc_steps) if preproc_steps else nn.Identity()

        self.encoder = my_resnet18(padding=padding)

        self.z_mean = nn.Linear(512, latent_dim)
        self.z_log_var = nn.Linear(512, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, decoder_channels),
            nn.ReLU(),
        )

        n_steps = math.ceil(math.log2(matrix_size))
        decoder_steps = []
        n_chans = decoder_channels
        for i in range(n_steps, 0, -1):
            if i != 1:
                next_chans = math.ceil(n_chans / 2)
                decoder_steps.extend(
                    [
                        nn.ConvTranspose2d(
                            n_chans,
                            next_chans,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                        nn.ReLU(),
                    ]
                )
            else:
                decoder_steps.append(
                    nn.ConvTranspose2d(
                        n_chans,
                        1,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
            n_chans = next_chans

        self.decoder_conv = nn.Sequential(*decoder_steps)
        self.decoder = nn.Sequential(
            self.decoder_fc,
            View((-1, decoder_channels, 1, 1)),
            self.decoder_conv,
        )

    def reparameterize(self, mean, log_var):
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)

        if not self.training:
            return mean

        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def forward(self, x):
        og_scale = torch.linalg.norm(x, ord="fro", dim=(-2, -1)) * self.matrix_size
        x_preproc_og = self.preproc(x)
        x_preproc_flipped = torch.flip(x_preproc_og, dims=(2, 3))
        x_encoded_og = self.encoder(x_preproc_og)
        x_encoded_flipped = self.encoder(x_preproc_flipped)
        x_encoded = torch.add(x_encoded_og, x_encoded_flipped)
        z_mean = self.z_mean(x_encoded)
        z_log_var = self.z_log_var(x_encoded)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z)
        return x_preproc_og, x_recon, z, z_mean, z_log_var, og_scale


class CircularResNet18Backbone(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        if in_channels != 1:
            raise ValueError("Official ShapeEmbed ResNet expects one input channel.")
        if base_channels != 64:
            raise ValueError("Official ShapeEmbed ResNet uses 64 base channels.")
        self.model = my_resnet18(padding=True)
        self.out_channels = 512

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
