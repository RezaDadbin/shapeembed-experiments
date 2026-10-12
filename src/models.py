import math
from typing import Literal

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


# =============================================================================
# Frobenius normalization
# =============================================================================

class NormalizeFro(nn.Module):
    """
    Paper step:
        Normalize each distance matrix by its Frobenius norm.

    Input:
        x: [B, 1, N, N]

    Output:
        x / ||x||_F
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(x, ord="fro", dim=(-2, -1), keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        return x / norm


# =============================================================================
# Circular padding convolution
# =============================================================================

class CircularConv2d(nn.Sequential):
    """
    Conv2d preceded by circular padding.

    This is where the paper's circular padding is applied.
    It is applied in the encoder, not when constructing the distance matrix.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__(
            nn.CircularPad2d(padding),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=bias,
            ),
        )


class CircularBasicBlock(resnet.BasicBlock):
    """
    ResNet-18 BasicBlock with circular padding in the 3x3 convolutions.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ):
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
            norm_layer=norm_layer,
        )

        self.conv1 = CircularConv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.conv2 = CircularConv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )


def make_circular_resnet18_encoder() -> nn.Module:
    """
    Paper encoder:
        modified ResNet-18
        one input channel
        circular padding
        no classification head

    Output:
        512-dimensional feature vector.
    """

    encoder = resnet.ResNet(
        block=CircularBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
    )

    encoder.conv1 = CircularConv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    encoder.maxpool = nn.Sequential(
        nn.CircularPad2d(1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
    )

    encoder.fc = nn.Identity()

    return encoder


# =============================================================================
# Decoder used by the original paper model
# =============================================================================

class ShapeEmbedDecoder(nn.Module):
    """
    Decoder path used for the original ShapeEmbed implementation in this code.

    The paper says the ResNet-18 encoder is mirrored in the decoder path.
    This implementation mirrors the downsampling idea by starting from a 1x1
    feature map and repeatedly upsampling with transposed convolutions until
    the matrix reaches N x N.

    Input:
        z: [B, latent_dim]

    Output:
        reconstructed distance matrix: [B, 1, N, N]
    """

    def __init__(
        self,
        latent_dim: int,
        matrix_size: int,
        initial_channels: int = 512,
    ):
        super().__init__()

        if matrix_size <= 0 or matrix_size & (matrix_size - 1) != 0:
            raise ValueError(
                f"matrix_size must be a positive power of two, got {matrix_size}"
            )

        self.latent_dim = latent_dim
        self.matrix_size = matrix_size
        self.initial_channels = initial_channels

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, initial_channels),
            nn.ReLU(inplace=True),
        )

        num_upsampling_steps = math.ceil(math.log2(matrix_size))

        layers = []
        in_channels = initial_channels

        for i in range(num_upsampling_steps, 0, -1):
            if i != 1:
                out_channels = math.ceil(in_channels / 2)

                layers.extend([
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.ReLU(),
                ])

                in_channels = out_channels

            else:
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        1,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(z.size(0), self.initial_channels, 1, 1)
        return self.net(x)


# =============================================================================
# Phase 2.2: MLP decoder replacement
# =============================================================================

class MLPDistanceMatrixDecoder(nn.Module):
    """
    Phase 2.2 modification:
        Replace the entire convolutional decoder path with an MLP.

    Important:
        This is used only for the "mlp_decoder" experiment.
        Encoder, latent sampling, KL loss, and other ShapeEmbed components stay.
    """

    def __init__(
        self,
        latent_dim: int,
        matrix_size: int,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.matrix_size = matrix_size

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, matrix_size * matrix_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        return x.view(z.size(0), 1, self.matrix_size, self.matrix_size)


# =============================================================================
# Shared encoder behavior: original + flipped matrix
# =============================================================================

class ReflectionInvariantEncodingMixin:
    """
    Paper trick:
        Encode D.
        Encode flipped D.
        Sum the two feature vectors.

    This makes the encoder unable to distinguish traversal direction.
    """

    def encode_with_reflection_invariance(self, x_pre: torch.Tensor) -> torch.Tensor:
        x_flipped = torch.flip(x_pre, dims=(-2, -1))

        h = self.encoder(x_pre)
        h_flip = self.encoder(x_flipped)

        return h + h_flip


# =============================================================================
# Original ShapeEmbed
# =============================================================================

class ShapeEmbedOriginal(nn.Module, ReflectionInvariantEncodingMixin):
    """
    Original ShapeEmbed model.

    Used for:
        model_type = "original"

    Also used internally for:
        model_type = "mlp_decoder"
    except that the decoder object is replaced by MLPDistanceMatrixDecoder.

    Forward:
        D
        -> Frobenius normalization
        -> circular ResNet-18 encoder on D and flipped D
        -> sum features
        -> mu, logvar
        -> reparameterization
        -> decoder
        -> reconstructed distance matrix
    """

    def __init__(
        self,
        latent_dim: int,
        matrix_size: int,
        decoder_kind: Literal["original", "mlp"] = "original",
        normalize_input: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.matrix_size = matrix_size

        self.normalize_input = NormalizeFro() if normalize_input else nn.Identity()

        self.encoder = make_circular_resnet18_encoder()

        self.to_mu = nn.Linear(512, latent_dim)
        self.to_logvar = nn.Linear(512, latent_dim)

        if decoder_kind == "original":
            self.decoder = ShapeEmbedDecoder(
                latent_dim=latent_dim,
                matrix_size=matrix_size,
            )
        elif decoder_kind == "mlp":
            self.decoder = MLPDistanceMatrixDecoder(
                latent_dim=latent_dim,
                matrix_size=matrix_size,
            )
        else:
            raise ValueError(f"Unknown decoder_kind: {decoder_kind}")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor):
        x_pre = self.normalize_input(x)
        h = self.encode_with_reflection_invariance(x_pre)

        mu = self.to_mu(h)
        # Cap the posterior std (std = exp(0.5*logvar) <= exp(-1.5) ~= 0.22).
        # With BETA_KL ~ 0 nothing pushes logvar down, so on a small dataset the
        # model learns a large std (~0.9), the decoder is trained to be flat over
        # a wide latent neighbourhood and mode-collapses (washed-out recons).
        # Clamping the noise fixes that; existing checkpoints are unaffected at
        # eval time (we decode mu, not a sampled z). Re-train to take effect.
        logvar = torch.clamp(self.to_logvar(h), max=-3.0)

        return x_pre, mu, logvar

    def forward(self, x: torch.Tensor) -> dict:
        x_pre, mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        return {
            "x_pre": x_pre,
            "recon": recon,
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }


# =============================================================================
# Phase 2.1: AE variant
# =============================================================================

class ShapeEmbedAEVariant(nn.Module, ReflectionInvariantEncodingMixin):
    """
    Phase 2.1:
        Replace the variational part with a deterministic autoencoder.

    Changed:
        mu/logvar/sampling/KL are removed.

    Not changed:
        input representation
        Frobenius normalization
        circular ResNet-18 encoder
        original decoder path
        indexation-invariant reconstruction loss
        distance-matrix regularizers
    """

    def __init__(
        self,
        latent_dim: int,
        matrix_size: int,
        normalize_input: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.matrix_size = matrix_size

        self.normalize_input = NormalizeFro() if normalize_input else nn.Identity()

        self.encoder = make_circular_resnet18_encoder()
        self.to_z = nn.Linear(512, latent_dim)

        self.decoder = ShapeEmbedDecoder(
            latent_dim=latent_dim,
            matrix_size=matrix_size,
        )

    def encode(self, x: torch.Tensor):
        x_pre = self.normalize_input(x)
        h = self.encode_with_reflection_invariance(x_pre)
        z = self.to_z(h)

        return x_pre, z

    def forward(self, x: torch.Tensor) -> dict:
        x_pre, z = self.encode(x)
        recon = self.decoder(z)

        return {
            "x_pre": x_pre,
            "recon": recon,
            "z": z,
            "mu": None,
            "logvar": None,
        }


# =============================================================================
# Phase 2.2: MLP decoder variant
# =============================================================================

class ShapeEmbedMLPDecoderVariant(ShapeEmbedOriginal):
    """
    Phase 2.2:
        Replace the decoder path with an MLP.

    Changed:
        decoder path only.

    Not changed:
        Frobenius normalization
        circular ResNet-18 encoder
        reflection-invariant encoding
        mu/logvar/reparameterization
        KL loss
        indexation-invariant reconstruction loss
        distance-matrix regularizers
    """

    def __init__(
        self,
        latent_dim: int,
        matrix_size: int,
        normalize_input: bool = True,
    ):
        super().__init__(
            latent_dim=latent_dim,
            matrix_size=matrix_size,
            decoder_kind="mlp",
            normalize_input=normalize_input,
        )


# =============================================================================
# Factory
# =============================================================================

def build_model(
    model_type: str,
    latent_dim: int,
    matrix_size: int,
) -> nn.Module:
    """
    Use these names in config.py:

        MODEL_TYPE = "original"
            paper's original ShapeEmbed

        MODEL_TYPE = "ae"
            Phase 2.1 deterministic AE variant

        MODEL_TYPE = "mlp_decoder"
            Phase 2.2 decoder-replacement variant
    """

    if model_type == "original":
        return ShapeEmbedOriginal(
            latent_dim=latent_dim,
            matrix_size=matrix_size,
            decoder_kind="original",
            normalize_input=True,
        )

    if model_type == "ae":
        return ShapeEmbedAEVariant(
            latent_dim=latent_dim,
            matrix_size=matrix_size,
            normalize_input=True,
        )

    if model_type == "mlp_decoder":
        return ShapeEmbedMLPDecoderVariant(
            latent_dim=latent_dim,
            matrix_size=matrix_size,
            normalize_input=True,
        )

    raise ValueError(
        f"Unknown model_type={model_type}. "
        f"Use 'original', 'ae', or 'mlp_decoder'."
    )
