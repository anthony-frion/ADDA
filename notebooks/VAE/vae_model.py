import torch

torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F


class Conv_Encoder1D(nn.Module):
    """1D convolutional encoder for a variational autoencoder.

    Args:
        input_channels: Number of channels in the input sequence.
        feature_list: Convolution channel sizes for each encoder stage.
        latent_dim: Size of the latent vector produced by the encoder.
        input_length: Length of the input sequence used to calculate the size of the linear layers.
    """

    def __init__(
        self,
        input_channels=1,
        feature_list=(16, 32, 64),
        latent_dim=16,
        input_length=200,
    ):
        super(Conv_Encoder1D, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.feature_list = feature_list
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        for out_channels in feature_list:
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=6,
                    padding=2,
                    stride=2,
                    dilation=1,
                    bias=True,
                    padding_mode="circular",
                )
            )
            self.conv_layers.append(nn.GroupNorm(4, out_channels))
            in_channels = out_channels
        self.activation = nn.GELU()
        self.mu = nn.Linear(feature_list[-1] * (input_length // (2 ** len(feature_list))), latent_dim)
        self.log_var = nn.Linear(feature_list[-1] * (input_length // (2 ** len(feature_list))), latent_dim)

    def forward(self, x):
        """Encode an input batch into mean and log-variance vectors.

        Args:
            x: Tensor with shape ``(batch, channels, length)``.

        Returns:
            A tuple ``(mu, log_var)`` each with shape ``(batch, latent_dim)``.
        """
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.GroupNorm):
                x = self.activation(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class Conv_Decoder1D(nn.Module):
    """1D convolutional decoder for a variational autoencoder.

    Args:
        output_channels: Number of channels in the reconstructed sequence.
        feature_list: Deconvolution channel sizes ordered from latent side
            toward output side.
        latent_dim: Size of the latent vector consumed by the decoder.
        output_length: Length of the reconstructed output sequence.
    """

    def __init__(
        self,
        output_channels=1,
        feature_list=(64, 32, 16),
        latent_dim=16,
        output_length=200,
    ):
        super(Conv_Decoder1D, self).__init__()
        self.output_channels = output_channels
        self.latent_dim = latent_dim
        self.feature_list = feature_list
        self.fc = nn.Linear(latent_dim, feature_list[0] * (output_length // (2 ** len(feature_list))))
        self.deconv_layers = nn.ModuleList()
        in_channels = feature_list[0]
        self.prpd = 2
        for out_channels in feature_list[1:] + [output_channels]:
            self.deconv_layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=6,
                    padding=6,
                    stride=2,
                    bias=True,
                )
            )
            if out_channels != output_channels:
                self.deconv_layers.append(nn.GroupNorm(4, out_channels))
            in_channels = out_channels
        self.activation = nn.GELU()

    def forward(self, z):
        """Decode latent vectors into reconstructed 1D sequences.

        Args:
            z: Latent tensor with shape ``(batch, latent_dim)``.

        Returns:
            Reconstructed tensor with shape ``(batch, output_channels, length)``.
        """
        x = self.fc(z)
        x = x.view(x.size(0), self.feature_list[0], -1)
        for i, layer in enumerate(self.deconv_layers):
            if isinstance(layer, nn.ConvTranspose1d):
                x = F.pad(x, pad=(self.prpd, self.prpd), mode="circular")
            x = layer(x)
            if isinstance(layer, nn.GroupNorm):
                x = self.activation(x)
        return x


class VAE(nn.Module):
    """Variational autoencoder composed of 1D convolutional encoder/decoder.

    Args:
        pretrained_checkpoint_path: Checkpoint path. Weights are loaded in ``__init__``.
        device: Device used for checkpoint loading and module placement.
    """

    def __init__(
        self,
        pretrained_checkpoint_path=None,
        device="cpu",
    ):
        super(VAE, self).__init__()
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        input_channels = 1
        feature_list = [32, 32, 32]
        latent_dim = 32
        input_length = 128

        self.encoder = Conv_Encoder1D(input_channels, feature_list, latent_dim, input_length)
        self.decoder = Conv_Decoder1D(input_channels, feature_list[::-1], latent_dim, input_length)

        checkpoint = torch.load(self.pretrained_checkpoint_path, map_location=torch.device(device), weights_only=True)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        self.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint from {self.pretrained_checkpoint_path}")
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)
        self.eval()

    def reparameterize(self, mu, log_var):
        """Sample from latent distribution using the reparameterization trick.

        Args:
            mu: Mean tensor of the latent Gaussian.
            log_var: Log-variance tensor of the latent Gaussian.

        Returns:
            A latent sample tensor with the same shape as ``mu``.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Run full VAE pass: encode, sample, and decode.

        Args:
            x: Input tensor with shape ``(batch, channels, length)``.

        Returns:
            A tuple ``(reconstruction, mu, log_var)``.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
