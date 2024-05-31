import torch
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
from .vmunet import VMUNet

__all__ = ["MEDIARMamba"]


class MEDIARMamba(nn.Module):
    """MEDIAR-Mamba Model"""

    def __init__(self, decoder_out_channels=24, ):  # Decoder configuration
        # Initialize the MAnet model with provided parameters
        super(MEDIARMamba, self).__init__()

        self.encoder_decoder = VMUNet()

        # Remove the default segmentation head as it's not used in this architecture
        self.segmentation_head = None

        # Modify all activation functions in the encoder and decoder from ReLU to Mish
        _convert_activations(self.encoder_decoder, nn.ReLU, nn.Mish(inplace=True))
        # _convert_activations(self.encoder, nn.ReLU, nn.Mish(inplace=True))
        # _convert_activations(self.decoder, nn.ReLU, nn.Mish(inplace=True))

        # Add custom segmentation heads for different segmentation tasks
        self.cellprob_head = DeepSegmentationHead(
            in_channels=decoder_out_channels, out_channels=1
        )
        self.gradflow_head = DeepSegmentationHead(
            in_channels=decoder_out_channels, out_channels=2
        )

  
    def forward(self, x):
        """Forward pass through the network"""
        # Ensure the input shape is correct
        # self.check_input_shape(x)

        # Encode the input and then decode it
        # features = self.encoder(x)
        # decoder_output = self.decoder(*features)
        decoder_output = self.encoder_decoder(x)
        # Generate masks for cell probability and gradient flows
        cellprob_mask = self.cellprob_head(decoder_output)
        gradflow_mask = self.gradflow_head(decoder_output)

        # Concatenate the masks for output
        masks = torch.cat([gradflow_mask, cellprob_mask], dim=1)

        return masks


class DeepSegmentationHead(nn.Sequential):
    """Custom segmentation head for generating specific masks"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        # Define a sequence of layers for the segmentation head
        layers = [
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(
                in_channels // 2,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity(),
            Activation(activation) if activation else nn.Identity(),
        ]
        super().__init__(*layers)


def _convert_activations(module, from_activation, to_activation):
    """Recursively convert activation functions in a module"""
    for name, child in module.named_children():
        if isinstance(child, from_activation):
            setattr(module, name, to_activation)
        else:
            _convert_activations(child, from_activation, to_activation)
