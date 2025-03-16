import logging
from typing import Optional, Sequence, Tuple

import hydra.utils
import torch
from torch import nn

from .tensor_ops import infer_dimension

pylogger = logging.getLogger(__name__)

def build_transposed_convolution(
    in_channels: int,
    out_channels: int,
    target_output_width,
    target_output_height,
    input_width,
    input_height,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (1, 1),
    output_padding: Tuple[int, int] = (0, 0),
    dilation: int = 1,
) -> nn.ConvTranspose2d:
    # kernel_w = (metadata.width - (fake_out.width −1)×stride[0] + 2×padding[0] - output_padding[0]  - 1)/dilation[0] + 1
    # kernel_h = (metadata.height - (fake_out.height −1)×stride[1] + 2×padding[1] - output_padding[1]  - 1)/dilation[1] + 1
    kernel_w = (
        target_output_width - (input_width - 1) * stride[0] + 2 * padding[0] - output_padding[0] - 1
    ) / dilation + 1
    kernel_h = (
        target_output_height - (input_height - 1) * stride[1] + 2 * padding[1] - output_padding[1] - 1
    ) / dilation + 1
    assert kernel_w > 0 and kernel_h > 0

    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(int(kernel_w), int(kernel_h)),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )


def build_dynamic_encoder_decoder(
    width,
    height,
    n_channels,
    hidden_dims: Optional[Sequence[int]],
    activation: str = "torch.nn.GELU",
    remove_encoder_last_activation: bool = False,
) -> Tuple[nn.Module, Sequence[int], nn.Module]:
    """Builds a dynamic convolutional encoder-decoder pair with parametrized hidden dimensions number and size.

    Args:
        width: the width of the images to work with
        height: the height of the images
        n_channels: the number of channels of the images
        hidden_dims: a sequence of ints to specify the number and size of the hidden layers in the encoder and decoder

    Returns:
        the encoder, the shape in the latent space, the decoder
    """
    modules = []

    if hidden_dims is None:
        hidden_dims = (32, 64, 128, 256)

    STRIDE = (2, 2)
    PADDING = (1, 1)

    # Build Encoder
    encoder_shape_sequence = [
        [width, height],
    ]
    running_channels = n_channels
    for i, h_dim in enumerate(hidden_dims):
        modules.append(
            nn.Sequential(
                (
                    conv2d := nn.Conv2d(
                        running_channels, out_channels=h_dim, kernel_size=3, stride=STRIDE, padding=PADDING
                    )
                ),
                nn.BatchNorm2d(h_dim),
                nn.Identity()
                if i == len(hidden_dims) - 1 and remove_encoder_last_activation
                else hydra.utils.instantiate({"_target_": activation}),
            )
        )
        conv2d_out = infer_dimension(
            encoder_shape_sequence[-1][0],
            encoder_shape_sequence[-1][1],
            running_channels,
            conv2d,
        )
        encoder_shape_sequence.append([conv2d_out.shape[2], conv2d_out.shape[3]])
        running_channels = h_dim

    encoder = nn.Sequential(*modules)

    encoder_out_shape = infer_dimension(width, height, n_channels=n_channels, model=encoder, batch_size=1).shape

    # Build Decoder
    hidden_dims = list(reversed(hidden_dims))
    hidden_dims = hidden_dims + hidden_dims[-1:]

    running_input_width = encoder_out_shape[2]
    running_input_height = encoder_out_shape[3]
    modules = []
    for i, (target_output_width, target_output_height) in zip(
        range(len(hidden_dims) - 1), reversed(encoder_shape_sequence[:-1])
    ):
        modules.append(
            nn.Sequential(
                build_transposed_convolution(
                    in_channels=hidden_dims[i],
                    out_channels=hidden_dims[i + 1],
                    target_output_width=target_output_width,
                    target_output_height=target_output_height,
                    input_width=running_input_width,
                    input_height=running_input_height,
                    stride=STRIDE,
                    padding=PADDING,
                ),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                hydra.utils.instantiate({"_target_": activation}),
            )
        )
        running_input_width = target_output_width
        running_input_height = target_output_height

    decoder = nn.Sequential(
        *modules,
        nn.Sequential(
            nn.Conv2d(hidden_dims[-1], out_channels=n_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ),
    )
    return encoder, encoder_out_shape, decoder
