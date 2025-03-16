import torch
from tqdm import tqdm
from typing import Optional, Sequence, Tuple
import torch.nn as nn
import math


def infer_dimension(width: int, height: int, n_channels: int, model: nn.Module, batch_size: int = 8) -> torch.Tensor:
    """Compute the output of a model given a fake batch.

    Args:
        width: the width of the image to generate the fake batch
        height:  the height of the image to generate the fake batch
        n_channels:  the n_channels of the image to generate the fake batch
        model: the model to use to compute the output
        batch_size: batch size to use for the fake output

    Returns:
        the fake output
    """
    with torch.no_grad():
        fake_batch = torch.zeros([batch_size, n_channels, width, height])
        fake_out = model(fake_batch)
        return fake_out


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
    activation: nn.GELU,
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
                else activation(),
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
                activation(),
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


class AE_conv_MNIST(nn.Module):
    """
    Convolutional Autoencoder with a bottleneck of size latent_dim.
    
    This implementation accepts flattened MNIST images, reshapes them to (1,28,28),
    applies a conv encoder, projects the convolutional output to the latent space via a linear layer,
    and then reconstructs the image via the decoder.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim = 2,
        hidden_size = (32, 64, 128, 256),
        latent_activation = nn.GELU,
    ):
        super().__init__()
        # For MNIST we assume image size 28x28 and 1 channel.
        self.image_shape = (1, 28, 28)
        self.latent_dim = latent_dim

        # Build convolutional encoder-decoder.
        # Using nn.GELU as default activation for conv blocks.
        self.encoder, self.encoder_out_shape, self.decoder = build_dynamic_encoder_decoder(
            width=self.image_shape[1],
            height=self.image_shape[2],
            n_channels=self.image_shape[0],
            hidden_dims=hidden_size,
            activation=nn.GELU,
            remove_encoder_last_activation=False,
        )
        # encoder_out_shape is a tensor shape [batch, channels, height, width]
        # We need the number of features (ignoring the batch dimension).  
        encoder_out_numel = math.prod(self.encoder_out_shape)

        # Project conv features to latent space.
        self.encoder_out = nn.Sequential(
            nn.Linear(encoder_out_numel, latent_dim),
            latent_activation() if latent_activation is not None else nn.Identity(),
        )
        # Project latent vector back to conv features.
        self.decoder_in = nn.Sequential(
            nn.Linear(latent_dim, encoder_out_numel),
            latent_activation() if latent_activation is not None else nn.Identity(),
        )

    def encode(self, x):
        """
        Encodes a flattened input batch into the latent space.
        
        Args:
            x (Tensor): Input tensor with shape [batch_size, 784].
        Returns:
            z (Tensor): Latent vectors of shape [batch_size, latent_dim].
        """
        x = x.view(-1, *self.image_shape)  # Reshape from (batch, 784) to (batch, 1, 28, 28)
        conv_out = self.encoder(x)  # (batch, C, H, W)
        conv_out_flat = conv_out.view(conv_out.size(0), -1)
        z = self.encoder_out(conv_out_flat)
        return z

    def decode(self, z):
        """
        Decodes latent vectors back to the original flattened image space.
        
        Args:
            z (Tensor): Latent vectors of shape [batch_size, latent_dim].
        Returns:
            x_rec (Tensor): Reconstructed images with shape [batch_size, 784].
        """
        decoder_in = self.decoder_in(z)
        # Reshape to the convolutional feature map shape (use shape[1:] to exclude batch dimension)
        decoder_in_conv = decoder_in.view(-1, *self.encoder_out_shape[1:])
        x_rec = self.decoder(decoder_in_conv)
        # Flatten to (batch, 784)
        x_rec = x_rec.view(x_rec.size(0), -1)
        return x_rec

    def forward(self, x):
        """
        Forward pass: encode then decode.
        """
        return self.decode(self.encode(x))

    def train_one_epoch(self, train_loader, optimizer, criterion, device='cuda'):
        loss_total = 0.0
        self.train()
        for x, _ in train_loader:
            x = x.to(device)
            reconstructed = self.forward(x)
            loss = criterion(reconstructed, x)
            loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = loss_total / len(train_loader)
        return epoch_loss

    def evaluate(self, data_loader, criterion, device='cuda'):
        self.eval()
        loss_total = 0.0
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                reconstructed = self.forward(x)
                loss = criterion(reconstructed, x)
                loss_total += loss.item()
        eval_loss = loss_total / len(data_loader)
        return eval_loss

    def fit(self, train_loader, test_loader, num_epochs, lr=1e-3, device='cuda', verbose=True):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_function = nn.MSELoss()
        train_loss_list, test_loss_list = [], []
        for epoch in tqdm(range(num_epochs), desc="Training Epochs", disable=not verbose):
            train_loss = self.train_one_epoch(train_loader, optimizer, criterion=loss_function, device=device)
            train_loss_list.append(train_loss)
            test_loss = self.evaluate(test_loader, criterion=loss_function, device=device)
            test_loss_list.append(test_loss)
            if verbose:
                print(f'Epoch #{epoch}')
                print(f'Train Loss = {train_loss:.3e} --- Test Loss = {test_loss:.3e}')
        return train_loss_list, test_loss_list

    def get_latent_embeddings(self, data_loader, device='cpu'):
        embeddings = []
        indices = []
        labels = []
        self.eval()
        with torch.no_grad():
            for x, (idx, lab) in data_loader:
                x = x.to(device)
                z = self.encode(x)
                embeddings.append(z)
                indices.append(idx)
                labels.append(lab)
        embeddings_concat = torch.cat(embeddings, dim=0)
        indices_concat = torch.cat(indices, dim=0)
        labels_concat  = torch.cat(labels, dim=0)
        sorted_order = torch.argsort(indices_concat)
        embeddings_sorted = embeddings_concat[sorted_order]
        indices_sorted = indices_concat[sorted_order]
        labels_sorted = labels_concat[sorted_order]
        return embeddings_sorted, indices_sorted, labels_sorted

    def validate(self, data_loader, device='cuda'):
        self.eval()
        losses = []
        criterion = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                reconstructed = self.forward(x)
                loss = criterion(reconstructed, x).mean(dim=1)
                losses.append(loss)
        losses = torch.cat(losses)
        mse_mean = losses.mean().item()
        mse_std = losses.std().item()
        return mse_mean, mse_std