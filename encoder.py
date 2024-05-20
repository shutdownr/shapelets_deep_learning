import torch

# Utility modules

class CausalTruncation(torch.nn.Module):
    """
    Truncation utility module for causal convolutions

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `P`), where `P`=L-padding.

    @param padding Padding to truncate 
    """
    def __init__(self, padding):
        super(CausalTruncation, self).__init__()
        self.padding = padding

    def forward(self, x):
        return x[:,:,:-self.padding]
    
class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)
    
# Causal Convolutions

class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConvolutionBlock, self).__init__()
        padding = dilation*(kernel_size-1)

        causal_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        weight_norm = torch.nn.utils.parametrizations.weight_norm(causal_conv)
        truncate = CausalTruncation(padding)
        # Using a ReLU here instead of a LeakyReLU
        relu = torch.nn.ReLU()

        causal_conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        weight_norm2 = torch.nn.utils.parametrizations.weight_norm(causal_conv2)
        truncate2 = CausalTruncation(padding)
        relu2 = torch.nn.ReLU()

        self.causal_conv = torch.nn.Sequential(weight_norm, truncate, relu,
                                                weight_norm2, truncate2, relu2)
        if in_channels != out_channels:
            self.residual_resample = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_resample = None
        
    def forward(self, x):
        conv = self.causal_conv(x)
        if self.residual_resample:
            out = self.residual_resample(x) + conv
        else:
            out = x + conv

        return out

class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        # First layer
        dilation = 1
        initial_conv_block = CausalConvolutionBlock(in_channels, channels, kernel_size, dilation)
        conv_blocks = [initial_conv_block]
        dilation *= 2

        # Intermediate layers
        for _ in range(1, depth-1):
            causal_conv_block = CausalConvolutionBlock(channels, channels, kernel_size, dilation)
            conv_blocks.append(causal_conv_block)
            dilation *= 2

        # Last layer
        final_conv_block = CausalConvolutionBlock(channels, out_channels, kernel_size, dilation)
        conv_blocks.append(final_conv_block)

        self.layers = torch.nn.Sequential(*conv_blocks)

    def forward(self, x):
        return self.layers(x)

# Encoder

class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        max_pooling = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, max_pooling, squeeze, linear
        )

    def forward(self, x):
        return self.network(x)
