import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A model that computes histograms for each channel of an input array.
    
    The histogram counts how many times each value (in range [0, num_bins-1])
    appears in each channel of the input array.
    """

    def __init__(self, num_bins: int):
        super(Model, self).__init__()
        self.num_bins = num_bins

    def forward(self, array: torch.Tensor) -> torch.Tensor:
        """
        Computes histogram for each channel.
        
        Args:
            array: Input tensor of shape [length, num_channels] containing
                   integer values in the range [0, num_bins - 1]
        
        Returns:
            histogram: Tensor of shape [num_channels, num_bins], where
                      histogram[c][b] contains the count of how many times
                      value b appears in channel c.
        """
        length, num_channels = array.shape
        output_dtype = torch.int32
        histogram = torch.zeros(
            num_channels, self.num_bins, dtype=output_dtype, device=array.device
        )
        
        # Compute histogram for each channel
        for c in range(num_channels):
            channel_data = array[:, c]
            hist = torch.bincount(channel_data, minlength=self.num_bins)
            histogram[c] = hist[:self.num_bins]
        
        return histogram


# Problem configuration
length = 10000
num_channels = 64
num_bins = 256


def get_inputs():
    # Generate random integers in range [0, num_bins-1]
    array = torch.randint(
        low=0,
        high=num_bins,
        size=(length, num_channels),
        dtype=torch.uint8
    ).contiguous()
    return [array]


def get_init_inputs():
    return [num_bins]
