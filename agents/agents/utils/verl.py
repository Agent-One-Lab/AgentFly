import torch

def pad_tensor_to_rank_size(tensor: torch.Tensor, rank_size: int) -> torch.Tensor:
    """
    verl DP Proto requires the batch size to be divisible by the DP size.
    This function pads the tensor to be divisible by the DP size using last row of the tensor.
    """
    pad_size = (rank_size - tensor.shape[0] % rank_size) % rank_size
    if pad_size == 0:
        return tensor
    else:
        last_row = tensor[-1, :]
        padded_tensor = torch.cat([tensor, last_row.repeat(pad_size, 1)], dim=0)
        return padded_tensor
